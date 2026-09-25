# Four Sparks (TP4): the production line

A measured TP4 serving line for four DGX Spark (GB10) nodes, built on this repository's launcher
(`start-tp4.sh` → `start.sh`), image layout, NVMe Engram row store, b12x MXFP8 route and memory
model. Everything it adds is armed only by `./start-tp4.sh`, which sets `DSV41_LAUNCHER=tp4`
before it execs `start.sh`. That flag is what turns on the named Dockerfile, `EXTRA_CONTAINER_ENV`,
the anchored worker rsync, EP 1 or 2 on the ring, and the production adapter hooks. The switches
still do nothing until the profile sets them. `./start.sh` leaves the flag unset, so the 3-Spark
profile (`.env`, `start.sh`) and the default `Dockerfile` keep their behaviour.

The per-change history, every earlier measurement and the experiments that were tried and not
adopted are in [knapcio/DeepSeek-V4.1-Flash-4x-DGX-Spark-TP4](https://github.com/knapcio/DeepSeek-V4.1-Flash-4x-DGX-Spark-TP4)
(`docs/history.md`, `CHANGELOG.md`). This page is the v2.2 state of that repository.

## Results

Production line (the last `EXTRA_CONTAINER_ENV` line of `.env.tp4.example` with `EP_SIZE=1`,
`Dockerfile.canary-roce` image), built from a fresh clone on all four nodes and measured on that
image with [sparkDash](https://github.com/MiaAI-Lab/sparkDash) 1.8.8: 256 new tokens, temperature 0,
thinking off, idle fleet, 2026-09-25, switched RoCE fabric. Two boots, healthy in 180 s and
160 s; KV pool 5.98-6.49M tokens (1M context). Raw output:
[`results/tp4/validation-20260925-v22.txt`](results/tp4/validation-20260925-v22.txt) (v2.2),
[`results/tp4/validation-20260925-v21.txt`](results/tp4/validation-20260925-v21.txt) (v2.1),
[`results/tp4/validation-20260925-v2.txt`](results/tp4/validation-20260925-v2.txt) (v2, the c2/c8 columns).

**Decode, aggregate tok/s (per stream in brackets)**

| prompt type | c1 | c2 † | c4 | c8 † | c16 |
|---|---:|---:|---:|---:|---:|
| prose | 89.8 | 120.4 (61.9) | 164.1 (41.5) | 237.6 (30.8) | 340.9 (22.2) |
| code | 132.4 | 175.0 (88.4) | 252.7 (64.1) | 309.8 (41.4) | 436.4 (29.2) |
| structured | 156.9 | 177.6 (103.3) | 239.6 (70.2) | 295.5 (44.2) | 565.5 (44.3) |
| json | 124.3 | 174.2 (89.8) | 311.2 (78.7) | 471.5 (60.2) | 653.0 (42.2) |

† c2 and c8 were not re-measured for v2.2; they are the v2 sweep's values. The c1, c4 and c16 columns
are from the v2.2 fresh-clone boot. Prose c1 is the median of five runs, 89.76 (88.97-90.21), after two
discarded warm-ups; the second boot gave 89.04 / 90.42 / 89.47. Code c1 is the median of three runs
(124.95 / 132.62 / 132.37). v2.1 measured prose c1 87.7 and code c1 124.8 on the same prompts, and
greedy output is byte-identical to v2.1's (hash check of 15 outputs on the first boot and 6 on the
second). With the deterministic MoE reduction the greedy text is identical run to run, so the sparkDash numbers repeat
within about ±1 tok/s. sparkDash's prose c1 is one prompt, and a stack that sums in a different order
(another fabric, another all-reduce) follows a different greedy text there, so compare step time or a
many-prompt benchmark across stacks. On 45 varied prompts (prose, structured and other catalogs, c1
greedy) the same image runs 61.1 / 99.0 / 74.7 tok/s (v2.1: 58.4 / 94.3 / 71.8); decode step
31.2-32.5 ms on prose and 37.4-37.8 ms on code at c1 (v2.1: 32.4-33.6 / 38.9-39.4). sparkDash uses a different set of prompts at each concurrency for the
non-prose types, so per-stream values are not comparable across columns. Sampled chat at the model
card's T=1 / top_p=0.95 with thinking (c1, 18 requests x 800 tokens on two prompt sets) ran 67.2 /
65.9 tok/s on the 2026-09-24 stack (sparkDash benches are greedy, where the draft temperature and
block verification do not act).

**Prefill, cold, tok/s by prompt length** (v2.2 one pass / v2.1 two passes; v2.2 changes no prefill
path that is on by default)

| | 4k | 16k | 32k | 64k | 128k | 262k |
|---|---:|---:|---:|---:|---:|---:|
| v2.2 | 3944 | 5789 | 5855 | 5850 | 5734 | 5286 |
| v2.1 | 4119 / 4840 | 5891 / 5846 | 5893 / 5868 | 5936 / 5903 | 5793 / 5761 | 5355 / 5360 |

sparkDash's prefill filler is one repeated token, so every filler token hits the same Engram row and
the row cache inflates these numbers (reported by koldfrontier in
[#21](https://github.com/MiaAI-Lab/DeepSeek-v4.1-Flash-DGX-Sparks/issues/21)). On real text
(documentation and source code, a unique prefix per prompt so nothing comes from the prefix cache) the
v2.2 image measured 3915 / 4743 / 4822 / 4946 / 4800 tok/s at ~4k / ~15k / ~30k / ~56k / ~123k tokens
(v2.1: 4202-4685 / 4824-4922 / 4827-4830 / 4942-4970 / 4822-4871 at ~4k / ~15k / ~32k / ~62k / ~120k).
Needle retrieval (a single phrase in varied filler at 37 % depth): PASS at 1,011,084 tokens on v2.2
(324.3 s, head `MemAvailable` low-water 6,029 MiB); earlier stacks passed at 99k, 198k and 746k as well.

**Quality.** `scripts/qeval.py` runs 75 auto-scored tasks (code executed against hidden asserts, JSON
schema-checked, numeric answers matched, format constraints enforced, prose checked for degeneration;
no LLM judge), one request at a time, temperature 0. The v2.1 fresh-clone image scores 72 of 75 (not re-run
for v2.2, whose greedy output is byte-identical to v2.1's); the three
misses (`code_interval_intersect`, `json_escape`, `math_m9`) fail identically on the stock profile.
Every speed change is meant to be lossless: same weights, every draft token verified by the target,
and each adapter either bit-identical to the stock path (checked at boot or in the in-image tests) or
exact in distribution (draft temperature, block verification). RoCEnante and prefill sequence
parallel sum in a different order than NCCL's all-reduce, so the numerics are not bit-identical to an
NCCL run, which is why the line is also scored. Run qeval from a worker, not from the head (it executes
model-generated Python).

## What runs

| Layer | Setting | What it does |
|---|---|---|
| Image | `Dockerfile.canary-roce` | upstream SGLang `dsv4.1` branch at `f80c91a4b` + rhys101's RoCEnante overlay adapted to TP4 + b12x main as `b12x_next` + all adapters; third-party sources fetched at build (`runtime/sources.manifest`) |
| Slots | `MAX_RUNNING_REQUESTS=16` | adds the c16 tier |
| Experts | `EP_SIZE=1`, `DSV41_MOE_B12X_NEXT=1`, `DSV41_MOE_B12X_NEXT_DETERMINISTIC=1` | every rank holds a quarter of all 384 experts and the routed MoE runs on b12x main: no expert-group straggler at the MoE all-reduce (all-reduce 5.9 → ~2.6 ms per step), MoE time unchanged. The deterministic reduction (per-slot buffer and a fixed-order sum instead of atomics) makes greedy output bit-identical run to run; its <= 8-row plans use a Triton route planner and zero the barrier words in one fill (bit-identical to b12x's internal planner; `DSV41_MOE_B12X_NEXT_DET_TRITON`, default 1) |
| Engram | `DSV41_CACHE_GIB=4`, `DSV41_CACHE_WAYS=16`, `DSV41_ENGRAM_PREFETCH=1` | row cache (67–76 % hits) and row lookups on a side stream, rows bit-identical |
| Draft | `DSPARK_BLOCK_SIZE=5`, `SGLANG_DSPARK_FOLDED_SAMPLING=2` | k=5 wins on code, ties on prose at TP4; sampled requests stay in the CUDA graph |
| Draft sampling | `DSV41_DRAFT_TAU=0.7`, `DSV41_BLOCK_VERIFY=1` | sharper draft proposals and block verification for sampled rows, both exact in distribution |
| Verify length | `DSV41_VERIFY_CAP=conf:0.1`, `DSV41_ROUTER_LIVE=1` | verifies only the drafts the confidence head expects to survive; the other verify rows reuse the anchor row's experts inside the router kernel, so they read no extra weights; greedy outputs identical |
| Kernels | `DSV41_SHARED_PAD_K=1`, `DSV41_WO_A_W8=1`, `DSV41_WO_A_W8_MID=1`, `DSV41_WO_A_W8_DROP=1`, `DSV41_DRAFT_HEAD_FP8=1`, `DSV41_DRAFT_HEAD_FP8_IMPL=tp4` | shared expert kept on the b12x kernel; `wo_a` and the draft LM head read exact fp8 twins of their weights |
| Replicated linears | `DSV41_REPLICATED_SPLIT=wqkv_a,engram.wkv`, `DSV41_DRAFT_MAIN_PROJ_SPLIT=1`, `DSV41_SPLIT_COMPACT_GATHER=1` | layers every rank computed in full are split by columns across ranks and all-gathered; enabled only where bit-identical on all ranks. A rank with a narrower slice runs its GEMM on a window of the widest slice's width (`DSV41_REPLICATED_SPLIT_WINDOW`, default 1), so no pad precedes the gather, and with the compact gather each rank sends only its own columns, which RoCEnante writes in place (no pad, reorder or cat; checked byte for byte against the old path per layer and row count at boot) |
| Transport | `SGLANG_ROCE_ALLREDUCE=1`, `SGLANG_ROCE_MAX_SIZE=2097152`, `DSV41_ROCE_GATHER=2097152`, `B12X_ROCE_HCA=rocep1s0f0,roceP2p1s0f0` | TP all-reduces and all-gathers up to 2 MiB over RoCEnante's one-shot RDMA kernel on both rails |
| Prefill | `CHUNKED_PREFILL_SIZE=4096`, `DSV41_INDEXER_CHUNKED=1`, `SPARK_PREFILL_TP_SPLIT=1` | bounded indexer transient (sglang#39187) plus the SG18 row split across ranks from 32k context |
| Loading | `DSV41_FAST_LOAD=1`, `DSV41_FAST_LOAD_TP_SLICE=auto`, `DSV41_AUTOTUNE_KEEP=1` | engine start 343 s → ~120 s; at EP1 each rank reads only its slice of every expert, into 256 MiB pinned slabs; MoE autotune cache kept across boots ([fast-load.md](fast-load.md)) |
| Prefill hc | `DSV41_HC_FUSED=1` | the hyper-connection mix statistics of prefill chunks in one pass over K instead of 80 partial slices: ~1.46 → ~0.83 ms per call at 4096 rows, bit-identical (checked on the first live call) |
| Prefill sequence parallel | `DSV41_PREFILL_SP=1`, `DSV41_PREFILL_SP_FP8=1` | at prefill chunks of >= 2048 rows the per-layer all-reduces become reduce-scatter + all-gather and the per-row work between them runs on each rank's quarter of the rows: prefill ~+20 % from 16k to 262k, decode untouched. `DSV41_PREFILL_SP_EXACT=1` keeps the all-reduce (bit-identical to the unsharded path) for ~+10 %; `_FP8` gathers the attention input as the MXFP8 bytes `wqkv_a` would compute itself (bit-identical per chunk size), +1.5-3 %. `DSV41_PREFILL_SP_FP8_MOE=1` (off, not in the line) gathers the MoE input as MXFP8 too: bit-exact, but flat on real-text prefill |
| L2 prefetch | `DSV41_L2_PREFETCH=1`, `DSV41_L2_PREFETCH_WOA=1` | during each RoCE collective of a decode step a side-stream kernel prefetches the first 6 MB of the weights that follow into L2; data untouched; decode step -1 ms at c1. `WOA` adds a window after `wq_b` that prefetches `wo_a` while the attention core runs: -0.4 ms/step |
| Decode-step sync | `DSV41_SPEC_SYNC_FREE=all` | drops the per-step rank-0 broadcasts whose values are already identical on every rank (the draft token of each Markov step, the verify epilogue's three, verify_cap's live length); the draft noise comes from a counter-based stream every rank computes alike. Audit mode counted 0 rank mismatches in ~60k checks; greedy output byte-identical; -0.53 to -0.59 ms/step in an in-boot A/B (~-0.2 ms across reboots) |
| Decode-step glue | `DSV41_EAGER_GLUE=all` | fewer eager kernels between the draft and verify CUDA graphs (the folded fence's six copies as one, sampling-param staging skipped while unchanged, verify_cap's update inside the draft graph as one kernel); bit-identical, ~-0.1 to -0.2 ms/step |
| Correctness | `DSV41_FOLDED_FENCE=1` | closes the sglang#40919 D2H race on the folded verify path |
| Serving | `--enable-cache-report`, `--sleep-on-idle`, `SGLANG_RUST_BUILD_MODE=never` | cached-token usage for clients, idle CPU 47 % → 14 %, avoids a `cargo` hang at start on the branch images |
| Optional | `DSV41_ENGRAM_DRM_NODE=/dev/dri/card0` | one Engram layer's cache in the GB10 display reservation, ~1.8 GiB more headroom; needs a host change ([display-reserve.md](display-reserve.md)) |

Each adapter is described in [tp4-adapters.md](tp4-adapters.md); the image layers on their own and the
line on a switchless ring are in [tp4-optional-setups.md](tp4-optional-setups.md).

## What the profile changes

Against the earlier TP4 example, all of it in `.env.tp4.example` plus gated adapters:

| Setting | Before | Now | Why (measured) |
|---|---|---|---|
| `EP_SIZE` | 4 | **1** (production), 2 (base and canary images) | On the production image every rank streams a 576-wide slice of every touched expert, so no expert group waits for another at the MoE all-reduce. The base and canary images stay at `EP_SIZE=2` (FlashInfer's MXFP4 MoE needs the per-rank width to be a multiple of 128), which already halves the EP4 straggler wait: NCCL time per step 16.5 → 10.2 ms |
| `DSV41_CACHE_GIB` / `WAYS` | 0 / 4 | **4 / 16** | Engram rows repeat (bigram/trigram heads): 67–76 % hit rate, 4x fewer NVMe reads |
| `DSPARK_BLOCK_SIZE` | 3 | **5** | on TP4 k=5 wins on code by ~10 % and ties on prose |
| `MAX_RUNNING_REQUESTS` | 8 | **16** | CUDA graphs to bs 16 cost ~5.6 GB and add a c16 tier (+64 % aggregate over c8) |
| `CHUNKED_PREFILL_SIZE` | 1024 | **4096** | safe only together with the indexer backport below |
| `DSV41_SHARED_PAD_K` | – | **1** | pads the shared expert's K 576 → 640 so it stays on the b12x MXFP8 kernel: −0.9 ms/step, bit-identical |
| `DSV41_INDEXER_CHUNKED` | – | **1** | sglang#39187: indexer logits scored in ≤ 2 GiB row chunks, tail-only candidate masks; a 262k cold prefill keeps ≥ 7 GiB free on the head |
| `SPARK_PREFILL_TP_SPLIT` (canary images) | – | **1** | SG18: the dense prefill indexer's query rows are partitioned across the four ranks from 32k context |
| `DSV41_SERIAL_WEIGHT_LOAD` | 1 | commented, in the ring block | the switched line was measured with the default loader and the fast loader's own pacing |
| `--enable-deepseek-v4-fp4-indexer` | off | on | no effect on V4.1 (no ratio-4 layers; its only indexer is fp4 by design); kept so the argument line matches the measured runs |

Unchanged: memory fraction 0.80, the 8M-token KV pin, 1M context, the NFS/Engram layout, the OpenAI
serving fixes, `--min-free-slots-delay 1`.

## How to run

```bash
cp .env.tp4.example .env.tp4          # HEAD_IP / WORKER_* / MODEL_DIR / fabric, as for any TP4 profile
# production: in .env.tp4 uncomment IMAGE=dsv41-4x-spark:canary-roce, BUILD_DOCKERFILE=Dockerfile.canary-roce
# and the last EXTRA_CONTAINER_ENV line, and set EP_SIZE=1 (the file's EP_SIZE=2 is for the base/canary images)
./start-tp4.sh doctor
./start-tp4.sh build                  # every node: fetches the pinned sources, bakes the adapters, runs the in-image tests
./start-tp4.sh share && ./start-tp4.sh pack   # first time only (NFS share, Engram shards)
./start-tp4.sh serve                  # ./start-tp4.sh stop | status | logs | smoke
```

`build` needs github.com during the image's `fetch` stage (and pypi.org, or `BUILD_ARGS="--build-arg
PIP_INDEX=<mirror>/simple"`). The fetched trees can be inspected on a host with
`scripts/fetch_runtime.sh runtime/fetched`.

The boot log must show these lines, otherwise the line is not active:

```
DSV41 shared-expert padding K: ... (5120, 576) -> (5120, 640)
DSV41 indexer chunked (sglang#39187 backport) ARMED: ...
Initialized DSpark draft runner. ... gamma=5, verify_num_draft_tokens=6
max_total_num_tokens=..., chunked_prefill_size=4096, ... max_running_requests=16
RoCEnante ready: world=4 hcas=...
[moe_b12x_next] armed: routed MoE on b12x_next a7d7d29b ...
[moe_b12x_next] INFO: routed MoE at EP_SIZE=1 ...
DSV41 hc_fused: first call (4096 rows) bit-identical to stock
DSV41_L2_PREFETCH: model forward bracketed (decode/verify only); v3: wo_a windows (6.0 MB)
[spec_sync_free] armed (all): skip draft=True accept=True vcap=True ...
[eager_glue] vcap: verify_cap live length captured into the draft graph ...
[replicated_split] model.layers.0.self_attn.wqkv_a: compact gather (roce) bit-identical, ON ...
```

## Measuring

- Bench with sparkDash's decode and prefill benches; discard the first two runs after a boot (cold
  Engram row cache, first-run warm-up).
- Any `#running-req` in the engine log above the concurrency being benched means foreign traffic
  landed in the window.
- A dashboard polling `nvidia-smi` every 2 s on every node costs ~0.6 ms per decode step; poll every
  10 s.
- sparkDash's fixed prompts are greedy: a change that alters rounding can flip a near-tie early in the
  256 tokens and move c1 by ±10 % without changing speed. Kernel A/Bs use step time on several prompts
  as well.
- For decode-path A/Bs without a reboot, `DSV41_AB_VARIANTS` (test only, `adapter/ab_variant.py`,
  driven by `scripts/ab_inboot.py`) captures one CUDA graph set per flag variant and switches between
  them at runtime on every rank at once; it roughly doubles decode graph memory, so never in
  production. `scripts/repeat_sha.py` hashes greedy outputs to check repeatability across runs and
  stacks.
- Greedy text equality is not a correctness gate for prefill changes: identical cold prompts of ~70k
  tokens produce different greedy continuations run to run. Correctness of the backports rests on the
  bitwise CPU tests, the needle tests and qeval.

## Rollback

Every layer is an env change: `EP_SIZE=2` without `DSV41_MOE_B12X_NEXT` returns the routed MoE to
FlashInfer, `DSV41_FAST_LOAD=0` restores the stock loader, `SGLANG_ROCE_ALLREDUCE=0` drops the RDMA
transport, `SPARK_PREFILL_TP_SPLIT=0` the row split, `IMAGE=dsv41-4x-spark:canary` the RoCEnante
overlay, `IMAGE=dsv41-4x-spark:local` the branch; unsetting `DSV41_SPEC_SYNC_FREE`, `DSV41_EAGER_GLUE`,
`DSV41_SPLIT_COMPACT_GATHER` and `DSV41_L2_PREFETCH_WOA` turns off the v2.2 decode-step changes
(`DSV41_REPLICATED_SPLIT_WINDOW=0` and `DSV41_MOE_B12X_NEXT_DET_TRITON=0` the two that are on by
default). The earlier TP4 example: `MAX_RUNNING_REQUESTS=8`,
`CHUNKED_PREFILL_SIZE=1024`, `DSPARK_BLOCK_SIZE=3`, `EXTRA_CONTAINER_ENV=""`, `EP_SIZE=4`,
`DSV41_CACHE_GIB=0`. The adapters stay in the image but do nothing when their gate is unset.

## Known limits

- The numbers above are on a switched RoCE fabric. On a switchless ring RoCEnante needs a path to the
  opposite node, built in the neighbours' ConnectX-7 hardware
  ([switchless-ring.md](switchless-ring.md#rocenante-on-the-ring-hardware-forwarded-opposite-node-paths));
  without it the collectives go through NCCL and decode is slower. Prefill is lower on a ring either
  way (one-link bisection).
- The EP1 line needs the `Dockerfile.canary-roce` image (it carries `b12x_next`). The routed MoE on
  b12x is numerically equivalent to FlashInfer's, not bit-identical (relative error against an fp32
  reference 4.5–4.8 % for both); qeval and the needle tests are unchanged.
- Single-stream prose speed is bounded by DSpark acceptance (~3 accepted tokens per step on prose
  against ~6 on code).
- The fast loader leaves the KV pool smaller than the stock loader and more variable between boots;
  `DSV41_FAST_LOAD=0` restores it at ~220 s per boot ([fast-load.md](fast-load.md)).
- RoCEnante is a new transport in the decode path; b12x has one open report of a rank wedging under
  long mixed-context traffic on an earlier revision
  ([b12x#313](https://github.com/local-inference-lab/b12x/issues/313)). The overlay's result-boundary
  health check fails the step instead of hanging, and NCCL is one env change away.

## Credits

- **Mia (MiaAI-Lab)**, this repository: the launcher, `boot.py`, the Engram NVMe row store and packed
  shards, the MXFP8 b12x routing, the chunked-prefill memory model, the thinking alias, the output cap,
  the loop abort, the TP4 profile this line started from, and sparkDash, the benchmark behind every
  number here.
- **local-inference-lab / Luke Alonso and Jason (original-el8)**, [b12x](https://github.com/local-inference-lab/b12x):
  RoCEnante, the one-shot RDMA all-reduce (the SG17 revision), and the fused MoE kernels that run the
  routed experts (b12x main at `a7d7d29b`, installed as `b12x_next` beside the SG17 copy, with four
  small patches in `scripts/`: 64-row tiles for 576-wide experts at prefill sizes, pre-quantized input
  launches, one barrier fill, the Triton route planner under deterministic output). The compact
  gather adds a columns all-gather to the SG17 RoCEnante copy (`scripts/b12x-roce-columns-gather.patch`).
- **rhys101**, [DeepSeek-V4.1-Flash-vLLM-DGX-Spark-8](https://github.com/rhys101/DeepSeek-V4.1-Flash-vLLM-DGX-Spark-8):
  the SG17 SGLang overlay that routes small tensor-parallel all-reduces to RoCEnante (with a TP4
  adaptation in `runtime/roce_tp4_adapt.py`), and the SG18 native prefill TP split
  (`adapter/spark_prefill_dense.py`, combined with the indexer backport in `adapter/indexer_chunked_v3.py`).
- **luxingcom (LuZ)**, [LuZ DGX Spark TP4 ring](https://github.com/luxingcom/LuZ-0.1.7-DeepSeek-v4.1-Flash-DGXspark-TP4-Ring):
  the first integration of b12x's fused MoE into SGLang on a four-Spark fleet, which showed the route.
- **sumsliu**, [dgx-spark-deepseek-v41](https://github.com/sumsliu/dgx-spark-deepseek-v41): the
  eight-Spark measurement that moving to expert tensor parallelism removes most of the all-reduce wait.
- **FujitsuPolycom and the sparkring contributors**, [sparkring](https://github.com/FujitsuPolycom/sparkring),
  and **rsync (@rchmagos)**: the hardware-forwarded opposite-node paths and the path-aware RoCEnante
  (`b12x.comm.roce_ring`) that run the line on a switchless ring; the NCCL transport patch for the ring.
- **@Saolence**: the switchless-ring launcher and dual-plane findings, the `BUILD_DOCKERFILE` /
  `BUILD_ARGS` build step, the anchored worker rsync and the pip-index build argument.
- **kpham-sgl**, [sgl-project/sglang#39187](https://github.com/sgl-project/sglang/pull/39187): the
  bounded dense-indexer prefill transient, backported as `adapter/indexer_chunked*.py`.
- **BBuf** and the SGLang `dsv4.1` branch contributors ([#39370](https://github.com/sgl-project/sglang/pull/39370),
  [#39646](https://github.com/sgl-project/sglang/pull/39646), [#39648](https://github.com/sgl-project/sglang/pull/39648),
  [#39653](https://github.com/sgl-project/sglang/pull/39653)): the decode kernel work in the canary images.
- **hushengkai**, for independently reproducing the EP2 / Engram cache / shared-expert padding changes
  on a second 4x GB10 fleet; **koldfrontier** for the sparkDash prefill-filler finding.
