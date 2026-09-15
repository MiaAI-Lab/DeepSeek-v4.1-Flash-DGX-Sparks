<h1 align="center">DeepSeek-V4.1-Flash on 3x and 4x NVIDIA DGX Sparks</h1>

<p align="center">
  <sub>by <a href="https://x.com/MiaAI_lab">Mia'a AI Lab</a></sub>
  <br><br>
  <a href="https://github.com/sponsors/MiaAI-Lab" target="_blank" rel="noopener noreferrer" style="display:inline-block;margin:0 8px;vertical-align:middle;"><img src="https://img.shields.io/badge/Sponsor%20me%20on%20GitHub-181717?style=for-the-badge&logo=githubsponsors&logoColor=white" alt="Sponsor me on GitHub" height="28" style="height:28px;width:auto;vertical-align:middle;border:0;" /></a>
  <a href="https://x.com/MiaAI_lab" target="_blank" rel="noopener noreferrer" style="display:inline-block;margin:0 8px;vertical-align:middle;"><img src="https://img.shields.io/badge/Follow%20me%20on%20X-000000?style=for-the-badge&logo=x&logoColor=white" alt="Follow Mia on X" height="28" style="height:28px;width:auto;vertical-align:middle;border:0;" /></a>
</p>

Serve [deepseek-ai/DeepSeek-V4.1-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash)
with SGLang on three DGX Spark boxes (GB10, SM121, 121.7 GiB unified memory each) over a
ConnectX-7 RoCE triangle: **native MXFP4 experts, FP8 dense weights, NVMe-resident Engram
tables, DSpark speculative decoding, tool calling and vision**, behind one OpenAI-compatible
endpoint.

| | |
|---|---|
| Decode, prose, 1 stream | 37.9 tok/s, TTFT 248 ms (~82 ms per speculative step) |
| Decode, prose, 2 / 3 / 4 streams | 58.9 / 71.2 / 78.6 tok/s aggregate (30.5 / 24.5 / 20.9 per stream), TTFT 424 / 311 / 383 ms |
| Decode, 4 Sparks (TP=4), 1 / 2 / 4 / 8 / 16 streams | 45.4 / 72.9 / 103.1 / 114.1 / 134.2 tok/s aggregate (45.4 / 37.9 / 26.7 / 23.2 / 22.0 per stream), TTFT 212 / 232 / 270 ms, 2.70 / 12.45 s |
| Prefill, 4 Sparks, 4K / 16K / 32K / 64K / 128K | 3,350 / 3,782 / 3,768 / 3,531 / 3,251 tok/s (TTFT 1.23 / 4.34 / 8.70 / 18.57 / 40.32 s) |
| Context, 4 Sparks (TP=4) | 1M configured (model maximum) and verified: a 1M-context needle test passed with the shipped profile (1024-token chunks, 8M KV pin, 0.80 memory fraction) |
| Context | 200k-256k limit configured (model max 1M); verified with the prefill empty-cache hook and 1024-token chunks: single prompts to 208k, 4 concurrent 46k prompts; KV pool 750k tokens |
| Memory left on the head while serving | ~6 GB (was <1 GB) |
| Greedy decoding | deterministic run to run |

## Why three Sparks

On disk the checkpoint is 476 GiB. After moving the two Engram n-gram tables (189 GiB) to
NVMe, the GPU-resident part is ~305 GiB of MXFP4 experts plus FP8 dense weights:

| Layout | Resident weights per GPU | Fits 121.7 GiB unified? |
|---|---:|:---|
| 2 ranks | 145 GiB | no |
| **3 ranks** | **~101 GiB** | **yes, with NVMe Engram** |
| 4×96 GB (upstream) | 77 GiB | yes |

Host RAM *is* GPU memory on a Spark. Everything that pins host memory (NCCL buffers,
row caches, tokenizer processes) competes with the model, and a head node that runs out of
memory stalls all three ranks because TP is synchronous. Most of what this repo does beyond
the upstream recipe is about that.

**Never use `OFFLOAD_MODE=ram` here.** The Engram tables would evict the model.

## Topology

```
spark1  10.0.0.1  rank 0   API :8888, NFSv4 export of the checkpoint, head-node processes
spark2  10.0.0.2  rank 1   mounts spark1's checkpoint over ConnectX (10.0.22.1)
spark3  10.0.0.3  rank 2   mounts spark1's checkpoint over ConnectX (10.0.23.1)
```

- TP=3, EP=3 (world size 3), NCCL over RoCE (`NCCL_NET=IB`), Gloo bootstrap on the LAN NIC.
- Workers do **not** copy the checkpoint: `./start.sh share` publishes spark1's tree via the
  `vllm-fn-nfs` exporter and creates the `dsv41-weights` docker volume on each worker.
- Each node keeps its own rank's Engram rows on local NVMe (`./start.sh pack`, ~63 GiB/node).
- `heads=64`, `o_groups=8`, `vocab=129280` and the 128 DSpark draft experts are not divisible
  by 3. `adapter/tp3_pad.py` pads heads 64→96, groups 8→12 (32 local heads, the smallest
  size the SM120 sparse-MLA kernels instantiate), vocab to a TP-aligned size and draft
  experts 128→129, zero-filling the extra shards. Rank 2's attention shard is therefore
  entirely padding; see *Padded shards* below for why that matters.

## Quick start

```bash
cd ~/NewModels/DS4.1
cp -n .env.example .env        # IPs, users and knobs; already set for spark1/2/3
./start.sh doctor              # ssh, docker, IB, checkpoint shards, image arch, busy GPU
./start.sh build               # base image + this repo's overlay, on all three nodes
./start.sh share               # NFSv4 export + docker volumes on the workers
./start.sh pack                # Engram shards onto each node's NVMe (once; ~10 min)
./start.sh serve               # workers first, then the head; streams the engine log
```

A full boot takes 12-13 minutes (8 min to read 476 GiB, then the draft model, KV pool and
CUDA-graph capture). `./start.sh status`, `./start.sh logs [worker1|worker2] [N]`,
`./start.sh smoke`, `./start.sh stop`.

```bash
curl http://10.0.0.1:8888/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"deepseek-v4.1-flash","messages":[{"role":"user","content":"What is 19 + 23?"}],
       "max_tokens":32, "chat_template_kwargs":{"thinking":false}}'
```

`enable_thinking` is accepted as an alias for `thinking` (vLLM / EXL3 clients). Either key turns thinking off; if both are set and disagree, `thinking` wins. Always send `max_tokens` (the EXL3 recipe does this on every smoke). Requests that omit it are filled and clamped at `DSV41_MAX_NEW_TOKENS` (default 32768) so a generation loop cannot run to the 1M context. A live token stream also cannot spin forever: an n-gram or same-line repeat aborts the request (see `DSV41_LOOP_*`). `--watchdog-timeout 1800` does not fire while tokens keep arriving.

Set `API_KEY` in `.env` to require a bearer token (`state/api-key` holds it).

## Four Sparks (TP=4)

`start-tp4.sh` is the same engine and image with a profile of its own: `.env.tp4`
(copied from `.env.tp4.example` on first run), `state-tp4/` and `logs-tp4/`, so one
checkout can drive a 3-node and a 4-node fleet. Everything TP4-specific lives in
`.env.tp4.example`: 4 workers (`WORKER_IPS`, `WORKER_HOSTS`, `NFS_SERVER_IPS`, one address
for all workers behind a switch or one per worker), `TP_SIZE=EP_SIZE=4`, and a runtime sized
for the ~40 GB per rank that TP4 leaves free (~77 GiB of weights per rank instead of ~101).
The profile ships with these defaults:

| TP4 setting | Value | Why |
|---|---|---|
| `CONTEXT_LENGTH` | 1,048,576 | model maximum; a 1M-context needle test passed at these settings |
| `MAX_TOTAL_TOKENS` | 8,000,000 | 13.4 GB of KV per rank: 8 requests at ~1M tokens each; see below |
| `MAX_RUNNING_REQUESTS` | 8 | also `--cuda-graph-max-bs-decode`: capture covers batch 1-8. Needs `--min-free-slots-delay 1` to be reachable; see below |
| `CHUNKED_PREFILL_SIZE` | 1024 | ~15 GB peak indexer transient at 1M context (4096 would need ~60 GB); the size the long-context envelope was verified at |
| `MEM_FRACTION_STATIC` | 0.80 | keeps ~24 GiB per rank outside the static pool for that transient |
| `DSV41_TP_PAD` | 0 | heads, o_groups, draft experts and vocab all divide by 4: no padded shards |

**The KV pin.** At TP4 the boot log's budget line — `DSV4 memory calculation:
bytes_per_full_token=1670.75 ... full_token=` — reported room for ~16M tokens (~26.7 GB per
rank) at `MEM_FRACTION_STATIC=0.90`. This profile runs 0.80, which takes ~13 GB off that
budget, and pins 8,000,000 tokens (13.4 GB of KV per rank) under it. These settings are
tested: the profile boots and serves as shipped. With `MAX_RUNNING_REQUESTS=8` the pool
holds 8 requests averaging 1,000,000 tokens each (8 × 1Mi = 8,388,608 would be the pin for
all 8 at the full context). The 0.80 fraction is what keeps ~24 GiB per rank outside the
static pool for the prefill transient, so if the pool ever has to shrink, lower the pin
rather than raise the fraction. **Long prefills:** the indexer's per-chunk transient peaks
at about 14 B × chunk × prefix (`docs/chunked-prefill-memory.md`): ~15 GB for a 1024-token
chunk at 1M context and ~60 GB for 4096, which is why the chunk is 1024. A 1M-context
needle-in-a-haystack test passed on TP4 with these settings, so the 1M envelope is verified
end to end rather than extrapolated; `scripts/verify/memguard.py` remains the tool for
ramping new prompt shapes, and dropping the chunk to 512 is the lever if it ever fires.
**The budget is not fixed:** it is computed from `MemAvailable`
after weights, which moves with the page cache between identical boots, and a pin the budget
cannot cover either kills the boot or is silently clamped down — both have been seen here.
Read `full_token=` from the boot log after any change to weights, TP size or NCCL buffers.
KV pages are touched only when used, so the pool costs nothing until it fills.

**Admission at concurrency 8.** SGLang classifies DSpark as DFlash-family
(`is_dflash_family = is_dflash or is_dspark`), and for that family with
`--max-running-requests >= 8` it auto-enables an admission delayer that holds new prefills
until `min(4, max(2, (max_running_requests + 5) // 6))` slots are free — 2 at 8 requests, 3
at 16. Under steady arrivals the running batch then never sustains the configured maximum:
about 7 of 8, about 13–14 of 16. The TP4 profile passes `--min-free-slots-delay 1` (an
explicit value wins; ≤1 disables) so 8 means 8. The TP3 profile is unaffected — the
formula is off below 8 running requests. Credit for finding this: reported publicly against
the 3-Spark recipe, with +43% aggregate at 8 streams; the mechanism is verified here against
`srt/managers/min_free_slots_delayer.py`, the throughput number is not yet reproduced on our
fleet.

```bash
cp -n .env.tp4.example .env.tp4      # IPs, ssh user, NFS addresses
./start-tp4.sh doctor && ./start-tp4.sh build && ./start-tp4.sh share
./start-tp4.sh pack                  # engram-l*-r<rank>of4.bin shards, once per node
./start-tp4.sh serve                 # ./start-tp4.sh stop|status|logs worker3
```

What changes against the 3-node profile:

| | 3 Sparks (TP3) | 4 Sparks (TP4) |
|---|---|---|
| resident weights per rank | ~101 GiB | ~77 GiB |
| free memory per rank while serving | ~6 GB | ~40 GB |
| attention shards | heads padded 64→96, groups 8→12; rank 2 is all padding | exact: 16 heads, 2 groups per rank, no wasted GEMMs |
| KV pool / context | 750k tokens / 256k limit, ~32k usable (memory-bound on the head) | 8M tokens (pinned) / 1M (model maximum; needle test passed at 1M) |
| concurrency | 4 | 8 by default (measured to 16) |
| NCCL per step | 104 collectives across 3 nodes | 104 collectives across 4 nodes (one more ring hop each) |
| decode, prose, 1 stream | 37.9 tok/s, TTFT 248 ms | 45.4 tok/s, TTFT 212 ms |
| decode, prose, 4 streams | 78.6 tok/s agg (20.9 per stream), TTFT 383 ms | 103.1 tok/s agg (26.7 per stream), TTFT 270 ms |

Measured on 4× DGX Spark with [sparkDash](https://github.com/MiaAI-Lab/sparkDash) (prose
decode, 256 completion tokens; prefill at 4K–128K). Decode is faster than the 3-node fleet at every concurrency — smaller
attention GEMMs per rank and no padded shards more than pay for the extra NCCL hop.
An independent switchless-ring reproduction (public OpenAI path, idle cluster,
greedy code, estimated decode window 129–641) is in
[`docs/tp4-switchless-ring-results.md`](docs/tp4-switchless-ring-results.md): C1 window
67.4 tok/s estimated, C8 aggregate 224.8 tok/s (historical DSpark k=5; current
default remains k=3).
The memory headroom is what makes the 1M context possible. Decode has been measured out to
16 streams (with `MAX_RUNNING_REQUESTS=16`): aggregate throughput is still climbing there
(134.2 tok/s) but per-stream rate has flattened at ~22 tok/s and TTFT degrades sharply past
4 streams — 2.70 s at 8, 12.45 s at 16 — as prefills queue behind each other. Concurrency 16
is a throughput operating point, not a latency one; the profile defaults to 8, and raising
`MAX_RUNNING_REQUESTS` (with the KV pin to match) is how to get the 16-stream point back.

**Decode** (prose, 256 tok):

| load | TTFT | aggregate | per stream |
|---|---:|---:|---:|
| ×1 | 212 ms | 45.4 tok/s | 45.4 tok/s |
| ×2 | 232 ms | 72.9 tok/s | 37.9 tok/s |
| ×4 | 270 ms | 103.1 tok/s | 26.7 tok/s |
| ×8 | 2.70 s | 114.1 tok/s | 23.2 tok/s |
| ×16 | 12.45 s | 134.2 tok/s | 22.0 tok/s |

**Prefill:**

| context | prompt tokens | TTFT | prefill |
|---|---:|---:|---:|
| 4K | 4,118 | 1.23 s | 3,350 tok/s |
| 16K | 16,400 | 4.34 s | 3,782 tok/s |
| 32K | 32,788 | 8.70 s | 3,768 tok/s |
| 64K | 65,557 | 18.57 s | 3,531 tok/s |
| 128K | 131,093 | 40.32 s | 3,251 tok/s |

Prefill peaks at ~3.8k tok/s around 16–32K and is still 3.2k at 128K.

Fabric: a Spark has two ConnectX-7 ports, so three nodes form a full triangle but four
cannot. A 4-node fleet needs either a RoCE switch, or the opt-in switchless-ring path in
[`docs/switchless-ring.md`](docs/switchless-ring.md). The latter requires four DACs, one
point-to-point subnet per cable, the patched `sparkring` NCCL on every rank, and
`NCCL_SWITCHLESS_RING_ONLY=1`; setting only environment variables against stock NCCL is not
enough. The guide includes physical cable diagrams, IP examples, patched-library verification,
local-weight operation and the exact logs that prove Tree/PAT setup was skipped. For a switched
fleet, route NCCL/NFS through the switch and use one `NFS_SERVER_IPS` address. In either layout,
set `IB_HCA`, `NCCL_SOCKET_IFNAME`/`GLOO_SOCKET_IFNAME`, `NFS_CLIENTS` and the GID index in
`.env.tp4`; `./start-tp4.sh doctor` validates the selected path before the cold boot.
`NCCL_DEBUG=INFO` for one boot shows the channels and protocol chosen.

Engram: `./start-tp4.sh pack` writes each rank's rows as `engram-l<layer>-r<rank>of4.bin`
(~48 GiB per node at TP4) onto local NVMe; the names carry the TP size, so `ENGRAM_DIR` can
be shared with a 3-node profile on the same machines.

The control scripts take any number of workers (`WORKER_IPS`/`WORKER_HOSTS` lists; the
legacy `WORKER1_IP`/`WORKER2_IP` pairs still work), so 5+ nodes only need a matching
`TP_SIZE` and `NNODES`. Nothing in the image is TP-specific; the padded-shard repair simply
finds nothing to repair at TP4.

## What is in the box

```
start.sh                 doctor / build / share / pack / serve / stop / status / logs / smoke
start-tp4.sh             same commands for 4 Sparks (profile: .env.tp4, state-tp4/, logs-tp4/)
boot.py                  in-container entrypoint: download+verify (optional), launch, smoke
Dockerfile               lmsysorg/sglang:dev-dsv41 (arm64) + adapter/ + runtime/ overlay
adapter/
  sitecustomize.py       import hooks that install the pieces below at process start
  encoding_compat.py     enable_thinking alias, publisher effort table, max_tokens cap
  loop_abort.py          n-gram / identical-token / same-line stop on a live decode loop
  tp3_pad.py             TP=3 padding of heads / groups / vocab / draft experts
  engram_backend.py      EngramEmbedding replacement: exact rows from NVMe via a host callback
  row_store.cpp          the row store: O_DIRECT reads, 96-thread miss servicing, packed shards
  mxfp8_b12x.py          routes MXFP8 dense linears to FlashInfer's b12x kernel; repairs the
                         block scales of padded shards
  prefill_empty_cache.py empties the allocator cache after each long prefill chunk (memory)
runtime/flash_mla_sm120.py  SM12x sparse-MLA dispatch (64-token page split for FlashInfer)
scripts/pack_engram.py   repack one rank's Engram rows (weight+scale adjacent) to local disk
scripts/profile/         torch-profiler helper and trace analysers (see Profiling)
files/                   NFS export helpers
benchmarks/, tests/      upstream benchmark, row-store, thinking-alias, encoder-parity, max-tokens and loop-abort tests
```

## Knobs (`.env`)

| Variable | Default | Notes |
|---|---|---|
| `TP_SIZE` / `EP_SIZE` / `NNODES` | 3 / 3 / 3 | world size = 3 GPUs |
| `CONTEXT_LENGTH` | 262144 | per-request limit; model max 1,048,576. The head's memory bounds what works: 32k prompts (4 concurrent) are verified, 64k+ exhausts the head (REPORT.md §17); use 32768 for unattended endpoints |
| `MAX_TOTAL_TOKENS` | 750000 | KV pool, pinned. 1,670.75 B/token/rank; see *Memory* |
| `MAX_RUNNING_REQUESTS` | 4 | also the CUDA-graph batch tiers (1-4) |
| `MEM_FRACTION_STATIC` | 0.95 | ≥0.944 needed; lowering it does not free RAM, it only starves KV |
| `SPEC_ALGO` / `DSPARK_BLOCK_SIZE` | DSPARK / 3 | 4-token verify window (1+k). k=5 over-drafts on chat/prose; TP3 ×1 greedy prose was 22.3→25.4 tok/s. Decode tables below were measured at k=5. `off` disables speculation |
| `DSPARK_SPS_TABLE` | /state/dspark_sps.json | profiled cost table; enables compact ragged verify when present |
| `EXTRA_SGLANG_ARGS` | `--fp8-gemm-backend flashinfer_cutlass --watchdog-timeout 1800` | required for the MXFP8 route (below) |
| `DSV41_MXFP8_BACKEND` | b12x | FlashInfer kernel for the FP8 dense projections: `b12x`, `cudnn`, `cutlass` |
| `SGLANG_FLASHINFER_MOE_FUSED_FINALIZE` | 0 | keep 0: the fused (atomic) finalize is nondeterministic |
| `SGLANG_DSV41_REASONING_EFFORT` | 75 | default budget for thinking-mode requests without a `reasoning_effort`. SGLang's V4.1 encoder maps `low`/`high`/`xhigh`/`max` to 25/50/75/100 and defaults to `high` = 50; the publisher's `encoding.py` says 50/75/100, default 75. The adapter remaps the tiers; this pins the default. Integer 1-100; see *Correctness notes* |
| `DSV41_MAX_NEW_TOKENS` | 32768 | fill-in and hard cap when a chat request omits `max_tokens` / `max_completion_tokens`, or asks for more. 0 leaves SGLang's remaining-context default (how VIS-04 ran to 714k tokens). |
| `DSV41_LOOP_ABORT` | 1 | stop a request whose output is an n-gram cycle, a same-token run, or the same decoded line repeated. `finish_reason=stop` / `matched=repetition`. 0 disables. The 2× EXL3 recipe has no equivalent; `--watchdog-timeout` does not fire on a live stream. |
| `DSV41_LOOP_NGRAM` / `DSV41_LOOP_REPEATS` | 32 / 4 | minimum cycle length in tokens, and consecutive copies required (keeps one copy). |
| `DSV41_LOOP_LINE_REPEATS` | 8 | identical decoded lines of at least 16 characters. 0 disables the line detector. |
| `NCCL_BUFFSIZE` / `NCCL_LL128_BUFFSIZE` / `NCCL_PROTO` / `NCCL_MAX_NCHANNELS` | 1 MiB / 256 KiB / `^LL128` / 8 | NCCL connection buffers: 4.7 GiB → 0.14 GiB pinned per node |
| `PYTORCH_CUDA_ALLOC_CONF` | `expandable_segments:False` | never `True`: with expandable segments every prefill of more than 64 query tokens returns NaN logits (garbage for prompts of 65-~4000 tokens and for 3+ concurrent requests; REPORT.md §17). The native allocator fragments on long prompts, hence the context cap |
| `DSV41_PREFILL_EMPTY_CACHE_TOKENS` | 8192 | `adapter/prefill_empty_cache.py`: after each prefill chunk of a sequence this long or longer, give the allocator's cached blocks back so a long prompt costs one chunk's transient, not the sum (REPORT.md §17). 0 disables |
| `DSV41_IO_THREADS` | 96 | Engram miss-servicing threads (NVMe does ~112k IOPS at QD 64) |
| `DSV41_CACHE_GIB` | 0 | Engram row cache; 0 because n-gram rows have ~0% reuse and RAM is GPU memory |
| `OFFLOAD_MODE` | nvme | keep |
| `SKIP_SMOKE` / `SMOKE_QUICK` | 0 / 1 | arithmetic smoke after `/health`; 0 = also JSON, tools, vision |

`.env` is sourced by bash: quote any value with spaces.

## How a decode step spends its time

Measured with torch-profiler traces of all three ranks (DSpark, batch 1, 42 steps):

| Component (per step, slowest rank) | ms | Notes |
|---|---|---|
| dense FP8 projections (`b12x` MXFP8 kernel, ~230 calls) | ~17 | was 52 on Triton / 50 on CUTLASS SM120 |
| MoE grouped FP4 GEMM (86 calls) | ~27 | at memory bandwidth; scales with the 6-token verify window |
| bf16 GEMMs: `wo_a` einsum fallback, lm_head | ~13 | `wo_a` has no FP8 kernel on SM121 yet |
| NCCL (104 collectives across 3 nodes) | ~13-16 | LL protocol over RoCE, median ~50-100 µs each |
| ~2000 small kernels (attention, hyper-connections, quant, norm) | ~10 | attention itself is ~2 ms |
| Engram host callbacks (2 layers) | ~1-3 | one 4 KiB O_DIRECT read per row |
| **total** | **~82** | GPU busy ~95% of the step; CPU overhead is hidden |

Where the 118 → 82 ms came from, in order of size:

1. **FP8 dense GEMMs.** SGLang sends any block size other than 128×128 to a Triton kernel
   unless `--fp8-gemm-backend flashinfer_*` is given; this model uses 32×32 ue8m0 blocks. The
   CUTLASS SM120 MXFP8 kernel FlashInfer picks by default (128×32 tile) was no faster at M=6.
   The kernel that fits is FlashInfer's `b12x` warp-level MMA kernel (16|32-row tiles), which
   SGLang's enum does not expose; `adapter/mxfp8_b12x.py` routes to it.
2. **Padded shards.** Rank 2's `wq_b`/`wo_b` are all zeros, and the never-written block-scale
   memory held denormals, so the MXFP8 re-encoding rejected them and 86 layers fell back to
   Triton. The adapter sets invalid scales on all-zero weight blocks to 1.0 (0 × 1 = 0).
3. **Worker Engram shards.** Unpacked workers read the checkpoint over NFS twice per row;
   rank 0 waited ~11 ms per step at the following all-reduce. `./start.sh pack` on every node.
4. **NCCL buffers** (memory, which becomes speed on this box; see below).

## Memory on a unified-memory box

- **KV is tiny.** Only the four `kv_source` layers store KV, so full-attention KV costs
  1,670.75 bytes per token per rank (MLA latent, FP8, replicated across TP): 131k tokens =
  0.22 GB, 1M tokens = 1.67 GB, plus a fixed 0.34 GB sliding-window pool.
- **The KV budget is what is left after weights**:
  `MemAvailable_after_weights − 0.05 × MemAvailable_before_weights − 0.1 GB`, and SGLang's
  "avail mem" on GB10 is literally the OS `MemAvailable`, so page cache at load time moves it
  by ±0.5 GB between identical boots. The pin (`MAX_TOTAL_TOKENS`) is silently clamped to the
  budget; read the `DSV4 memory calculation: ... available_bytes=` line of the boot log.
- **NCCL** allocated 512 connection buffers × 9.19 MiB (Simple + LL128 + LL) in pinned host
  RAM per node, 4.7 GiB that showed up as unreclaimable `Shmem`. Decode all-reduces are 61 KB
  and use the LL protocol; prefill uses Simple. The `NCCL_*` knobs above cut it to 139 MB and
  turned the head's 0.2-0.8 GB of free memory into ~6 GB, which is why the boot-time "KV
  lottery" and the reclaim stalls went away. NCCL logs 2 communicators × 8 channels now.
- Hierarchical cache, CPU offload and `swa_full_tokens_ratio` do not apply: host RAM is GPU
  RAM, KV is small, and the SWA pool is cap-sized.
- **Long prompts and the allocator.** Each 2048-token prefill chunk allocates an indexer-logits
  buffer proportional to the prefix so far. With PyTorch's native caching allocator a freed
  smaller block cannot serve the next larger request, so reserved memory grows with the
  square of the prompt length (8 GiB reserved with nothing live after 64 chunks in a synthetic
  test): a 64k-128k-token prefill exhausted the head. `expandable_segments:True` would coalesce
  that freed space (0.6 GiB in the same test) **but cannot be used here**: with it every
  prefill of more than 64 query tokens produces NaN logits (garbage for 65-~4000-token prompts
  and for any prefill batch of 3+ requests; REPORT.md §17). So the native allocator stays and
  `CONTEXT_LENGTH` defaults to 256k but only ~32k of it is usable on the head. On top of the fragmentation, a long prompt costs an
  upfront transient on the head that scales with its length (the Engram rows of the whole
  prompt are fetched before the first chunk runs, plus 1.67 KB/token of KV and the
  `CHUNKED_PREFILL_SIZE × context/4 × 4 B` indexer-logits buffer per chunk); the measured
  ramp is in REPORT.md §17.

## Correctness notes

- **Fused MoE finalize is off.** FlashInfer's fused finalize sums the six expert outputs with
  atomic bf16 adds inside the second MoE GEMM. The autotuner selected it for the 32-token
  bucket only (17-32-token prefills, decode at concurrency 3-4), which made greedy outputs
  differ run to run by up to ~1 nat on first-token logprobs and rounds to bf16 at every add.
  With `SGLANG_FLASHINFER_MOE_FUSED_FINALIZE=0` repeated requests are byte-identical at every
  length tested; cost ~0.3 ms per step. Batches at concurrency >1 can still differ between
  runs because batch composition changes which GEMM tiles run; that is normal SGLang
  behaviour without `--enable-deterministic-inference`.
- **Reasoning budget.** DeepSeek-V4.1 takes a numeric reasoning effort (1-100) as a prompt
  prefix in thinking mode. SGLang's built-in V4.1 encoder maps the named tiers
  `low`/`high`/`xhigh`/`max` to 25/50/75/100 and defaults to `high`, so every thinking request
  ran at budget 50 and `low` meant 25; the publisher's `encoding/encoding.py` maps
  `low`/`high`/`max` to 50/75/100 with default 75 (evaluations used 100). `adapter/encoding_compat.py`
  remaps the tiers to the publisher's table (`xhigh` kept as an alias for 75) and
  `SGLANG_DSV41_REASONING_EFFORT=75` pins the default. Per request: the OpenAI `reasoning_effort`
  field (a tier, or a 0-0.99 float = budget/100) or `"chat_template_kwargs":{"reasoning_effort":N}`;
  `medium` is not a V4.1 tier and falls back to the default with a warning. The budget changes
  reasoning length and DSpark acceptance, so state it with any thinking-mode number; the decode
  tables in this README are thinking-off. Same correction as local-inference-lab/rtx6kpro R37.
- **Thinking toggle.** SGLang reads `chat_template_kwargs.thinking`. vLLM-style
  `enable_thinking` is copied onto `thinking` before encode/parse, so either name works.
  `python3 tests/test_thinking_alias.py` covers the alias; `python3 tests/test_chat_encoding.py`
  diffs tool and thinking-off renders against the checkpoint's `encoding/encoding.py` (and
  against SGLang's `encoding_dsv41` inside the image).
- **Completion cap.** Chat requests that omit `max_tokens` fill the remaining context on
  this OpenAI path. The 2× EXL3 recipe has no server-side cap either (vLLM does the same
  fill) but every one of its smokes and README curls sends `max_tokens`. This recipe now
  does both: examples send `max_tokens`, and `adapter/encoding_compat.py` fills/clamps at
  `DSV41_MAX_NEW_TOKENS` (default 32768). Set it to 0 to replay an uncapped harness.
- **Loop abort.** VIS-04 was a semantic loop, not a hung GPU: 714k tokens over ~2.6 h while
  `--watchdog-timeout 1800` stayed quiet. `adapter/loop_abort.py` finishes the request after
  four consecutive copies of a ≥32-token n-gram, 64 identical tokens, or eight identical
  decoded lines, with OpenAI `finish_reason=stop` (`matched=repetition`). The 2× EXL3 recipe
  has no decode-side detector. `python3 tests/test_loop_abort.py` covers the detector; set
  `DSV41_LOOP_ABORT=0` to turn it off.
- `runtime/flash_mla_sm120.py` sends prefills of more than 64 query tokens to FlashInfer's
  sparse-prefill specialisation and everything else through the decode kernel with a
  256→64-token page split; both paths were verified deterministic.
- The autotune cache lives in `~/.cache/sglang/flashinfer/autotune/` on each node (root-owned,
  bind-mounted into the containers). Clear it on all nodes after changing a kernel flag.

## Profiling

```bash
./scripts/profile/profile_decode.sh 45 /tmp/prof        # arms POST /start_profile, waits
# ...send one request from a worker while it runs...
scp trace.json.gz zurih@10.0.0.3:/tmp/ && ssh zurih@10.0.0.3 'cd /tmp && python3 analyze_steps.py trace.json.gz'
```

`analyze_steps.py` prints the per-step GPU budget by category and, with a step index, the
collapsed kernel sequence of that step; `analyze_trace.py` prints the kernel table, GPU idle
gaps and NCCL latency percentiles; `engram_wait.py` isolates the all-reduce that follows each
Engram gather (how long rank 0 waits for the workers' NVMe reads). Analyse on a worker: no
second CUDA context can be created on a node while its rank is up (`cudaMemGetInfo` itself
fails), and the head has little RAM to spare.

## Operating notes

- Run load generators and analysis from spark2/spark3, never on the head: it hosts the HTTP
  server, tokenizer, detokenizer and the NFS export, and is the rank the others wait for.
- Measure decode as `usage.completion_tokens / elapsed` on non-streaming requests or from the
  `gen throughput` field of `Decode batch` log lines; counting SSE chunks under-reports ~3.5×
  because each chunk carries a whole accepted speculative block.
- The TP4 decode and prefill tables above were produced with
  [sparkDash](https://github.com/MiaAI-Lab/sparkDash), which drives the OpenAI endpoint at a
  fixed stream count and reports aggregate and per-stream rates alongside TTFT.
- `./start.sh build` rsyncs the repo to the workers with `--delete`; `engram/` and the state,
  logs and `.env` are excluded. Do not edit `start.sh` while a `./start.sh` command is running
  (bash reads scripts incrementally).
- The spare containers on spark3 add jitter to rank 2 (it was 5-8% slower on identical GEMMs).
  Stop what you can while serving.
- `NCCL_DEBUG=INFO` with `NCCL_DEBUG_SUBSYS=INIT,ENV` for one boot shows the communicator,
  channel and protocol choices.

## Still on the table

- `wo_a` runs as a bf16 einsum (cuBLAS `cutlass_80` kernels, ~11 ms per step); an MXFP8 path
  needs a small change in `models/deepseek_v4.py`.
- The MoE cost scales with the 6-token verify window. A profiled SPS table
  (`python -m sglang.benchmark.dspark_sps_profiler all --base-url ...`, pushed to every rank
  by `start.sh`) turns on compact ragged verify, which mainly pays at concurrency ≥2.
- Fewer, fused small kernels (~2000 per step) and the 13-16 ms of cross-node latency are the
  remaining floor at TP=3.

## Attribution

- The recipe skeleton, `boot.py`, the Engram row-store idea, `benchmarks/`, `tests/` and the
  overall Docker layout originate from
  [0xSero/deepseek-v4.1-flash-4x-rtx-pro-6000](https://github.com/0xSero/deepseek-v4.1-flash-4x-rtx-pro-6000)
  (MIT, notice retained in `LICENSE.upstream-MIT`). Everything Spark-specific (TP=3 padding,
  NFS sharing, packed shards, the b12x route, the padded-scale repair, the NCCL and finalize
  settings) was added here.
- `runtime/flash_mla_sm120.py` is Apache-2.0 from SGLang (`LICENSE.sglang`).
- Serving engine: [SGLang](https://github.com/sgl-project/sglang) dev build with
  [FlashInfer](https://github.com/flashinfer-ai/flashinfer) kernels.
- Model weights stay on Hugging Face / local NVMe and are not redistributed. Pinned checkpoint:
  `deepseek-ai/DeepSeek-V4.1-Flash` @ `fb2764a5cf321eaa5070ca8f9e892818f477c16d`.

## License

This repository is licensed under the **GNU Affero General Public License v3.0 or later**
(`LICENSE`). It builds on 0xSero's MIT-licensed recipe (notice kept in `LICENSE.upstream-MIT`)
and includes one Apache-2.0 file from SGLang (`LICENSE.sglang`); see `NOTICE` for the full
breakdown. Model weights are not part of this repository.
