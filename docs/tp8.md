# Eight Sparks (TP8): the production line on a switched fabric

The TP4 production line ([docs/tp4.md](tp4.md)) runs on eight DGX Spark (GB10) nodes with the same
launcher path, image and adapters. `./start-tp8.sh` is `./start-tp4.sh` with its own profile:
`.env.tp8` (copied from `.env.tp8.example` on first run), `state-tp8/` and `logs-tp8/`. It sets
`DSV41_LAUNCHER=tp4`, so the TP4 behaviour (named Dockerfile, `EXTRA_CONTAINER_ENV`, anchored worker
rsync, production hooks) is what runs. The TP3 and TP4 profiles are unchanged.

TP8 is the lowest-latency layout for one user at a time. For many concurrent users, two TP4 replicas
behind the SGLang router serve more; both were measured side by side below ("TP8 or 2 × TP4?").

## What TP8 changes

Only settings, plus one adapter generalisation:

| Setting | TP4 | TP8 | Why |
|---|---|---|---|
| `NNODES`, `TP_SIZE` | 4 | 8 | 7 workers in `WORKER_IPS` / `WORKER_HOSTS` |
| `EP_SIZE` | 1 | **2** | the routed MoE on b12x_next needs a per-rank expert width that is a multiple of 64. At TP8, EP 1 gives every rank 2304 / 8 = 288 columns of each expert, which b12x_next rejects at load; EP 2 gives 576, the TP4/EP1 shape. `start.sh` now refuses TP8 + EP 1 with the b12x_next MoE instead of failing mid-load |
| `DSV41_FAST_LOAD_EP_SIZE` (in `EXTRA_CONTAINER_ENV`) | 1 | **2** | matches `EP_SIZE` |
| `DSV41_SHARED_PAD_K` | pads 576 → 640 | pads **288 → 384** | `adapter/shared_pad_k.py` now derives the shared expert's K from `TP_SIZE` (2304 / TP) instead of a fixed 576, so TP8 keeps the shared expert on the b12x MXFP8 kernel instead of the CUTLASS fallback (43 layers padded; first call checked, b12x accepted the shape). TP3's 768 is already a multiple of 128, so the switch is a logged no-op there |
| `NFS_SHARE` | 1 | **0** (recommended) | every node reads its own copy of the checkpoint through a local docker volume; eight nodes reading one NFS export was not tested. Create the volume once per worker (below) |
| `IB_HCA`, `B12X_ROCE_HCA` | both rails | both rails | switched fabric only; the switchless ring stays TP4 |

Everything else — the canary-roce image, RoCEnante, prefill sequence parallel, the fast loader, the
decode adapters, `MAX_RUNNING_REQUESTS=16`, 1M context, the 8M-token KV pin, `CHUNKED_PREFILL_SIZE=4096`
— is the TP4 production line as shipped.

## Results

Production line (`.env.tp8.example`), `Dockerfile.canary-roce` image built from this branch, 8 DGX
Spark on a switched 200G RoCE fabric (two rails per node), measured 2026-09-26 with
[sparkDash](https://github.com/MiaAI-Lab/sparkDash) 1.8.8 (`754f40a`): 256 new tokens, temperature 0,
thinking off, idle fleet, two warm-up passes discarded. KV pool 8.0M tokens (1M context). Raw output:
[`results/tp8/20260926/`](results/tp8/20260926/) (`validation.txt` has every summary).

**Decode, aggregate tok/s (per stream in brackets)**

| prompt type | c1 | c2 | c4 | c8 | c16 |
|---|---:|---:|---:|---:|---:|
| prose | **111.9** | 149.2 (79.2) | 209.7 (55.4) | 293.3 (38.2) | 386.2 (25.7) |
| code | **184.9** | 250.0 (125.0) | 342.6 (86.9) | 445.8 (61.1) | 563.1 (36.8) |
| structured | 203.9 | 225.9 (124.4) | 302.2 (94.5) | 302.4 (54.1) | 561.9 (48.1) |
| json | 177.2 | 253.7 (128.2) | 372.5 (96.0) | 545.1 (71.5) | 708.9 (45.7) |

Prose c1 is the median of four runs (100.1 / 112.5 / 112.5 / 111.2) after the warm-ups; the other
cells are one sweep per prompt type. As on TP4, sparkDash uses a different prompt set at each
concurrency for the non-prose types, so per-stream values are not comparable across columns.

Against the TP4 line ([docs/tp4.md](tp4.md), same sparkDash release and method):

| | TP4 (4 Sparks) | TP8 (8 Sparks) | change |
|---|---:|---:|---:|
| prose c1 | 87.7 | 111.9 | +28 % |
| code c1 | 124.8 | 184.9 | +48 % |
| structured c1 | 152.4 | 203.9 | +34 % |
| json c1 | 118.9 | 177.2 | +49 % |
| prose c16 aggregate | 342.7 | 386.2 | +13 % |
| code c16 aggregate | 438.3 | 563.1 | +28 % |
| json c16 aggregate | 659.9 | 708.9 | +7 % |

**Prefill, cold, tok/s by prompt length** (sparkDash, one pass)

| 4k | 16k | 32k | 64k | 128k | 262k |
|---:|---:|---:|---:|---:|---:|
| 2715 | 6182 | 6219 | 6087 | 6140 | 5169 |

The 4k cell is a single cold pass (1.51 s to first token) and is noisy; the distinct-token runs below
put 4k at 0.70–1.25 s. As on TP4, sparkDash's prefill filler is one repeated token, which the Engram
row cache favours. With distinct token sequences (a fresh random prefix per request, `/generate` with
`input_ids`, one excluded warm-up, three trials each):

| prompt | 1 × 4,096 | 1 × 32,768 | 1 × 131,072 | 8 × 32,768 concurrent |
|---|---:|---:|---:|---:|
| input tok/s | 4,419 | 5,816 | 5,620 | 6,050 (aggregate) |

TP4 → TP8 roughly keeps prefill (TP4 sparkDash 5,855 / 5,900 / 5,925 / 5,797 at 16k–128k) while
decode rises 28–49 % at c1. Prefill is bound by the fabric: each rank's share of the compute halves,
but the per-layer collectives do not shrink.

**Quality and long context.** The capability suite (arithmetic, two waves of eight concurrent
arithmetic requests, one- and four-image vision, JSON schema, tool round trip) passes, and exact
three-record retrieval passes at 32,866, 131,170 and 299,098 prompt tokens (5.4 s, 22.0 s and 58.8 s
end to end). Both checks are the `validation/` scripts of
[rhys101/DeepSeek-V4.1-Flash-vLLM-DGX-Spark-8](https://github.com/rhys101/DeepSeek-V4.1-Flash-vLLM-DGX-Spark-8);
their outputs are in the results directory. qeval was not run on TP8.

## TP8 or 2 × TP4?

With eight Sparks the other layout is two independent TP4 replicas (this repository's TP4 line on
spark1-4 and spark5-8, `EP_SIZE=1` as shipped, each started with `./start-tp4.sh` from its own head)
behind the SGLang router that ships in the image: `scripts/router.sh start http://HEAD_A:8888
http://HEAD_B:8888`. Both layouts were measured on the same fleet the same day with the same sparkDash
method; the router ran `--policy round_robin`. Aggregate tok/s at the same total number of concurrent
requests; 2 × TP4 is the median of three trials (per-cell spread 1–3 %, structured up to ±10 %), TP8
one sweep. TP8 runs 16 slots, so it has no c32.

| requests | prose TP8 / 2×TP4 | code TP8 / 2×TP4 | structured TP8 / 2×TP4 | json TP8 / 2×TP4 |
|---:|---|---|---|---|
| 1 | **111.9** / 86.0 | **184.9** / 126.8 | **203.9** / 155.2 | **177.2** / 124.3 |
| 8 | 293.3 / 307.4 | 445.8 / 448.8 | 302.4 / 260.9 | 545.1 / 561.6 |
| 16 | 386.2 / **469.3** | 563.1 / 607.9 | 561.9 / **730.7** | 708.9 / **884.3** |
| 32 | — / 673.0 (22.3) | — / 836.3 (28.6) | — / 914.9 (45.2) | — / 1278.1 (42.6) |

(c32 per-stream tok/s in brackets.) On varied prompts — the 8-category community benchmark
(coding, json, narrative, prose, math, reasoning, summary, format; a unique tag per request so nothing
comes from the prefix cache; 150–200-token answers), end to end including prefill, median of three
runs — 2 × TP4 serves:

| | c1 | c8 | c16 | c32 |
|---|---:|---:|---:|---:|
| aggregate tok/s, mean of 8 categories | 83.7 | 362.2 | 559.8 | 809.8 |
| per-stream decode tok/s | 83.7 | 56.5 | 43.8 | 31.6 |
| mean time to first token | 0.20 s | 0.33 s | 0.42 s | 0.60 s |

Structured output (coding, format, math) runs 1.1–1.3k tok/s aggregate at c32, free-form writing
(prose, summary, narrative) 0.43–0.49k. Cold prefill with distinct token sequences:

| | TP8 | 2 × TP4 |
|---|---:|---:|
| 1 × 131,072 | **5,620** | 4,519 |
| 8 × 32,768 concurrent, aggregate | 6,050 | **9,264** |
| 32 × 8,192 concurrent, aggregate (per request) | — | 10,229 (320) |
| 32 × 32,768 concurrent, aggregate (per request) | — | 10,038 (314) |

- **One user at a time:** TP8. Decode is 30-50 % faster per request and a long single prefill ~25 %
  faster.
- **2-8 concurrent requests:** about even (within ~10 % either way, by prompt type).
- **16 and more, or many prefills at once:** 2 × TP4 — 8-30 % more at c16, twice the slots and KV
  pool, ~50 % more concurrent prefill. At high concurrency a TP8 step pays its fixed per-step costs
  once for all eight GPUs and its one-shot all-reduces write to seven peers instead of three, so two
  independent TP4 pipelines do more work per second.
- **Router policy matters.** With `--policy cache_aware` every sparkDash prose stream (one shared
  prompt) was routed to the same replica and c32 queued behind its 16 slots (prose c32 296 tok/s
  against 673 with `round_robin`). `round_robin` (the `scripts/router.sh` default) or `power_of_two`
  suit mixed traffic; `cache_aware` pays off when shared prefixes are common and load is spread. Each
  replica keeps its own prefix cache.

Raw output: [`results/tp8/20260926-2xtp4/`](results/tp8/20260926-2xtp4/) (`robust/` for the three-trial
tables and the varied-prompt runs, `round-robin/` and `cache-aware/` for the first passes and the
prefill bursts).

## Quick start

```bash
cp .env.tp8.example .env.tp8   # 7 workers, fabric interfaces, MODEL_DIR, both HCAs
# on every worker, once: a local volume over that node's own copy of the checkpoint
docker volume create --driver local --opt type=none --opt o=bind \
  --opt device=/path/to/DeepSeek-V4.1-Flash dsv41-weights
./start-tp8.sh doctor
./start-tp8.sh build            # builds on the head and every worker (or build once and docker save | load)
./start-tp8.sh pack             # engram-l*-r<rank>of8.bin, ~24 GiB per node
./start-tp8.sh serve            # ./start-tp8.sh stop | status | logs | smoke
```

The boot log must show, in addition to the TP4 lines:

```
DSV41 shared-expert K padding ARMED: down_proj (5120, 288) -> (5120, 384) ...
[moe_b12x_next] INFO: routed MoE at EP_SIZE=2 (this rank: EP rank 0, 192 experts x N=576) ...
RoCEnante ready: world=8 hcas=rocep1s0f0,roceP2p1s0f0 ... max_size=2097152
max_total_num_tokens=8000000, chunked_prefill_size=4096, ... max_running_requests=16, context_len=1048576
```

## Known limits

- One of three TP8 boots on this fleet stalled in the fast loader on one rank (no shard read
  started; the other seven ranks finished and timed out at the post-load barrier). A restart came up
  normally. `DSV41_FAST_LOAD=0` restores the stock loader if it recurs.
- The measured image is the TP4 v2.1 production line on `main`. The v2.2 decode-step changes of PR #37
  (spec sync-free, eager glue, compact RoCE gather, `wo_a` prefetch window) were also run at TP8/EP2
  on a separate stack on the same fleet and worked unchanged there; they are not measured on this page.
- With `DSV41_MOE_B12X_NEXT_DETERMINISTIC=1` the small MoE plans use b12x's internal route planner
  instead of the Triton one on this image (logged at boot; the Triton-planner patch is part of the
  open TP4 v2.2 PR #37). Outputs stay deterministic.
- Switched fabric only. The switchless ring and its opposite-node paths are four-node constructions.

## Credits

The TP8 profile, the `shared_pad_k` generalisation, the EP guard and these measurements are from
[rhys101](https://github.com/rhys101) ([DeepSeek-V4.1-Flash-vLLM-DGX-Spark-8](https://github.com/rhys101/DeepSeek-V4.1-Flash-vLLM-DGX-Spark-8)).
Everything that runs is the TP4 production line: see [docs/tp4.md](tp4.md) for its credits
(knapcio and the contributors listed there).
