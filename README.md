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
       "chat_template_kwargs":{"thinking":false}}'
```

Set `API_KEY` in `.env` to require a bearer token (`state/api-key` holds it).

## Four Sparks (TP=4)

`start-tp4.sh` is the same engine and image with a profile of its own: `.env.tp4`
(copied from `.env.tp4.example` on first run), `state-tp4/` and `logs-tp4/`, so one
checkout can drive a 3-node and a 4-node fleet. Everything TP4-specific lives in
`.env.tp4.example`: 4 workers (`WORKER_IPS`, `WORKER_HOSTS`, `NFS_SERVER_IPS`, one address
for all workers behind a switch or one per worker), `TP_SIZE=EP_SIZE=4`, and a runtime sized
for the ~40 GB per rank that TP4 leaves free (~77 GiB of weights per rank instead of ~101):

| TP4 setting | Value | Why |
|---|---|---|
| `CONTEXT_LENGTH` | 1,048,576 | model maximum; fits with ~10 GB of slack per rank |
| `MAX_TOTAL_TOKENS` | 4,000,000 | 6.7 GB of KV per rank |
| `MAX_RUNNING_REQUESTS` | 8 | CUDA graphs for batch 1-8 |
| `CHUNKED_PREFILL_SIZE` | 4096 | indexer logits 4.3 GB per chunk at 1M context |
| `MEM_FRACTION_STATIC` | 0.90 | keeps runtime slack for long prefills |
| `DSV41_TP_PAD` | 0 | heads, o_groups, draft experts and vocab all divide by 4: no padded shards |
| `EP_SIZE` | 2 | the step pays the max over the EP groups; two groups instead of four halves the expert-routing straggler. NCCL 16.5 → 10.2 ms per step at bs=1, MoE unchanged |
| `DSV41_CACHE_GIB` / `_WAYS` | 4 / 16 | Engram rows do get reused: 67-76% hit, 4.1× fewer NVMe reads. Affordable at TP4, not at TP3 |
| `--min-free-slots-delay` | 1 | without it the 8th request slot is never used |
| `--enable-deepseek-v4-fp4-indexer` | on | attention 3.5 → 2.7 ms per step |

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
| KV pool / context | 750k tokens / 256k limit, ~32k usable (memory-bound on the head) | 4M tokens / 1M (model maximum) |
| concurrency | 4 | 8 |
| NCCL per step | 104 collectives across 3 nodes | 104 collectives across 4 nodes (one more ring hop each) |

Measured on a four-Spark fleet with the settings above, using `scripts/verify/throughput.py`
as shipped (prose, 300 tokens, from a worker against the head over the rail, idle server):

| conc | temp 0.0 | temp 0.7 |
|---:|---|---|
| 1 | 30.7 → **35.3** | 28.2 → **34.1** |
| 2 | 47.2 → **50.8** | 46.3 → **50.7** |
| 3 | 57.7 → **62.2** | 54.3 → **58.6** |
| 4 | 68.7 → **74.0** | 66.1 → **71.4** |

Mean **+10.5%** (left: stock `.env.tp4.example`; right: with `EP_SIZE=2`, the Engram cache,
`--min-free-slots-delay 1` and the FP4 indexer). Decode step time at bs=1, taken from the engine's
own `Decode batch` lines as `accept len / gen throughput` and therefore content-independent:
**69.0 → 61.0 ms median**. Output quality is unchanged — 75 auto-scored tasks compared pairwise
after each step, zero broken every time, plus 30/30 on long-context needle retrieval up to 262k.

Fabric: a Spark has two ConnectX-7 ports, so three nodes form a full triangle but four
cannot; a 4-node fleet needs a RoCE switch (all NCCL and NFS traffic through it, one
`NFS_SERVER_IPS` address) or a ring with NCCL routed over it. Set `NCCL_IB_HCA`,
`NCCL_SOCKET_IFNAME`/`GLOO_SOCKET_IFNAME`, `NFS_CLIENTS` and the GID index in `.env.tp4` for
that fabric; `./start-tp4.sh doctor` checks reachability and the NFS mounts before a
13-minute boot. `NCCL_DEBUG=INFO` for one boot shows the channels and protocol chosen.

Engram: `./start-tp4.sh pack` writes each rank's rows as `engram-l<layer>-r<rank>of4.bin`
(~48 GiB per node at TP4) onto local NVMe; the names carry the TP size, so `ENGRAM_DIR` can
be shared with a 3-node profile on the same machines.

The control scripts take any number of workers (`WORKER_IPS`/`WORKER_HOSTS` lists; the
legacy `WORKER1_IP`/`WORKER2_IP` pairs still work), so 5+ nodes only need a matching
`TP_SIZE` and `NNODES`. Nothing in the image is TP-specific; the padded-shard repair simply
finds nothing to repair at TP4.

Most measurements in this README come from the 3-node fleet. The TP4 profile has since been
booted and tuned on a four-Spark fleet (see the settings above); TP4-specific numbers are marked
as such.
Expect the same per-step structure (dense GEMMs, MoE, NCCL) with smaller attention GEMMs
per rank and one extra network hop per collective; the memory headroom is what makes the
long context and the higher concurrency safe, not a faster step.

## What is in the box

```
start.sh                 doctor / build / share / pack / serve / stop / status / logs / smoke
start-tp4.sh             same commands for 4 Sparks (profile: .env.tp4, state-tp4/, logs-tp4/)
boot.py                  in-container entrypoint: download+verify (optional), launch, smoke
Dockerfile               lmsysorg/sglang:dev-dsv41 (arm64) + adapter/ + runtime/ overlay
adapter/
  sitecustomize.py       import hooks that install the pieces below at process start
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
benchmarks/, tests/      upstream benchmark and row-store tests
```

## Knobs (`.env`)

| Variable | Default | Notes |
|---|---|---|
| `TP_SIZE` / `EP_SIZE` / `NNODES` | 3 / 3 / 3 | world size = 3 GPUs |
| `CONTEXT_LENGTH` | 262144 | per-request limit; model max 1,048,576. The head's memory bounds what works: 32k prompts (4 concurrent) are verified, 64k+ exhausts the head (REPORT.md §17); use 32768 for unattended endpoints |
| `MAX_TOTAL_TOKENS` | 750000 | KV pool, pinned. 1,670.75 B/token/rank; see *Memory* |
| `MAX_RUNNING_REQUESTS` | 4 | also the CUDA-graph batch tiers (1-4) |
| `MEM_FRACTION_STATIC` | 0.95 | ≥0.944 needed; lowering it does not free RAM, it only starves KV |
| `SPEC_ALGO` / `DSPARK_BLOCK_SIZE` | DSPARK / 5 | 6-token verify window; `off` disables speculation |
| `DSPARK_SPS_TABLE` | /state/dspark_sps.json | profiled cost table; enables compact ragged verify when present |
| `EXTRA_SGLANG_ARGS` | `--fp8-gemm-backend flashinfer_cutlass --watchdog-timeout 1800` | required for the MXFP8 route (below) |
| `DSV41_MXFP8_BACKEND` | b12x | FlashInfer kernel for the FP8 dense projections: `b12x`, `cudnn`, `cutlass` |
| `SGLANG_FLASHINFER_MOE_FUSED_FINALIZE` | 0 | keep 0: the fused (atomic) finalize is nondeterministic |
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
- `./start.sh build` rsyncs the repo to the workers with `--delete`; `engram/` and the state,
  logs and `.env` are excluded. Do not edit `start.sh` while a `./start.sh` command is running
  (bash reads scripts incrementally).
- The spare containers on spark3 add jitter to rank 2 (it was 5-8% slower on identical GEMMs).
  Stop what you can while serving.
- `NCCL_DEBUG=INFO` with `NCCL_DEBUG_SUBSYS=INIT,ENV` for one boot shows the communicator,
  channel and protocol choices.

## Still on the table

- `wo_a` runs as a bf16 einsum (cuBLAS `cutlass_80` kernels, ~11 ms per step) **at TP=3 only**;
  an MXFP8 path needs a small change in `models/deepseek_v4.py`. **At TP=4 this is already
  fixed**: the fast path in `_apply_wo_a_bf16_matmul` is gated on `o.shape[1:] == (2, 4096)` and
  `wo_a.shape == (2, 1024, 4096)`, i.e. exactly 2 o_groups per rank, which is `o_groups 8 / TP4`.
  The TP3 pad to 12 groups gives 4 per rank and the kernel is skipped. A TP4 decode profile shows
  `_wo_a_partial` at 3.69 ms and `_wo_a_reduce` at 0.06 ms.
- ~~A profiled SPS table turns on compact ragged verify~~ — **this crashes the engine on this
  model; do not try it.** `boot.py` picks the table up and logs `DSpark ragged-verify scheduler
  enabled (mode=compact, …)`, then the engine dies during CUDA-graph capture:
  `srt/layers/engram.py:296  assert num_tokens == bs * block` —
  *"engram target-verify expects one equal block per request, got 42 tokens for 8 requests of 6"*.
  DeepSeek-V4.1's Engram hasher hard-requires a uniform verify block across requests; compact
  ragged verify exists precisely to make them non-uniform. The two features are mutually exclusive.
  `--speculative-dspark-align-verify-tokens-to-graph-tier` does not help (raggedness is the
  problem, not the rounding), and `cap-accept` keeps the block uniform but verifies everything,
  so it buys no compute.
- Fewer, fused small kernels (~2000 per step) and the cross-node latency are the remaining floor.
  At TP=4 with `EP_SIZE=2` that latency is 10.2 ms per step at bs=1, of which roughly 4 ms is
  transfer and the rest is still waiting on the slowest rank.
- Measured and rejected on a four-Spark fleet, so nobody spends a boot on them twice:
  `DSV41_IO_THREADS` above 96 (the drive already runs at 89-100k IOPS during the burst, sampled
  from `/sys/block/nvme0n1/stat` every 2 ms — the stall is volume ÷ IOPS, not queue depth);
  `DSV41_RESIDENT_SCALES` (dead code with packed shards — the branch sits inside the `else` of
  `if (s->packed_fd >= 0)`); `NCCL_ALGO=Tree` (the trace already shows `..._RING_LL`);
  `DSV41_MXFP8_BACKEND=cudnn` (21.1 ms against b12x's 20.2, every shape accepted — b12x stays);
  and raising the draft's precision with `SGLANG_DSPARK_FP32_LM_HEAD=1` +
  `SGLANG_DSPARK_OPT_MARKOV_W2_BF16=0` (step 69 → 75 ms for ~1.6% more acceptance).

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
