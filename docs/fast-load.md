# Fast weight loading (2026-09-18)

`adapter/fast_load.py`, gate `DSV41_FAST_LOAD=1`. Engine start on the production stack went
from 343 s to 124 s; the values that reach the model are bitwise the same. The one trade-off:
the KV pool comes out 3-6 % smaller (7.27 M vs 7.47-7.82 M tokens on the same image), see the
last section. The gate ships on.

## Where the 285 s went

Profiled with `py-spy dump` every 10 s and `/proc/diskstats` every 5 s during production boots:

- The "Multi-thread loading shards" bar (~45 s) is not the read. safetensors' `get_tensor`
  returns mmap-backed tensors; the bytes move when the model's 24-thread copy pool runs
  `expert_data.copy_(loaded_weight)` in FusedMoE (`_load_w13` / `_load_w2`), and the model
  never waits for those futures until the enumeration is done.
- Those copies page-fault the mmap in 4-128 KB pieces (`read_ahead_kb` = `max_sectors_kb` =
  128 on the Sparks). Per rank ~130 GB at 0.45-0.6 GB/s on the two ranks with `moe_tp_rank`
  0 and ~1.5-2 GB/s on the other two: 230-290 s versus 95-130 s, every boot, same hardware.
  The same NVMe delivers 5.8 GB/s to 16 `pread` threads over the same byte ranges and
  10.5 GB/s O_DIRECT sequential.
- The checkpoint is 510 GB: routed experts 289 GB (384 experts, so 144 GB per EP rank),
  Engram tables 203 GB (never copied; the row store serves them from NVMe and the loader only
  checks their shape), everything else 10.5 GB, and the DSpark draft (`mtp.*`) 7.9 GB.
- The draft load re-opened all 48 shards to consume 2401 `mtp.*` tensors that live in three
  of them: 44-63 s per rank.

## What was tried and why it is what it is

| step | result |
|---|---|
| `posix_fadvise(WILLNEED)` on this rank's ranges when a shard is opened | no gain: the kernel caps it at the readahead window (128 KB per call) |
| explicit threaded `pread` warming of the page cache at shard open | reads ran at 4-5 GB/s for 40 s, then the copies still faulted at 0.5 GB/s: the enumeration ran 45 s ahead of the copies, and on GB10 the GPU weights are system memory, so the page cache is squeezed while the weights land and warmed pages are evicted before use |
| the above plus pacing the copies (`maybe_executor_submit` wrapped with a budget) | ranks 1/3: 108-130 s -> 62-65 s; ranks 0/2 unchanged: their faults are the slow kind whatever is in the cache |
| eager reads into host memory (`torch.empty` + `preadv`), returned from `get_tensor` | load 66 s on every rank, but ~11 GB stayed resident afterwards (glibc keeps freed multi-MB chunks once its dynamic mmap threshold has grown) and the KV pool, sized from the head's free memory, shrank 5x |
| eager reads into anonymous `mmap` buffers (`torch.frombuffer`), pool shut down and `malloc_trim` after each load | load 73 s, the process itself back to baseline, but the KV pool still 10-14 % smaller than a same-image control boot: SGLang sizes it from `MemAvailable` (integrated GPU), and ~1.5 GB had left `MemAvailable` without showing up in any process or kernel counter. A standalone test reproduced it: after a burst of concurrent host-to-device copies from *pageable* memory the driver keeps a staging pool (~0.4 GB per 3 GB burst); pinned sources leave nothing behind |
| **eager reads into pinned host memory (torch's caching host allocator), everything released before the engine measures memory** | **current code: load 71-83 s per rank, KV pool 7.27 M vs 7.75 M in the control boot of the same image (-6 %, or -3 % against the low end of the control spread)** |
| upstream `--weight-loader-prefetch-checkpoints` + `--weight-loader-disable-mmap` | OOM-killed the head (disable-mmap stages whole 10 GB shards in RAM, nine in flight) |

## How it works

1. `safetensors.safe_open` is wrapped (as used by `weight_utils`). For a shard, the names this
   rank will copy are every tensor except the routed experts another EP rank owns and the Engram
   tables. Under EP2, owned experts are read whole even though MoE-TP copies half of w1/w3 (the
   extra read is ~50 GB at NVMe speed). At EP1 they are TP-sliced (next section).
   Those tensors are read by a 16-thread pool into pinned host buffers (torch's caching host
   allocator, so the copy into the parameter is a plain DMA; anonymous mmaps when CUDA is not
   available). The buffers are 256 MiB slabs shared by many tensors (next-to-last section);
   `get_tensor` returns them, everything else stays the stock mmap tensor. The EP rank comes from the runtime
   context, then `parallel_state`, then `DSV41_FAST_LOAD_EP_SIZE`; `n_routed_experts` from
   `config.json` (nested under `text_config` for V4.1) or `DSV41_FAST_LOAD_N_EXPERTS`. If the
   layout is unknown only the non-expert tensors are read eagerly.
2. `maybe_executor_submit` in `deepseek_v4` is wrapped with a byte budget
   (`DSV41_FAST_LOAD_INFLIGHT_GB`, default 6): the enumeration, and with it the loader's window
   and the eager reads, stays just ahead of the copies. A copy whose source is a view into a slab
   charges the whole slab, once, until the last queued copy from that slab finishes; other sources
   charge their own bytes. Host memory in flight is bounded by that budget plus one slab plus the
   loader's window (`--model-loader-extra-config {"num_threads":1}` = 2 shards).
3. The DSpark draft's `load_weights` is marked; during it shards without `mtp.*` are handed back
   as empty handles and the three real ones are read eagerly.
4. When the target's `load_weights` returns and again after the draft's, before the engine
   measures its memory: the reader pool is shut down, the header cache dropped, `gc.collect`,
   `malloc_trim`, `torch.cuda.empty_cache`, the pinned blocks handed back to the driver
   (`torch._C._host_emptyCache`) and every shard's page cache dropped (`POSIX_FADV_DONTNEED`).
   The log line reports RSS, `MemAvailable` and the CUDA allocator before and after.

## Expert tensor parallelism (EP_SIZE=1)

At TP4 with `EP_SIZE=1` (routed MoE on `adapter/moe_b12x_next.py`) every rank holds a quarter of
every one of the 384 experts. Reading owned experts whole would mean all of them: 279 GiB per rank
and a pinned window of 2 x 7.4 GB, the read pattern behind the host-memory livelock. So when
`moe_ep_size == 1` and `moe_tp_size > 1` (read from `get_parallel()` like FusedMoE does, then
`parallel_state`, then `DSV41_FAST_LOAD_EP_SIZE` with TP), routed expert tensors are TP-sliced
(`DSV41_FAST_LOAD_TP_SLICE`, default `auto`):

- `w1`/`w3` weight and scale (`[2304, 2560]` I8, `[2304, 160]` E8M0): FusedMoE narrows dim 0 by
  `shape // moe_tp_size` at `moe_tp_rank`, a contiguous 576-row block, read with one `pread`.
- `w2` weight and scale (`[5120, 1152]` I8, `[5120, 72]` E8M0): FusedMoE narrows dim 1, 288 of
  every 1152 bytes per row. Every 4 KB page holds needed bytes, so the device reads the whole tensor
  either way. Whole rows go into a per-thread 8 MB bounce buffer and the rank's columns are copied
  out. That costs ~0.35 ms per tensor, against ~2.6 ms for 5120 per-row `pread`s, which are held back
  by the GIL.
- `get_tensor` returns a full-shape stand-in (a `torch.Tensor` wrapper subclass) holding only the
  slice. `shape`/`size`/`dim`/`device` report the checkpoint tensor, and `narrow` along the sliced
  dim within the slice returns the resident (pinned, contiguous) bytes. Any other operation, or a
  narrow outside the slice, raises with the name of the tensor, so if the engine's layout differs
  the boot fails instead of loading unread bytes. The pacing budget counts the slice, not the full
  shape.
- The draft's routed experts (3 stages x 128, same FusedMoE groups) are sliced the same way.
- With `DSV41_FAST_LOAD_TP_SLICE=0`, or with the MoE-TP layout unknown, EP1 leaves the routed
  experts to the stock mmap tensors instead of reading them whole. `=1` also slices the owned experts
  under EP2. EP2 at the default `auto` gets the same read list as before.

Per rank, from the checkpoint headers:

| | target read | target kept | largest shard kept | 2-shard window | draft read / kept |
|---|---:|---:|---:|---:|---:|
| EP2 (today) | 155.2 GB (144.5 GiB) | 155.2 GB | 3.80 GB | 7.6 GB | 7.9 / 7.9 GB |
| EP1, experts whole (not used) | 299.6 GB (279.0 GiB) | 299.6 GB | 7.41 GB | 14.8 GB | 7.9 / 7.9 GB |
| **EP1, TP-sliced** | **155.2 GB (144.5 GiB)** | **83.0 GB** | **1.99 GB** | **4.0 GB** | **4.3 / 2.5 GB** |
| EP2 + `TP_SLICE=1` | 107.1 GB | 83.0 GB | 1.99 GB | 4.0 GB | 5.5 / 4.3 GB |

EP1 reads exactly EP2's bytes (half of every expert, against all of half the experts) and keeps half
of them in a pinned window about half of EP2's. `tests/test_fast_load_tp_slice.py` (image build, CPU)
builds shards with the real expert shapes and dtypes and runs every rank at EP1 and EP2, target and
draft, through FusedMoE's own `_load_w13`/`_load_w2` (a mirror of their narrowing where sglang does
not import). The bytes that land in the expert parameter match the stock path. The test also checks
that EP2 gets the legacy read list, that 19 kinds of out-of-slice access raise, that the budget is
paced on slices, and that the bounce buffers are released. The checkpoint test checks the EP1 totals
and compares every slice of one real shard with the stock tensor's narrow.

## Pinned slabs (2026-09-24, booted on the fleet)

The KV pool investigation found 1.1-1.9 GiB
more memory missing from the head's `MemAvailable` at pool sizing with the fast loader than with
the stock one (3.3-3.5 GiB non-torch at EP2, 4.0 at EP1, against 1.2-2.1 GiB stock). No counter
shows it: not shmem, not the page cache, not the CUDA cache. The suspect is driver-side
state left by thousands of small `cudaHostAlloc` blocks. From the checkpoint headers
(`count_allocs.py` in that directory, the caching host allocator simulated over the loader's 3-shard window):

| per rank | eager tensors (pinned requests before) | new `cudaHostAlloc` before | slabs | new `cudaHostAlloc` with slabs | peak pinned before / after |
|---|---:|---:|---:|---:|---:|
| EP1 target | 93,680 | 7,224 | 330 | 27 | 7.9 / 5.8 GiB |
| EP1 draft | 2,401 | 2,401 | 11 | 11 | 3.3 / 2.5 GiB |
| EP2 target | 47,600 | 3,738 | 610 | 48 | 15.0 / 10.9 GiB |
| EP2 draft | 2,401 | 2,401 | 32 | 32 | 10.5 / 7.8 GiB |

- `_EagerShard` packs the shard's eager tensors, in the order the loader yields them (sorted
  names), into slabs of `DSV41_FAST_LOAD_SLAB_MB` (default 256, rounded down to a power of two), each
  tensor 4 KiB aligned. A tensor bigger than a slab gets its own buffer. Every tensor handed to
  the model, and every TP slice, is a view into its slab. The bytes, dtypes and shapes are unchanged.
- Full slabs all fall in one power-of-two bin of torch's caching host allocator, so a slab freed
  by an earlier shard is reused by the next one. The same packing also drops the power-of-two
  rounding waste per tensor (EP1's 1.47 MB slices took 2 MB blocks), so the peak is ~28 % lower.
- A slab is freed when the last view into it dies. `__exit__` now drops every future nobody
  asked for, because such a future would keep its tensor, and so the whole slab, alive. The model's
  `load_weights` copies each tensor into its parameter and keeps nothing. The only holders that
  outlive one step are the pair buffers (compressor `wkv`/`wgate`, `wq_a`/`wkv`, and the
  bf16 `wo_a` weight/scale dequant), and those are empty when the load ends. After each load, `_release_all` checks each slab's storage
  weak reference. If a slab is still alive, it logs the tensor names that hold it
  (`slab of ... still alive after the load`); after the target load that is an error (the boot
  stops before the KV pool is sized), after the draft load a warning.
- Memory is bounded. The loader window holds whole shards as before, and the paced copies add at
  most `DSV41_FAST_LOAD_INFLIGHT_GB` plus one slab: a copy from a slab charges the whole slab, once
  for all its copies in flight, because one slow small copy keeps the whole slab alive after its
  siblings finish. Sources outside a slab charge their own bytes.
- `DSV41_FAST_LOAD_SLAB_MB=0` restores one pinned buffer per tensor.
- `DSV41_FAST_LOAD_MEMLOG=1` adds one `DSV41 fast load memlog: phase=...` line per load. It logs
  MemTotal, MemAvailable, SecPageTables and the other meminfo counters, CUDA reserved, and NVML's per-process GPU memory
  (host PIDs), plus the pinned requests, slabs and peak live of this module and
  torch's process-wide `num_host_alloc` / `num_host_free`. It also gives `unaccounted_gib` (MemTotal minus
  the counters minus CUDA reserved) and `nontorch_since_load_start_gib` (MemAvailable lost since the
  target's `load_weights` began, minus the CUDA reserved growth). It also works with `DSV41_FAST_LOAD=observe`.

Alternative that was not taken: `PYTORCH_CUDA_ALLOC_CONF=pinned_use_cuda_host_register:True`
(malloc + `cudaHostRegister` instead of `cudaHostAlloc`). It changes the driver path but not the
count: 93k per-tensor requests would still become thousands of registrations. glibc serves the
sub-32 MB ones from its arenas, and the first eager loader showed those arenas keep freed
chunks (~11 GB resident). The flag is also process-wide, so it moves SGLang's own pinned
buffers too. It is worth one A/B boot on top of slabs only if slabs do not move the non-torch
number. At 256 MiB every slab is above the mmap threshold, so registration would then be clean.

Tests: `tests/test_fast_load_slab.py` (image build, CPU; the slabs are mmaps without CUDA, same
packing and views). It covers the layout invariants, bytes equal to stock `safe_open` for every rank at EP1 and EP2
(target and draft, slab 4 MiB / 256 MiB / off, through FusedMoE's own loaders in the image), one
buffer per slab, lifetimes (unrequested futures, escaped plain/sliced views reported by name and
freed), a paced load through a copy of sglang's buffered iterator (budget kept, live slabs
within window + budget + 2 slabs), and the same load with one slow small copy per slab (live slabs
still within that bound; with the old per-tensor charge they are not).

Booted on the fleet 2026-09-24 (EP1 target): 330 slabs, 0 alive after release, and the KV pool
0.73M tokens larger than with one pinned buffer per tensor.

## Measured (production image + fast load, 2026-09-18)

| | before | after |
|---|---:|---:|
| target `load_weight`, rank 0 / 1 / 2 / 3 | 225-246 / 95 / 246 / 114 s | 71-74 / 83 / 73 / 71 s |
| draft `load_weight` | 38-50 s | 3-6 s |
| `Engine startup timings: load_weight` | 276-278 s | 77-89 s |
| `scheduler_e2e` (start to ready) | 343-354 s | 124-129 s |
| `max_total_num_tokens` (same image, gate off vs on, same night) | 7.47-7.82M | 7.27M pinned buffers; 6.2-6.8M mmap buffers |
| decode prose c1 / c4 agg, code c1, structured c1 | 57.0 / 118.6 / 107 / 115 | 56.8 / 120.8 / 92-107 / 99-111 |
| prefill 4k / 32k / 128k | 4070 / 4554 / 4364 | 2747-3227 / 4209-4232 / 4474-4494 |
| needle 131k | PASS | PASS |

Decode and prefill numbers are single sparkDash runs right after each boot and sit inside the
run-to-run spread of the production table (the first point after a boot reads low). Raw outputs:
[`docs/results/fastload-20260918/`](https://github.com/knapcio/DeepSeek-V4.1-Flash-4x-DGX-Spark-TP4/tree/main/docs/results/fastload-20260918) in the TP4 repository.

## Correctness

- `tests/test_fast_load_slab.py` (image build): slab packing, byte equality at EP1/EP2, slab
  lifetimes and the escape report, paced loading within the memory bound (section above).
- `tests/test_fast_load_pacing.py` (runs in every image build): the paced submit never exceeds
  the budget, completes every copy, leaves the synchronous path untouched and leaks no permit on
  an exception.
- In-image checkpoint test (needs the weights mounted, not part of the build): driving the real
  `buffered_multi_thread_safetensors_weights_iterator` in draft phase over all 48 shards yields
  exactly the 2401 `mtp.*` tensors, 45 shards skipped; every eager tensor of a target shard and
  of the draft is bitwise equal (`view(uint8)`) to the stock `safe_open` tensor across all five
  dtypes present (`float8_e4m3fn`, `float8_e8m0fnu`, `int8`, `bfloat16`, `float32`); RSS returns to
  baseline after a shard's tensors are released.
- The model's own loader still does every narrow and copy; the adapter only changes where the
  source bytes are resident.

## Why the KV pool shrank, and the incident

SGLang sizes the KV pool from `psutil.virtual_memory().available` on integrated GPUs
(`get_available_gpu_memory`, "these devices use sysmem as device mem"). With the mmap-buffer
loader, `MemAvailable` at that moment was 1.4-1.7 GB lower than with the stock loader on the same
image, while the process (RSS, anonymous maps), the CUDA caching allocator (`memory_reserved`
identical to the byte) and the kernel counters (slab, page tables, unevictable) all matched. What
did not show anywhere is driver memory: a standalone test on a worker copying 3 GB from pageable
host memory with 24 threads leaves ~0.4 GB missing from `MemAvailable` after the buffers are
freed, a second burst adds nothing, and copies from pinned memory leave nothing. So the eager
buffers are now pinned (torch's caching host allocator, returned with `_host_emptyCache` before
the engine measures). That closed most of the gap: `MemAvailable` after the draft load is
35.1 GB against 35.9 GB for the stock loader (pool 7.27 M vs 7.75 M). The last ~0.8 GB shows up
only as `Mapped` (file pages mapped by the scheduler process, 1.5 GB vs 0.6 GB) with the CUDA
allocator, anonymous memory, slab and page tables identical; not chased further. The raw snapshots are in [`docs/results/fastload-20260918/`](https://github.com/knapcio/DeepSeek-V4.1-Flash-4x-DGX-Spark-TP4/tree/main/docs/results/fastload-20260918)
(`control-boot-observe*.txt`, `control-late.txt` for the stock loader; `fastload-verify-v*.txt`
for the fast-load boots).

The checkpoint test that validates bitwise equality held the whole draft (7.9 GB) in memory; run
with pinned buffers on a *serving* worker it pushed the node into a state where SSH stopped
answering and the fleet went down. It now compares tensor by tensor and holds nothing, and it
must only be run with the fleet stopped. The pinned-buffer boot ran the next morning
(`fastload-verify-v13-pinned.txt`, control `control-boot-observe3.txt`).

## Host-memory fragmentation after a load (issue #23)

Two independent reports (a 2-node EXL3 pack and a 4-node FP4 pack) saw a node wedge hard after a
model load: SSH times out while ICMP still answers, `nvidia-smi` blocks, no OOM kill, and the kernel
log shows `NVRM: ... Out of memory [NV_ERR_NO_MEMORY] ... _memdescAllocInternal` with gigabytes
still free. On GB10 driver allocations are unmovable host pages; after a load the order-8+ free
blocks in `/proc/buddyinfo` zone `Normal` can be near zero, and the driver's contiguous request
cannot be satisfied (MemAvailable stays healthy, so `scripts/verify/memguard.py`, which watches
MemAvailable, does not fire). This is the reporters' diagnosis, not something measured on this
repo's fleet. Their repair is one compaction after the load, as root:

```bash
awk '{t=0; for(i=13;i<=15;i++) t+=$i; s+=t} END{print "order8+ (>=1MiB):", s}' /proc/buddyinfo
echo 1 | sudo tee /proc/sys/vm/compact_memory     # 0.04-0.06 s, no memory cost
```

Do not raise `vm.min_free_kbytes` / `vm.watermark_scale_factor` instead (reported to livelock a loaded
box). The loader above already drops each shard's page cache when its handle closes
(`POSIX_FADV_DONTNEED` in `adapter/fast_load.py`) and again after `load_weights`, rather than only at
the end; whether that narrows the fragmenting window has not been measured.
