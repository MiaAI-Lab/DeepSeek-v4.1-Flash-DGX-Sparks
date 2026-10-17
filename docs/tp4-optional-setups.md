# TP4: optional setups

The TP4 production line in `.env.tp4.example` uses the `Dockerfile.canary-roce` image with the prefill TP split ([docs/tp4.md](tp4.md)). This page explains the image layers on their own, and what the production line needs on a switchless ring. Measurements quoted here are from the day each layer was added; current numbers are in [docs/tp4.md](tp4.md#results).

## The upstream `dsv4.1` branch image

`Dockerfile.canary` keeps the same base image but swaps the SGLang python tree for the upstream `dsv4.1` branch at a pinned commit (`f80c91a4b`, 2026-09-16) and installs the kernel packages that branch pins (`sglang-kernel 0.4.7`, `sgl-deep-gemm 0.2.0`, aarch64 wheels from PyPI). It carries the branch's mHC / metadata / communication kernel work and decodes faster than the base image at every concurrency. It is pinned, not tracked: the build's `fetch` stage downloads the branch tree at the commit in `runtime/sources.manifest` and refuses it unless its sha256 matches; refreshing it means a new manifest row and a re-check that every adapter still finds its hook (the branch is refactoring file layout at the time of writing).

```bash
docker build -f Dockerfile.canary -t dsv41-4x-spark:canary .   # on the head and on every worker
# .env.tp4:
#   IMAGE=dsv41-4x-spark:canary
#   BUILD_DOCKERFILE=Dockerfile.canary   # or let `./start-tp4.sh build` build it everywhere
#   EXTRA_CONTAINER_ENV="DSV41_INDEXER_CHUNKED=1 SGLANG_DSPARK_FOLDED_SAMPLING=2"
./start-tp4.sh serve
```

`SGLANG_RUST_BUILD_MODE=never` is required on the branch images: the dsv4.1 tree probes a Rust toolchain to build its image preprocessor, and that `cargo --version` call can hang before the HTTP server starts (workers come up, `/health` never answers). `never` keeps the PIL image path.

`SGLANG_DSPARK_FOLDED_SAMPLING=2` matters: the branch folds only the greedy draft proposal into the CUDA graph by default, and sampled requests (temperature > 0, i.e. normal chat) would take the eager path. With it forced, sampled decode runs ~5 % slower than greedy on this image (it was equal on the base image); without it, ~9 % slower.

`./start-tp4.sh build` compiles `BUILD_DOCKERFILE` (default `Dockerfile`, and it has to stay inside the repository so the rsync that stages the workers sees it) and tags the result `$IMAGE`; `BUILD_ARGS` carries anything else the recipe needs, e.g. `--build-arg PIP_INDEX=https://<mirror>/simple` on a network without pypi.org.

## RoCEnante for the tensor-parallel all-reduces

`Dockerfile.canary-roce` adds the SG17 SGLang overlay from rhys101's eight-Spark work on top of the canary image: every tensor-parallel SUM all-reduce of at most 512 KiB (bf16/fp32) goes through b12x's one-shot RDMA all-reduce over both RoCE rails instead of NCCL, inside the CUDA graphs, with a transport health check at every result boundary (a stalled transfer fails the step instead of hanging the rank). The overlay was written for TP8; `runtime/roce_tp4_adapt.py` relaxes it to TP4/TP8 and to one or two rails. The RDMA proxy is plain C over libibverbs, compiled on first use inside the container (`B12X_ROCE_CACHE_DIR`).

```bash
docker build -f Dockerfile.canary-roce -t dsv41-4x-spark:canary-roce .    # on every node
# .env.tp4:
#   IMAGE=dsv41-4x-spark:canary-roce
#   EXTRA_CONTAINER_ENV="DSV41_INDEXER_CHUNKED=1 SGLANG_DSPARK_FOLDED_SAMPLING=2 SGLANG_ROCE_ALLREDUCE=1 SGLANG_ROCE_MAX_SIZE=2097152 B12X_ROCE_HCA=rocep1s0f0,roceP2p1s0f0 B12X_ROCE_CACHE_DIR=/state/b12x-roce B12X_COMPILE_CACHE_DIR=/state/b12x-compile"
./start-tp4.sh serve
```

`B12X_ROCE_HCA` lists the RDMA devices to stripe across (both rails of the switched fabric here; their port-1 GID at `NCCL_IB_GID_INDEX` must be populated). The boot log must show `RoCEnante ready: world=4 hcas=...` and later `ROCE_TP8_ROUTE ... bytes=491520`. Measured on this fleet (512 KiB route, same boot as the canary row): +3 % prose c1, +8 % structured c1, +3.5 % code c16 over the canary image; needle PASS at 131k and 262k; a soak of three 16-stream code/prose waves concurrent with a 262k cold prefill completed with zero transport errors. `SGLANG_ROCE_MAX_SIZE` defaults to the overlay's 512 KiB; the 16-request decode step's all-reduce is 983 KB, so 2 MiB (b12x's own default) routes it too: c16 aggregate +4.5 %, code c1 +3 %, structured +3 %, and sampled decode becomes equal to greedy. The cost is a new transport in the decode path: b12x reports one open issue where a rank wedged under long mixed-context traffic on an earlier revision ([b12x#313](https://github.com/local-inference-lab/b12x/issues/313)); the result-boundary health check in the overlay is the mitigation, and NCCL is one env change away (`SGLANG_ROCE_ALLREDUCE=0`).

## Switched fabric: partly cabled NICs and PFC traffic classes

Field notes from two switched four-Spark fleets (issues #2 and #38). Check them against your own fabric; none of this is the default.

- **`IB_HCA` must name ports that are up.** The example's `rocep1s0f0,rocep1s0f1` assumes both ports of the first NIC are cabled. If only the `f1` port of each NIC is wired (the `f0` ports report `PORT_DOWN` in `ibstat`), use one port from each NIC, e.g. `IB_HCA=rocep1s0f1,roceP2p1s0f1`, and point `B12X_ROCE_HCA` at the same live ports.
- **`NCCL_IB_TC`.** If your switch runs PFC on a specific traffic class (one fleet uses TC3/DSCP26, which is `NCCL_IB_TC=106`), NCCL traffic that is not tagged lands in a best-effort queue. The launcher does not set `NCCL_IB_TC`, but `EXTRA_CONTAINER_ENV` reaches the head and every worker, so append `NCCL_IB_TC=106` (or your class) there. RoCEnante's RDMA proxy is separate from NCCL; check its traffic class on the switch.
- **A host NFS server and the container exporter cannot share one machine.** If `nfs-kernel-server` ran from another recipe, stopping the service can leave `[nfsd]` kernel threads behind that stop the container's `rpc.nfsd` from registering with rpcbind. The container then looks healthy (2049 and 111 listening) while `rpcinfo -t 127.0.0.1 nfs 4` answers "Program not registered", which is exactly what the `share` readiness probe tests. Tear the host side down fully (`rpc.nfsd 0`, unmount `/proc/fs/nfsd`, `modprobe -r nfsd`, disable `rpcbind` and `rpc-statd`) or use `NFS_SHARE=0` with local weights.
- **`$HOME/.cache/huggingface` must be writable** by the launching user when the NFS export is reused (`nfs_publish_model` creates `dsv41-native` under it); a root-owned directory from an unrelated container aborts `serve` with a `mkdir ... Permission denied`.

## The production line on a switchless ring

The ring itself (cabling, the patched NCCL, `NCCL_SWITCHLESS_RING_ONLY=1`, local weights with `NFS_SHARE=0`) is set up as in [docs/switchless-ring.md](switchless-ring.md); the ring block of `.env.tp4.example` has the lines. Everything in the production line is fabric-independent except RoCEnante.

RoCEnante's one-shot all-reduce and all-gather write directly into every
peer's buffers, and a four-node ring has no direct link between opposite nodes (RoCE queue pairs do not
follow IP routing), so out of the box a ring runs the production line with `SGLANG_ROCE_ALLREDUCE=0` and without
`DSV41_ROCE_GATHER`: the tensor-parallel collectives then go through the patched NCCL (NCCL's small-message
floor is ~56 us against ~16 us per all-reduce, ~90 all-reduces per decode step). rsync (@rchmagos) got RoCEnante running
on a four-node ring by adding the opposite-node path from
[FujitsuPolycom/sparkring](https://github.com/FujitsuPolycom/sparkring) (commit `f16b5f4`, research-only): the neighbours'
ConnectX-7 forward the opposite-node traffic in hardware, and `DSV41_ROCE_RING=1` loads sparkring's path-aware
RoCEnante (`b12x.comm.roce_ring`, fetched at build). Setup, the planner (`scripts/ring_mesh/`) and measurements are in
[switchless-ring.md](switchless-ring.md#rocenante-on-the-ring-hardware-forwarded-opposite-node-paths): with the
v2 production line the ring measured qeval 81.8 / 84.7 / 85.9 tok/s over three runs (71-72/75) and a decode step within a few percent of
the switched numbers; prefill stays ~5 % lower from the ring's one-link bisection.

Before listing more than two devices in `IB_HCA`, read
["Devices past the second are never advertised"](switchless-ring.md#devices-past-the-second-are-never-advertised):
NCCL accepts the extra devices, publishes listener GIDs for only the first two, and
reports nothing, so a four-device board serves on half of it until the dual-PCI-domain
flags are in place; the port counters are the proof.

The ring's `EP_SIZE` is free (1, 2 or 4); only `NNODES == TP_SIZE == 4` is required, because the ring spans the tensor-parallel group. Note for the non-ring path: torch resolves `libnccl.so.2` through its own RPATH, so only `NCCL_OVERLAY_PIP=1` replaces the library torch uses; the `LD_LIBRARY_PATH` mount reaches `ctypes` users only.

## Prefill TP split (with either canary image)

`adapter/spark_prefill_dense.py` is rhys101's SG18 helper with its topology check relaxed from eight ranks to four or eight; `adapter/indexer_chunked_v3.py` calls it from inside the #39187 path when a prefill chunk has at least `SPARK_PREFILL_TP_MIN_ROWS` rows (1024) and the context is at least `SPARK_PREFILL_TP_MIN_CONTEXT` (32768). Each rank scores only its slice of the query rows, in row chunks of at most 2 GiB of fp32 logits, and publishes only the tail rows of the candidate masks; the top-k and block ids travel as an int all-gather (no floating-point collective). The bitwise CPU test covers the split at world sizes 1, 2 and 4, with and without tail-only publishing, down to one row per chunk. Enable with

```
EXTRA_CONTAINER_ENV="... SPARK_PREFILL_TP_SPLIT=1 SPARK_PREFILL_TP_MIN_CONTEXT=32768 SPARK_PREFILL_TP_MIN_ROWS=1024"
```

and look for `DSV41 prefill TP split (v3) rank=0 ... end=1024` in the boot log. Decode is unaffected (the draft runner keeps the stock path). The helper refuses thresholds below 32768 tokens / 1024 rows at boot (`ValueError`), and a sweep with that floor relaxed to 16k gained nothing outside the ±5–10 % run-to-run spread of short prefills, so 32768 stays. `runtime/flash_mla_sm120.canary.py` also carries SG18's scratch zero-initialisation (masked candidates gather slot 0; keeping the scratch finite avoids a NaN through a zero probability).
