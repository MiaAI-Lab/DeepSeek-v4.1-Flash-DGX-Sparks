# Running TP4 on a switchless 4-node ring

This note documents how to run the 4-node profile (TP4/EP4) on four DGX Sparks
wired as a **ring with no RoCE switch** — the alternative the README mentions
("or a ring with NCCL routed over it") but does not spell out. The TP4 profile
was only ever validated with `./start-tp4.sh doctor`; the settings below are
what a real boot on a 4-node ring needs.

Everything here is opt-in. With `NCCL_SWITCHLESS_RING_ONLY` unset, `start.sh`
behaves exactly as before.

## Why the default configuration cannot boot

A 4-node ring has no direct fabric path between opposite nodes (rank0 ↔ rank2,
rank1 ↔ rank3). IP traffic between them is forwarded by the two transit nodes,
so `ping`/TCP across the diagonal works fine.

NCCL ignores that forwarding plane. It builds **both** a ring and a tree
topology, and the tree wants rank0 to reach rank2 directly:

```
NCCL INFO Trees [0] 2/-1/-1->0->-1            <- tree still planned
NCCL INFO Connected all rings, use ring PXN 0 GDR 0
...
RuntimeError: NCCL error: unhandled system error
```

The ring itself connects (`Connected all rings`); the boot then dies in
`ncclTransportTreeConnect` while setting up the tree's queue pairs, because
RoCE queue-pair establishment does not follow the IP route that would take the
traffic through a transit node.

## What is required

1. **A NCCL build that can skip the tree/PAT transports.** Upstream
   `FujitsuPolycom/sparkring` carries the patch set under
   `spark_transport/nccl/` — in particular `nccl-2.30.7-switchless-cycle.patch`,
   which adds `NCCL_PARAM(SwitchlessRingOnly, "SWITCHLESS_RING_ONLY", 0)` and
   makes NCCL skip tree and PAT transport setup when it is set.
   `runtime/sparkring/source_image/build_nccl.py` in that repo applies the
   patches; build the library on one node and copy it to the others.
2. **That library installed where the container will actually load it** —
   see "Pitfall 2" below.
3. The fabric interface, HCA list and RoCE v2 GID index, as for a switched
   deployment.

## Building the patched NCCL

The library used here was built locally on one of the Sparks; nothing in this
repository builds it. The recipe, for reproducibility:

| item | value |
|---|---|
| NCCL source | upstream `nccl` @ `73cf112295c33aee2b895f329f592f2a9b4b0f97` |
| patch | `spark_transport/nccl/nccl-2.30.7-dual-pci-domain.patch` (31610 B, md5 `1ea3719be357b2cebd5af169dda16fce`) |
| build script | `runtime/sparkring/source_image/build_nccl.py` |
| target arch | `-gencode=arch=compute_121,code=sm_121` (GB10 / SM121) |

`build_nccl.py` verifies the source archive and the compiler wrapper against
`source-lock.json`, runs the CPU-only `tests/routing_handle/compat.cc` test
first, then builds with

```bash
make -C src -j16 src.build CUDA_HOME=<cuda> CUDA_LIB=<cuda>/lib \
     BUILDDIR=<build> NVCC=<wrapper> \
     NVCC_GENCODE=-gencode=arch=compute_121,code=sm_121
```

and refuses to install the result unless it is an AArch64 ELF64 whose hash
matches the locked `runtime.nccl_sha256`.

Note that `nccl-2.30.7-dual-pci-domain.patch` already carries the
switchless-cycle change; `nccl-2.30.7-switchless-cycle.patch` is the same
change as a standalone patch.

The binary this guide was written against:

| property | value |
|---|---|
| size | 63 842 376 bytes |
| md5 | `907db791051b43bd4841ae572aab9c25` |
| BuildID | `05cb631cff46c79fc7e655496f951df1229cfb3f` |
| format | ELF 64-bit LSB shared object, ARM aarch64, debug_info, not stripped |

Check any candidate library before trusting it:

```bash
strings libnccl.so.2.30.7 | grep -c SWITCHLESS_RING_ONLY   # > 0 required
strings libnccl.so.2.30.7 | grep -c SKIP_TREE_CONNECT      # 0 = no skip-tree patch
```

The patched library only adds the ring-only switch and the dual-PCI-domain
routing; it does not touch NCCL's compute kernels.

## Configuration

In `.env.tp4` (the commented block in `.env.tp4.example` shows the same):

```bash
NCCL_SWITCHLESS_RING_ONLY=1        # opt in
NCCL_HOST_DIR=$HOME/nccl-2.30.7    # patched libnccl.so.2.30.7 lives here
NCCL_ALGO=Ring
NCCL_IB_SUBNET_PREFIX_LEN=24
NCCL_MIN_NCHANNELS=4
NCCL_MAX_NCHANNELS=4               # TP4 memory optimisation keeps this at 4
NCCL_P2P_LEVEL=SYS
```

Setting `NCCL_SWITCHLESS_RING_ONLY=1` does two things:

* `start.sh` injects the ring-only NCCL settings into every rank (head and
  workers), and
* the patched library is bind-mounted **over** the container's pip-installed
  NCCL instead of being added to `LD_LIBRARY_PATH`.

Then boot as usual:

```bash
./start-tp4.sh doctor     # expect the same two warnings as a switched setup
./start-tp4.sh serve      # ~14 min cold start on 4x GB10
```

## Confirming it worked

The patched library logs these lines; if you do not see them, the ring, not the
transports, is what is failing:

```
NCCL INFO NCCL_SWITCHLESS_RING_ONLY set by environment to 1.
NCCL INFO Connected all rings, use ring PXN 0 GDR 0
NCCL INFO Tree transport setup disabled by NCCL_SWITCHLESS_RING_ONLY
NCCL INFO PAT transport setup disabled by NCCL_SWITCHLESS_RING_ONLY
```

Note that the tree is still *planned* (`NCCL INFO Trees [0] 2/-1/-1->0->-1` is
printed as usual) — only its transport setup is skipped, which is what used to
fail.

## Pitfalls

### 1. `NCCL_SKIP_TREE_CONNECT` is a different patch

`sparkring` ships two independent patches:

| patch | variable | effect |
|---|---|---|
| `nccl-2.30.7-switchless-cycle.patch` | `NCCL_SWITCHLESS_RING_ONLY` | skips tree **and** PAT transport setup |
| `nccl-2.30.7-skip-tree-pat.patch` | `NCCL_SKIP_TREE_CONNECT` | returns early from `ncclTransportTreeConnect` / `ncclTransportPatConnect` |

A library built with one patch does not recognise the other's variable, and
stock NCCL ignores both. Setting `NCCL_SKIP_TREE_CONNECT=1` against a stock or
`switchless-cycle`-only library therefore does nothing at all: the tree is
still built and the boot still fails. `start.sh` sets both, so either build
works.

### 2. Do not put the patched library on `LD_LIBRARY_PATH`/`LD_PRELOAD`

The base image already carries a pip-installed NCCL at

```
/opt/sglang/lib/python3.12/site-packages/nvidia/nccl/lib/libnccl.so.2
```

Adding a second one via `LD_LIBRARY_PATH` (or `LD_PRELOAD`) makes DeepEP abort
before NCCL is initialised:

```
AssertionError: Duplicate NCCL runtime found in the current system:
/nccl/libnccl.so.2.30.7 vs /opt/sglang/.../nvidia/nccl/lib/libnccl.so.2
```

raised by `check_nccl_so()` in `deep_ep/__init__.py`, which requires exactly
one NCCL runtime.

The working arrangement is to keep exactly one copy: bind-mount the patched
library **on top of** the pip path.

```bash
-v $HOME/nccl-2.30.7/libnccl.so.2.30.7:/opt/sglang/lib/python3.12/site-packages/nvidia/nccl/lib/libnccl.so.2:ro
```

`start.sh` does this via `NCCL_PIP_SO` / `NCCL_OVERLAY_PIP`; set
`NCCL_OVERLAY_PIP=0` to get the old directory-mount + `LD_LIBRARY_PATH`
behaviour back.

### 3. RoCE routing is not IP routing

Forwarding across the diagonal works at the IP layer (TCP to the far node's
fabric address succeeds), which makes the ring look healthy. It does not help
NCCL: queue-pair setup is per link and does not consult the routing table.
That is the whole reason the tree transport has to be skipped rather than
"routed".

## Running without NFS

The profiled workers normally read the checkpoint from the head's NFS export.
Each Spark can also hold it locally; no `start.sh` change is needed, because
the worker only refers to a docker volume *name*:

```bash
# on every node: flatten the HF snapshot into a plain directory
cp -rlL ~/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V4.1-Flash/snapshots/<rev>/. ~/dsv41-model/
# and make $NFS_VOLUME a local bind volume
docker volume create --driver local --opt type=none \
  --opt device=$HOME/dsv41-model --opt o=bind dsv41-weights
```

then set `MODEL_DIR=$HOME/dsv41-model`, `NFS_VOLUME=dsv41-weights` and
`NFS_SHARE=0` in `.env.tp4`. `NFS_SHARE=0` matters: with it enabled,
`cmd_share` would recreate `dsv41-weights` as an NFS volume and discard the
bind. `cp -rlL` shares inodes with the HF blob store, so the flattened
directory costs no extra disk space.

`./start-tp4.sh doctor` still prints *"spark2/spark3 use docker NFS volume
dsv41-weights"* — that line is hard-coded and does not mean NFS is in use.

## Tested configuration

Four DGX Spark (GB10) wired as a ring, TP4/EP4, 1M context, DSpark k=5, weights
local on every node (`NFS_SHARE=0` + bind volume), ring settings as above.
Booted from `./start-tp4.sh serve` on 2026-09-11.

| check | result |
|---|---|
| cold start | ~10 min to `healthy` |
| head container | `Up 10 minutes (healthy)` |
| 3 worker containers | `Up 10 minutes (healthy)` |
| `GET /v1/models` | `deepseek-v4.1-flash`, `max_model_len: 1048576` |
| `./start-tp4.sh smoke` (`19+23`) | `42`, `finish_reason: stop` |
| fatal errors in the log | none |

Evidence that the ring-only path is what ran:

```
[TP0 EP0] sglang is using nccl==2.30.7
[0] NCCL INFO NCCL version 2.30.7+cuda13.2
[0] NCCL INFO Connected all rings, use ring PXN 0 GDR 0
[0] NCCL INFO NCCL_SWITCHLESS_RING_ONLY set by environment to 1.
[0] NCCL INFO Tree transport setup disabled by NCCL_SWITCHLESS_RING_ONLY
[0] NCCL INFO PAT transport setup disabled by NCCL_SWITCHLESS_RING_ONLY
```

Evidence that the overlay, not `LD_LIBRARY_PATH`, is what is mounted
(`docker inspect dsv41-head`):

```
mounts:
  /home/<user>/nccl-2.30.7/libnccl.so.2.30.7
      -> /opt/sglang/lib/python3.12/site-packages/nvidia/nccl/lib/libnccl.so.2
env:
  NCCL_SWITCHLESS_RING_ONLY=1
  NCCL_ALGO=Ring
  NCCL_SKIP_TREE_CONNECT=1
  NCCL_IB_SUBNET_PREFIX_LEN=24
  NCCL_MIN_NCHANNELS=4
  NCCL_MAX_NCHANNELS=4
  NCCL_P2P_LEVEL=SYS
  LD_LIBRARY_PATH=/usr/local/nvidia/lib:/usr/local/nvidia/lib64:...   # no /nccl
```

All three workers show the same mount and the same variables
(`docker inspect dsv41-worker` on each host).

Further readings from the same boot:

* Engram rows served from the local per-rank packed shards (`packed=True`);
* per-rank KV budget: `available_bytes=26.72 GB`,
  `bytes_per_full_token=1670.75`, `MAX_TOTAL_TOKENS=4000000`;
* `/v1/responses` (streaming and non-streaming) and `/v1/chat/completions`
  all return 200.

## Limitations

* Needs a patched NCCL; nothing in this repository builds it. The patch set and
  a build script live in `FujitsuPolycom/sparkring`.
* Only ring-shaped collectives are exercised on the diagonal; throughput on
  non-adjacent hops is bounded by the transit node's ConnectX forwarding.
* Numbers above are from one 4-node fleet; a switched deployment should be
  faster on the diagonal and remains the recommended layout.
