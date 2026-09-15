# Four DGX Sparks on a switchless ring

This repository's independently developed launcher integration was live-tested on
four DGX Sparks with local weights and a patched NCCL. The NCCL transport patch
is credited to [FujitsuPolycom/sparkring](https://github.com/FujitsuPolycom/sparkring).
It is an external prerequisite; this repository does not build or distribute it.

The ring transport is opt-in with `NCCL_SWITCHLESS_RING_ONLY=1`. Switched TP3/TP4
transport defaults remain unchanged. `.env.tp4.example` enables
`DSV41_SERIAL_WEIGHT_LOAD=1` as a GB10 memory safeguard, independently of topology.
Current `main` retains DSpark **k=3**, the thinking alias and reasoning budget 75,
the 32,768-token output cap, and loop abort. The historical ring benchmarks used
**k=5**; they do not establish k=3 performance.

## Why the ring needs a patch

Four nodes with two fabric ports each cannot form a complete graph. In this
ring, rank0–rank2 and rank1–rank3 have no direct link. Even if IP forwarding is
configured and TCP reaches the opposite node, RoCE queue-pair setup cannot use
an ordinary IP route through another Spark as a substitute for a direct link.

Stock NCCL can report `Connected all rings` and then fail in
`ncclTransportTreeConnect` while connecting the tree. `NCCL_ALGO=Ring` alone
was insufficient in the tested build. The `sparkring` switchless-cycle patch
recognises `NCCL_SWITCHLESS_RING_ONLY` and skips both tree and PAT transport
setup, while retaining subnet-aware port selection for adjacent peers.

## Physical cable and address example

Use four compatible DACs, each joining two CX7 ports, with no diagonal cables:

```text
  rank0                         rank1
  CX7-0 -------- cable 1 ------- CX7-1
  CX7-1                         CX7-0
    |                             |
  cable 4                       cable 2
    |                             |
  CX7-0                         CX7-1
  CX7-1 -------- cable 3 ------- CX7-0
  rank3                         rank2
```

```text
cable 1: rank0 CX7-0 <-> rank1 CX7-1
cable 2: rank1 CX7-0 <-> rank2 CX7-1
cable 3: rank2 CX7-0 <-> rank3 CX7-1
cable 4: rank3 CX7-0 <-> rank0 CX7-1
```

Each node's ordinary **management NIC** still connects to the same LAN for SSH,
Gloo, NCCL TCP bootstrap and `DIST_INIT_ADDR`. `HEAD_IP` and `WORKER_IPS` refer
to that LAN. Keep these separate from the point-to-point CX7 addresses below.
The `IB_HCA` launcher setting becomes `NCCL_IB_HCA` in each container.

Give each cable its own /24. These are illustrative private **fabric** addresses,
not management addresses or a dump of the live fleet. `.10` is the originating
rank in the edge list, including the rank3→rank0 closing edge.

| Cable | First end | Second end |
|---|---|---|
| 1 | spark1 `enp1s0f0np0` 10.10.0.10/24 | spark2 `enp1s0f1np1` 10.10.0.11/24 |
| 2 | spark2 `enp1s0f0np0` 10.10.1.10/24 | spark3 `enp1s0f1np1` 10.10.1.11/24 |
| 3 | spark3 `enp1s0f0np0` 10.10.2.10/24 | spark4 `enp1s0f1np1` 10.10.2.11/24 |
| 4 | spark4 `enp1s0f0np0` 10.10.3.10/24 | spark1 `enp1s0f1np1` 10.10.3.11/24 |

Confirm the Linux interface↔physical-port↔HCA mapping on each node; names can
vary. Configure the link IPs and a matching MTU at both ends; the tested fleet
used MTU 9000. Separate subnets let the patched NCCL choose the port connected
to each peer. A single subnet across all four cables was not tested. Avoid
`198.18.0.0/15` if local proxies use that benchmark range for fake DNS answers.

For example, inspect the tested HCA names on each node:

```bash
for d in /sys/class/infiniband/rocep1s0f0 /sys/class/infiniband/rocep1s0f1; do
  echo "$d: $(cat "$d/ports/1/state") $(cat "$d/ports/1/rate")"
done
```

In ring mode, `doctor` and `serve` require every port listed in `IB_HCA` to be
ACTIVE. Use exact HCA names, optionally with `:port`, separated by commas;
NCCL prefix/exclusion filters are not supported by ring preflight. Both commands
find a common IPv4-mapped RoCE v2 GID index across the selected ports on each
node. If `NCCL_IB_GID_INDEX` is set, they validate that index instead. The tested
fleet used index 3. Management IPs need not occur in a fabric GID table.

## Patched NCCL provenance and installation

The recorded external source references for the live-tested library are:

| Item | Reference |
|---|---|
| NCCL source commit | `73cf112295c33aee2b895f329f592f2a9b4b0f97` |
| Patch path in `FujitsuPolycom/sparkring` | `spark_transport/nccl/nccl-2.30.7-dual-pci-domain.patch` |
| Patch Git blob | `f4853e84334eaa3f980dce69a12660d8f1774d7c` |
| Recorded patch-changing commit | `4b9b6a0a213456b96400c5e1a1cf59c20ff892c8` |
| External build helper | `runtime/sparkring/source_image/build_nccl.py` |
| Target | AArch64, GB10 / SM121, `-gencode=arch=compute_121,code=sm_121` |
| Recorded library size | 63,842,376 bytes |
| Recorded ELF BuildID | `05cb631cff46c79fc7e655496f951df1229cfb3f` |

These references document the earlier build. They were not fetched or rebuilt
during the offline merge review. Use the external project's pinned build
instructions and source locks; a BuildID, file size or marker is not an
integrity/signature check.

For a future online setup, fetch the recorded **blob**, rather than the moving
`main` contents endpoint, and verify its Git object identity:

```bash
gh api repos/FujitsuPolycom/sparkring/git/blobs/f4853e84334eaa3f980dce69a12660d8f1774d7c \
  --jq .content | base64 -d > nccl-2.30.7-dual-pci-domain.patch
git hash-object nccl-2.30.7-dual-pci-domain.patch
# expected: f4853e84334eaa3f980dce69a12660d8f1774d7c
```

The recorded dual-PCI-domain patch includes the switchless-cycle change. Do
not blindly apply a second standalone copy of the same patch. The alternative
`NCCL_SKIP_TREE_CONNECT` patch is insufficient for this launcher's validation:
ring mode specifically requires a library carrying `SWITCHLESS_RING_ONLY`.
The launcher also supplies `NCCL_SKIP_TREE_CONNECT=1` for builds recognising it;
an unknown environment variable is silently ignored by NCCL.

Install the same trusted build on all four nodes as `libnccl.so.2.30.7` or
`libnccl.so.2` in the configured directory. If both are present, the versioned
file wins. Compare SHA-256 checksums across nodes after copying it.

```bash
# On each node, after installing the library:
sha256sum "$HOME/nccl-2.30.7/libnccl.so.2.30.7"
grep -qa SWITCHLESS_RING_ONLY "$HOME/nccl-2.30.7/libnccl.so.2.30.7"
```

`doctor` and `serve` check readability and the marker on every node. This is a
compatibility sanity check; it cannot authenticate a binary or prove its routing
implementation is correct.

## Configuration and overlay

Copy `.env.tp4.example` to `.env.tp4`, fill in the management and storage settings,
and uncomment its ring block:

```bash
NCCL_SWITCHLESS_RING_ONLY=1
NCCL_HOST_DIR=$HOME/nccl-2.30.7
NCCL_ALGO=Ring
NCCL_IB_SUBNET_PREFIX_LEN=24
NCCL_MIN_NCHANNELS=4
NCCL_MAX_NCHANNELS=4
NCCL_P2P_LEVEL=SYS
DSV41_SERIAL_WEIGHT_LOAD=1
```

The ring-only lines are commented in the shipped example. A copied profile
uses switched transport settings until explicitly enabled; those settings
cannot boot the tested switchless ring. Four channels are the tested ring
memory setting; the switched TP4 default remains eight.

`NCCL_HOST_DIR` names the head directory. A path under the head's `$HOME` maps
to the same relative path under each worker's `$HOME`; another absolute path
is used unchanged. Set `NCCL_WORKER_DIR` to an absolute directory to override
that mapping for all workers. Spaces and shell metacharacters are quoted.

Ring mode defaults `NCCL_OVERLAY_PIP=1` and requires it. The patched library is
bind-mounted read-only over the image's pip NCCL:

```text
/opt/sglang/lib/python3.12/site-packages/nvidia/nccl/lib/libnccl.so.2
```

Override `NCCL_PIP_SO` if a different image installs it elsewhere. Preflight
checks that this target exists using the locally installed serving image in a
disposable container with no network or GPU access. Build the serving image
on each node before expecting `doctor` to pass.

Adding a second library through `LD_LIBRARY_PATH` or `LD_PRELOAD` caused
DeepEP's `check_nccl_so()` to abort with `Duplicate NCCL runtime found`. The
overlay replaces the pip library without adding a second NCCL search path.
Outside ring mode, `NCCL_OVERLAY_PIP=1` can be enabled independently;
`NCCL_OVERLAY_PIP=0` retains the original directory mount and loader path.

The serial loader disables SGLang's asynchronous weight-copy decision to avoid
queued host-to-device copies retaining checkpoint pages in GB10's shared CPU/GPU
memory. It is passed to every rank and defaults off outside the TP4 safe profile.
It requires `sglang.srt.model_loader.utils.should_async_load`; an incompatible
SGLang version fails explicitly. It does not guarantee that every model or
context configuration will fit memory.

## Local weights with `NFS_SHARE=0`

For a ring, the tested arrangement stores the complete checkpoint locally on
every node. The head mounts `MODEL_DIR` directly; workers mount the Docker volume
named by `NFS_VOLUME`. Prepare the local directories and **worker** volumes before
serving. With a Hugging Face cache and destination on the same filesystem:

```bash
# On every node; replace REV with the downloaded snapshot revision.
cp -rlL "$HOME/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V4.1-Flash/snapshots/REV/." \
  "$HOME/dsv41-model/"
# On each worker, using a fresh volume name to avoid replacing an NFS volume:
docker volume create --driver local --opt type=none \
  --opt "device=$HOME/dsv41-model" --opt o=bind dsv41-local-weights
```

Hardlinks require the same filesystem and share data with the cache; keep the
checkpoint immutable. Use a full copy if source and destination differ in
filesystem, and budget the disk space accordingly. Set:

```bash
MODEL_DIR=$HOME/dsv41-model
NFS_VOLUME=dsv41-local-weights
NFS_SHARE=0
```

With `NFS_SHARE=0`, `share` is a no-op and `serve` refuses to replace containers
if the head directory or a worker volume lacks readable, nonempty `config.json`
or the expected shard files. It never falls back to NFS setup. `doctor` and `status` inspect the actual
worker volume, using the existing serving image without pulling another image.
These checks do not hash the whole checkpoint or verify its revision.

`NFS_SHARE=1` retains NFS operation: `:/` for a direct checkpoint export and
`:/NFS_EXPORT_NAME` beneath a reused HF cache export. A ring's opposite node
has no direct CX7 link to the head; any NFS layout must separately provide a
reachable TCP path and matching export ACLs. Do not copy the switched example's
`NFS_SERVER_IPS` blindly onto this fabric.

## Boot and verify

```bash
./start-tp4.sh doctor
./start-tp4.sh serve
```

Both commands validate the selected NCCL path on every rank. `serve` repeats
preflight after any required image build and blocks startup on failures before
replacing existing serving containers. It logs each node's selected GID index.

For a diagnostic boot, set `NCCL_DEBUG=INFO`; normal operation can return to WARN.
The historical successful boot included:

```text
NCCL INFO NCCL_SWITCHLESS_RING_ONLY set by environment to 1.
NCCL INFO Connected all rings, use ring PXN 0 GDR 0
NCCL INFO Tree transport setup disabled by NCCL_SWITCHLESS_RING_ONLY
NCCL INFO PAT transport setup disabled by NCCL_SWITCHLESS_RING_ONLY
```

Tree topology may still be printed: the patch skips transport setup, not tree
planning. Missing lines can also mean insufficient logging or an earlier loader
failure; inspect the first error before diagnosing cables. Inspect container
mounts on all ranks to confirm the pip overlay and absence of an added `/nccl`
loader path.

## Historical validation and remaining limits

The earlier live fleet used DGX OS 7.5.0, kernel `6.17.0-1031-nvidia`, driver
`580.173.02`, Docker 29.6.2, and the arm64 `lmsysorg/sglang:dev-dsv41` base image.
All four ranks served at TP4/EP4, 1M configured context, 8M pinned KV tokens,
DSpark k=5, serial loading and local packed Engram shards. The model listing,
`19+23`, thinking, tool calls and concurrent serving checks passed. See
[the benchmark report](tp4-switchless-ring-results.md) for measurements and raw
results, including the distinction between estimated window speed and measured
wall-clock throughput.

The final merge review used host-side mocks and tests only. It did not re-run
the modified launcher on Sparks, rebuild the external library, test k=3 ring
performance, or repeat a 1M-context needle. Port/GID checks cannot establish
cable order, end-to-end reachability, identical model revisions, or sufficient
memory on a live cluster. The result is specific to this four-node ring; it is
not a general multi-hop RoCE fabric or a switched-versus-ring A/B benchmark.
