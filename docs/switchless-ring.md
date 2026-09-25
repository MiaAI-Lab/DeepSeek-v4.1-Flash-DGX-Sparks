# Four DGX Sparks on a switchless ring

This launcher's ring transport and NCCL-overlay foundation derives from
[PR #3](https://github.com/MiaAI-Lab/DeepSeek-v4.1-Flash-DGX-Sparks/pull/3)
by [@Saolence](https://github.com/Saolence). This PR rebases and extends that
work with current-main compatibility, GB10 UMA-safe serial loading, hardened
`NFS_SHARE=0`, all-rank preflight fixtures, OpenAI harnesses and additional
measurements. The underlying NCCL transport patch is credited to
[FujitsuPolycom/sparkring](https://github.com/FujitsuPolycom/sparkring); it is an
external prerequisite and this repository does not build or distribute it.

The ring transport is opt-in with `NCCL_SWITCHLESS_RING_ONLY=1`. Switched TP3/TP4
transport defaults remain unchanged. The ring block of `.env.tp4.example` carries
`DSV41_SERIAL_WEIGHT_LOAD=1` (commented, with the ring flags) as the GB10 memory
safeguard the tested ring used. The TP4 profile runs DSpark **k=5** (the value of the
historical ring benchmarks); `start.sh` keeps **k=3** as its default, with the thinking
alias and reasoning budget 75, the 32,768-token output cap, and loop abort.

The second half of this guide covers what the TP4 production line
([docs/tp4.md](tp4.md)) adds on a ring: both ConnectX-7 cards
([Devices past the second are never advertised](#devices-past-the-second-are-never-advertised)),
RoCEnante over hardware-forwarded opposite-node paths
([RoCEnante on the ring](#rocenante-on-the-ring-hardware-forwarded-opposite-node-paths)),
and measurements of the TP4 profile on a ring.

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
memory. It is passed to every rank and defaults off; uncomment it in the ring block
of `.env.tp4.example`.
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

## Devices past the second are never advertised

`NCCL_IB_HCA` accepts any number of devices, and NCCL says nothing when it cannot
use them. The switchless-cycle patch publishes at most **two** listener GIDs per
rank — `gidSlot < 2` in `net_ib/connect.cc`, in both the original and the patched
loop. Devices after the second are therefore absent from the handle every peer
receives, no peer can match their subnet, and their ports carry zero bytes. The
ring still forms and serves; it just runs on the first two devices.

The symptom is easy to miss because the channel plan looks right:

```
NCCL INFO NET/IB : Using [0]rocep1s0f0:1/RoCE [1]rocep1s0f1:1/RoCE \
                   [2]roceP2p1s0f0:1/RoCE [3]roceP2p1s0f1:1/RoCE [RO]
NCCL INFO Channel 02/0 : 0[0] -> 1[0] [send] via NET/IB/2
```

Both lines name device 2, and it still moves nothing. What actually happens is a
silent collapse onto the first device, visible only in the routing log:

```
NCCL INFO NET/IB: Subnet-aware routing: overriding dev 2 with dev 0
NCCL INFO NET/IB: Subnet-aware routing: overriding dev 3 with dev 0
```

`doctor` does not detect this; check the routing records and port counters below.

### Raising the cap

A four-Spark board exposes its ConnectX-7 functions through **two PCI root
domains** (`0000:` and `0002:` here), which NCCL discovers as four separate
devices:

```
[0] rocep1s0f0    pciPath=/sys/devices/pci0000:00/.../0000:01:00.0
[1] rocep1s0f1    pciPath=/sys/devices/pci0000:00/.../0000:01:00.0
[2] roceP2p1s0f0  pciPath=/sys/devices/pci0002:00/.../0002:01:00.0
[3] roceP2p1s0f1  pciPath=/sys/devices/pci0002:00/.../0002:01:00.0
```

FujitsuPolycom/sparkring's cumulative
[`nccl-2.30.7-dual-pci-domain.patch`](https://github.com/FujitsuPolycom/sparkring/blob/main/spark_transport/nccl/DUAL_PCI_DOMAIN.md)
raises the bound to four behind a flag, and adds a fallback that substitutes a
device **within the same PCI root** rather than collapsing across domains. It is the
patch recorded under [Patched NCCL provenance and installation](#patched-nccl-provenance-and-installation)
above and already contains the switchless-cycle changes, so it must not be layered over them.

```ini
NCCL_IB_EXTENDED_IPV4_GIDS=1     # publish up to four IPv4-mapped listener GIDs
NCCL_IB_PRESERVE_PCI_DOMAIN=1    # substitute within the selected PCI root
NCCL_IB_ROUTE_DIAGNOSTICS=1      # one record per final QP: which device it landed on
NCCL_IB_QPS_PER_CONNECTION=1
```

Set `IB_HCA` to all four devices and the ring uses both planes. Every value has to
reach every rank, head and workers alike: put the four flags in `EXTRA_CONTAINER_ENV`
in `.env.tp4`, which `start.sh` passes to the head and every worker.

The flags are read at NCCL init, so the effect is visible before any request:

```
NCCL INFO NET/IB ListenerRouting format=ipv4-v1 advertised=4 observed=4
```

`advertised=2` means the cap is still in force. The routing records then stop
collapsing: `overriding dev 3 with dev 2` stays inside the second PCI root instead
of reaching for `dev 0`.

### The second plane's network has to exist before those flags can reach it

The flags raise the *publication* bound; they do not give the second card a network.
On the four Sparks the second ConnectX-7 came cabled and up (200G, link detected) but
unconfigured: no IPv4, MTU 1500, and the RoCE v2 GID that `NCCL_IB_GID_INDEX` selects
(index 3 here) reading back all-zero. NCCL cannot match a subnet for a device that has
no GID at the selected index, so the channel plan names it and the port still carries
nothing:

```text
NET/IB : Using [0]rocep1s0f0 [1]rocep1s0f1 [2]roceP2p1s0f0 [3]roceP2p1s0f1
Channel 02/0 : 0[0] -> 1[0] [send] via NET/IB/2      <- moves nothing
```

`port_xmit_data` on `roceP2p1s0f0` stayed flat for the whole run while the first card
carried 100 % of inter-node traffic — the 0.00 GB column in the table below. Setting the
four flags without this step leaves the ports where they were: the flags decide whether a
device may be advertised, the addressing decides whether there is a GID to advertise.

Both ports of the second card are part of the ring, one cable each to the two neighbours,
with the same geometry as the first card (`f0` to the next rank's `f1`). Each node
therefore needs one address per cable, a 9000 MTU, and a route for the two /24s it does
not sit on: those subnets exist only at the IP layer, through a transit neighbour.

A minimal template, per node as `/etc/netplan/41-sparkring-plane2.yaml`. Cable `<i>` is
the leg rank`i` -> rank`(i+1) mod 4`; shown for rank0, the other three are the same file
with `i` shifted and the two /24s adapted (they are placeholders — use your own scheme):

```yaml
network:
  version: 2
  renderer: NetworkManager        # the Sparks' fabric ports are NetworkManager-managed
  ethernets:
    enP2p1s0f0np0:                # -> RDMA device roceP2p1s0f0
      addresses: [10.10.0.10/24]  # cable 0, this end
      dhcp4: false
      dhcp6: false
      mtu: 9000
      optional: true             # never block boot on it
      routes:
      - to: 10.10.1.0/24          # cable 1, one hop away
        via: 10.10.0.11           # rank1's f1, on this cable
    enP2p1s0f1np1:                # -> RDMA device roceP2p1s0f1
      addresses: [10.10.3.11/24]  # cable 3, this end
      dhcp4: false
      dhcp6: false
      mtu: 9000
      optional: true
      routes:
      - to: 10.10.2.0/24          # cable 2, one hop away
        via: 10.10.3.10           # rank3's f0, on this cable
```

`sudo netplan apply`, then confirm the four addresses and the MTU are up
(`ip -br addr`, `ip -d link show enP2p1s0f0np0`) and that the interface names still
map to the `IB_HCA` list you set. The network is only right when the counters agree:
with the addressing and the four flags in place the second root moved 63.60 GB under a
64k prefill plus 16 streams and took 49.7 % of inter-node traffic, and the routing
record reads

```text
NET/IB: Subnet-aware routing: overriding dev 3 with dev 2 preserving PCI root pci0002:00
```

instead of collapsing to `dev 0`. Keep that order when debugging — addressing first,
then the flags, then the counters. The `NET` subsys stays useful here: this record is
the only place that states which root a channel landed on.

Removing the plane takes two steps, not one: NetworkManager can write its own
`/etc/netplan/90-NM-*.yaml` stanzas for these interfaces, and any of those that survive
bring the addresses back on the next `netplan apply` (`grep -l enP2p1s /etc/netplan/*`).
If you are rolling the plane back completely, `rm` those too and flush the addresses
(`ip addr flush dev enP2p1s0f0np0 enP2p1s0f1np1`).

### Channel count

`NCCL_MIN_NCHANNELS` and `NCCL_MAX_NCHANNELS` decide how many channels share the
devices. Four channels over four devices gives one channel per device, which is
the mapping that reaches all of them:

```ini
NCCL_MIN_NCHANNELS=4
NCCL_MAX_NCHANNELS=4
```

These are the ring block's values in `.env.tp4.example`.

Eight channels over four devices still round-robins 0,1,2,3,0,1,2,3, so it is not
wrong, but a four-versus-eight comparison on this workload found no serving
benefit and 0.14 GiB more head-node shared memory
([sparkring#193](https://github.com/FujitsuPolycom/sparkring/issues/193)).

### What it is worth

Measured here on four Sparks, TP4 / EP2, DSpark k=5, one 64k prefill plus 16
concurrent streams, IB port counters before and after:

| port | PCI root | before | after |
|---|---|---:|---:|
| `rocep1s0f0` | 0000 | 65.45 GB | 32.20 GB |
| `rocep1s0f1` | 0000 | 65.45 GB | 32.19 GB |
| `roceP2p1s0f0` | 0002 | **0.00 GB** | **31.80 GB** |
| `roceP2p1s0f1` | 0002 | **0.00 GB** | **31.80 GB** |

Half the traffic moves to the second plane. The total is unchanged — this spreads
the same collectives over twice the ports, it does not make them smaller. The
ported case is bounded by what the ring was waiting on, not by cable bandwidth:
the ports ran at roughly 5 % of line rate under this load, so expect a low
single-digit prefill gain and no decode change, matching the
[contributor measurement](https://github.com/FujitsuPolycom/sparkring/blob/main/performance/records/transport/nccl-dual-domain-deepseek.md)
of +5.43–6.82 % prefill for this exact model and runtime. Verify with counters and
the routing records rather than trusting the channel plan.

### Diagnosing it

`ListenerRouting` and the routing records are logged at the `NET` level. With
`NCCL_DEBUG_SUBSYS=INIT,ENV` they never appear and the collapse is invisible:

```ini
NCCL_DEBUG=INFO
NCCL_DEBUG_SUBSYS=INIT,ENV,NET    # add NET while validating; drop it afterwards
```

## Pitfalls

* **`EP_SIZE` is free, `TP_SIZE` is not.** The ring needs `NNODES == TP_SIZE == 4`
  (the ring spans the tensor-parallel group). `EP_SIZE` only decides how the MoE
  experts are grouped. `./start.sh` still requires `EP_SIZE=4`. `./start-tp4.sh`
  also accepts 2 and 1: 2 on the base and canary images, 1 on the TP4 production line.
* **A wrong GID index is the usual failure.** All ports listed in `IB_HCA` must
  share one nonzero IPv4-mapped RoCE v2 GID; the preflight finds it or validates
  your `NCCL_IB_GID_INDEX` override. Management IPs need not appear in the GID table.
* **Do not also set `NCCL_SWITCHLESS_RING_ONLY=1` with a switched fabric.** The ring
  skips the tree, which is a performance loss when the tree is reachable.
* **A ring is not a non-blocking fabric.** Opposite ranks talk through a transit
  node, so a four-node ring's bisection bandwidth is one link, not two. Without the
  opposite-node paths below, expect the decode numbers under Measured rather than the switched ones.

## RoCEnante on the ring: hardware-forwarded opposite-node paths

**Status: research-only**, as sparkring labels its hardware-forwarded mesh (`plan.py` writes `"status": "research-only"`).
It is off by default, and nothing in this section changes a switched setup.

RoCEnante's one-shot all-reduce and all-gather write every rank's payload straight into every
peer's buffers, and a four-node ring has no link between opposite nodes, so out of the box a ring
runs the production line with `SGLANG_ROCE_ALLREDUCE=0` and the collectives go through the patched
NCCL (~56 us per decode-size all-reduce, ~90 of them per decode step). The missing path can be built
without a switch or extra cables, in the neighbours' ConnectX-7 hardware, with the design of
[FujitsuPolycom/sparkring](https://github.com/FujitsuPolycom/sparkring) (`cx7_hairpin_diagonal`,
commit `f16b5f4`):

- the sender's NIC re-tags the RDMA packets of the opposite-node queue pairs (flow label 16383, i.e.
  UDP source port 65535) from EtherType 0x0800 to 0x88b5 (an RDMA-TX flow rule, `mlx5-rdma-tx-rewrite-probe`);
- a `/32` route sends them to the neighbour on that cable;
- on the neighbour a `skip_sw` tc flower rule matches the tag and the two MACs, restores 0x0800,
  rewrites the destination MAC and redirects the packet out of its other port (mlx5 hairpin queues).

No CPU touches the forwarded packets and the kernel forwards nothing (`nstat IpForwDatagrams` stays
flat); a marked packet that misses the rule is dropped, never routed in software. Every rank uses
two paths per peer over its four RDMA functions: the neighbours over their own cable, the opposite
node through each neighbour (one path per PCIe domain). `DSV41_ROCE_RING=1` makes the SG17 overlay
load `b12x.comm.roce_ring`, sparkring's path-aware RoCEnante (provenance and local changes in
`runtime/b12x/roce_ring-provenance.json`), instead of `b12x.comm.roce`; unset, nothing changes.
`DSV41_L2_PREFETCH` hooks either package.

### What each step is worth

Same four-Spark ring, same day, one change at a time (qeval = `scripts/qeval.py`, 75 tasks at c1;
step time = the engine's `spec_verify_ct`, time to first token subtracted):

| Step | Effect |
|---|---|
| NCCL ring -> RoCEnante over the mesh (c5cee32 stack, 80 KB cap) | decode step -5.2 % (median over 51 qeval tasks, faster on 50), qeval median 76.3 -> 79.5 tok/s |
| proxy idle spins 200000 -> 20000000 (the SG17 value) | c1 step 40.6 -> 37.1 ms (-8.6 %), c2 -7 %, c4 -6 %, c8 -2 %; with 200000 the proxy was asleep in 27 % of samples during decode |
| `hairpin_queue_size` 8192, 256 KB cap, two-wave off | c2 -1 %, c4 -1.3 %; c1 unchanged (its collectives are 50-60 KB) |
| the TP4 stack's v2 (prefill SP, L2 prefetch, draft head) | c1-c8 step -1 to -3 %, prefill +17-18 % at 16k-128k, +10 % at 262k |

### Results

The TP4 production line at v2 (knapcio/DeepSeek-V4.1-Flash-4x-DGX-Spark-TP4 `7ac7123`) with the ring
additions, built with `Dockerfile.canary-roce`, measured on the four-Spark ring the same day, against
the switched v2 numbers (v2.1's `DSV41_PREFILL_SP_FP8` came later; it is fabric-independent and was not
in this run; v2.1 on the switched fabric gave prose c1 87.7 and prefill ~5.8-5.9k tok/s at 16k-128k, and
[docs/tp4.md](tp4.md) now shows v2.2). Raw
output: [`docs/results/tp4/ring-mesh-20260925.txt`](results/tp4/ring-mesh-20260925.txt).

| | Ring (this) | Switched (v2) |
|---|---:|---:|
| qeval median tok/s (75 tasks), pass | 81.8 / 84.7 / 85.9 (3 runs), 71-72/75 | 83.5, 72/75 |
| decode step, prose-type prompts | 33.3 ms (2.0 tok/step) | 33.0 ms (2.27 tok/step) |
| decode step, code-type prompts | 38.1 ms (3.74 tok/step) | 39.2 ms (3.87 tok/step) |
| sparkDash 1.8.8 prose c1 / c16 | 80.9 / 345.1 | 86.5 / 342.7 |
| code c1 / c16 | 120.5 / 446.6 | 122.6 / 438.3 |
| structured c1 / c16 | 150.2 / 547.7 | 152.4 / 572.2 |
| json c1 / c16 | 134.2 / 671.5 | 118.9 / 659.9 |
| prefill 16k-128k / 262k | 5,393-5,567 / 4,971 | 5,644-5,818 / 5,214 |
| phrase needle | PASS at 1,030,651 tokens (305 s) | PASS at 1,011,084 tokens (322 s) |

The step-time prompts differ (the switched run's are not published), so the two columns are near but not
identical acceptance. Prose c1 on sparkDash is the single-prompt case under Mesh pitfalls below. Prefill
stays 4-5 % under the switched fabric at 16k-128k and 5 % at 262k: the large prefill collectives run on
NCCL over the ring's one-link bisection. The KV pool was 5.56 M tokens on this boot and 6.33-6.35 M on
the two before it (the fast loader's boot-to-boot spread).

### Setup

1. **The ring as above**, with both planes addressed (four RDMA functions per node, MTU 9000, the
   RoCEv2 GID at index 3), and the NIC profile sparkring's hardware forwarding was qualified on:
   `hairpin_num_queues` 4, `flow_steering_mode` `hmfs`, eswitch `legacy`, `hw-tc-offload on` on all
   four fabric netdevs (sparkring's
   [driver configuration notes](https://github.com/FujitsuPolycom/sparkring/blob/main/docs/GLM53_SPARK_MTP3_MESH_QUICKSTART.md#connectx-7-driver-configuration-for-hardware-forwarding)).
   `scripts/ring_mesh/inventory.sh` prints all of it per node; `plan.py` refuses a node whose links,
   MTU, GID, TC offload or steering mode do not match.
2. **The source marker**, built on every node from a sparkring checkout at `f16b5f4`
   (source sha256 `8684a696…`; it built to `2828c07e…` here, the binary sparkring records):

   ```bash
   git clone https://github.com/FujitsuPolycom/sparkring ~/sparkring && git -C ~/sparkring checkout f16b5f4
   sudo install -d /opt/dsv41-mesh/bin
   cc -O2 -Wall -Wextra ~/sparkring/spark_transport/fabric/cx7_hairpin_diagonal/native/mlx5_rdma_tx_rewrite_probe.c \
      -o /tmp/mlx5-rdma-tx-rewrite-probe -libverbs -lmlx5 && sudo install -m 755 /tmp/mlx5-rdma-tx-rewrite-probe /opt/dsv41-mesh/bin/
   ```
3. **The plan**, from the head, with the hosts in TP rank order (the head, then `WORKER_HOSTS`):

   ```bash
   python3 scripts/ring_mesh/plan.py --sparkring ~/sparkring --out ring-mesh spark1 spark2 spark3 spark4
   ```

   It inventories the nodes over SSH, reads the cabling from the fabric subnets, numbers the ring the
   way sparkring's planner needs it (every f0 port cabled to the next node's f1; it refuses anything
   else), and has sparkring's planner build the RoCEnante selection: per node two `/32` routes, two
   tc rules and two markers. It writes `mesh-up-<host>.sh` / `mesh-down-<host>.sh` and `env.txt`,
   the `EXTRA_CONTAINER_ENV` additions with the per-rank peer maps already translated to the TP rank
   order (sparkring numbers the ring by cabling direction, which need not match it).
4. **Install on every node** with the engine stopped (applying the hairpin size re-initialises each
   fabric function); `$HOST` is that node's name as given to `plan.py`:

   ```bash
   sudo install -m 755 ring-mesh/mesh-up-$HOST.sh /opt/dsv41-mesh/mesh-up.sh
   sudo install -m 755 ring-mesh/mesh-down-$HOST.sh /opt/dsv41-mesh/mesh-down.sh
   sudo install -m 755 scripts/ring_mesh/hairpin.sh /opt/dsv41-mesh/
   sudo install -m 644 scripts/ring_mesh/dsv41-mesh.service scripts/ring_mesh/dsv41-mesh-marker@.service /etc/systemd/system/
   sudo systemctl daemon-reload && sudo systemctl enable --now dsv41-mesh
   ```

   `dsv41-mesh.service` runs at every boot: it waits for the fabric links, sets `hairpin_queue_size`
   (a `driverinit` parameter that resets at boot) and applies the routes, rules and markers; `mesh-up.sh`
   refuses a rule that did not land in hardware. Start the engine after it is active.
   On a first install, re-run `plan.py` once the unit is active. `plan.py` takes the RoCE size cap from the
   hairpin queue size it finds, so a plan made before the unit set 8192 emits the safe 80 KB cap (81920)
   instead of 262144.
5. **Verify the path** before booting the engine: every rule shows `in_hw` (`tc -s filter show dev
   <netdev> ingress`), and an RDMA write to the opposite node goes through the neighbour's rule, not its
   kernel (`ib_write_lat -d <dev> -x 3 --flow_label=16383` against the opposite node's port: ~10 us at
   61 KB here against 8.5 us to a direct neighbour; the neighbour's rule counters rise and its
   `IpForwDatagrams` does not).
6. **`.env.tp4`**: build `Dockerfile.canary-roce` as usual and append `env.txt` to the production
   `EXTRA_CONTAINER_ENV`, replacing its `B12X_ROCE_HCA`, `SGLANG_ROCE_MAX_SIZE` and `DSV41_ROCE_GATHER`.
   The boot log shows `RoCEnante ready: world=4 hcas=rocep1s0f0,rocep1s0f1,roceP2p1s0f0,roceP2p1s0f1`
   and `DSV41_L2_PREFETCH: RoCE collectives prefetch the next weights into L2`.

### The size cap and the hairpin queues

The forwarded traffic crosses the neighbour in a hairpin queue, and at the driver default
(`hairpin_queue_size` 1024) a burst of more than ~100 KB per message overflows it:
`rx_out_of_buffer` rises on the forwarding ports, the far end counts `out_of_sequence` /
`packet_seq_err`, and go-back-N retransmits make a 120 KB all-reduce 3-4x slower than NCCL. Measured
in CUDA graphs, all four ranks, drops summed over all 16 functions:

| all-reduce | 60 KB | 100 KB | 120 KB | 240 KB | 480 KB | 960 KB |
|---|---:|---:|---:|---:|---:|---:|
| queue 1024: us/op | 23 | 36 | 111-141 | 91-109 | 167-191 | 378-449 |
| queue 1024: drops | 0 | 163 | many | many | many | many |
| queue 8192, two-wave off: us/op | 24 | 35 | 40 | 60 | 95 | 274 (two-wave on) |
| queue 8192: drops | 0 | 0 | 0 | 0 | 0 | 64 |

At 1024 keep `SGLANG_ROCE_MAX_SIZE=DSV41_ROCE_GATHER=81920` (every c1 decode collective is 50-60 KB, so
c1 loses nothing); at 8192 the 256 KB cap also moves c2-c4 and the draft's 129 KB vocabulary gathers.
`plan.py` picks the cap from the smallest queue it finds. The package's two-wave schedule (direct
paths first, forwarded paths after, from 128 KB) only costs once nothing drops:
`B12X_ROCE_TWO_WAVE_THRESHOLD_BYTES=0` turns it off.

### Mesh pitfalls

- **Never re-initialise a fabric function, stop `dsv41-mesh` or a marker while the engine runs.**
  Opposite-node traffic stops, and a re-init drops every RDMA queue pair on that function; the RoCE
  health check then fails the step. `hairpin.sh` skips functions already at the value, so re-running
  it with the same value is safe.
- The marker rewrites every RDMA packet with UDP source port 65535 on its device, whatever the
  destination: keep that port reserved on the fabric.
- Without the mesh neither package has a path to the opposite node (`DSV41_ROCE_RING=1` or not): a
  ring without it runs `SGLANG_ROCE_ALLREDUCE=0` and no `DSV41_ROCE_GATHER`, as before.
- **Measuring.** The mesh changes the reduction order, so sparkDash's single greedy prose prompt takes
  a different text and its acceptance moves with it: here prose c1 read 72.0 on the mesh against 76.4 on
  NCCL while the step was 5 % faster. Compare step time (`spec_verify_ct`) or qeval's median over its
  75 tasks; that median itself moves 2-3 % from run to run (81.8 / 84.7 / 85.9 on one boot here), so take
  the median of several runs.
- CPU pinning does not help: the scheduler on the X925 cores measured -5.5 %, the proxy threads alone
  on dedicated X925 cores neutral.

Rollback: `DSV41_ROCE_RING=0 SGLANG_ROCE_ALLREDUCE=0` without `DSV41_ROCE_GATHER` in `.env.tp4`, then
`sudo systemctl disable --now dsv41-mesh` on every node (removes the markers, rules and routes).

## Measured on the TP4 profile

The ring without the opposite-node paths (collectives on NCCL): four GB10 Sparks in a ring (`a-b-c-d-a`, no switch), TP4 / EP2, 1M context,
DSpark k=5, local weights (`NFS_SHARE=0`), canary image, ring switch on. sparkDash's
benchmark panel, one engine, no other load. Measured by [@Saolence](https://github.com/Saolence)
on the TP4 profile's own ring launcher, whose switch and NCCL settings are the ones above.

Boot, first time:

```
NCCL INFO Connected all rings, use ring PXN 0 GDR 0
NCCL INFO NCCL_SWITCHLESS_RING_ONLY set by environment to 1.
NCCL INFO Tree transport setup disabled by NCCL_SWITCHLESS_RING_ONLY
NCCL INFO PAT transport setup disabled by NCCL_SWITCHLESS_RING_ONLY
parallel: nnodes=4 TP=4 EP=2
```

All four ranks healthy, `/health` 200, `--enable-cache-report` live
(cold request `prompt_tokens_details: None`, warm `{'cached_tokens': 1024}`).

### Prefill, cold

| context | 1k | 4k | 8k | 16k | 32k | 64k |
|---|---:|---:|---:|---:|---:|---:|
| prompt tokens | 1,041 | 4,116 | 8,213 | 16,405 | 32,793 | 65,555 |
| TTFT | 406 ms | 1.11 s | 2.01 s | 3.89 s | 7.67 s | 15.43 s |
| tok/s | 2563 | 3712 | 4081 | 4218 | 4274 | 4249 |

**Caveat, the same one as the TP4 prefill table in [docs/tp4.md](tp4.md):** sparkDash's prefill filler is one
repeated token, so every filler token hits the same Engram row and the row cache
(`DSV41_CACHE_GIB`) inflates the 16k-128k column by roughly 9-20 % (reported by
koldfrontier in MiaAI-Lab/DeepSeek-v4.1-Flash-DGX-Sparks#21). Treat the shape as real and
the absolute numbers as an upper bound; a cold single request with a 53,613-token natural
prompt took **17 s** on the same boot, which is the same order as the 32k row above.

### Decode, 400 output tokens

Prose:

| concurrent streams | 1 | 2 | 4 | 8 |
|---|---:|---:|---:|---:|
| aggregate tok/s | 60.5 | 82.9 | 116.4 | 186.1 |
| per stream | 60.5 | 41.4 | 30.3 | 24.0 |
| TTFT | 182 ms | 209 ms | 263 ms | 286 ms |

Code:

| concurrent streams | 1 | 2 | 4 | 8 |
|---|---:|---:|---:|---:|
| aggregate tok/s | 107.5 | 189.6 | 326.4 | 521.6 |
| per stream | 107.5 | 94.8 | 81.6 | 65.2 |
| TTFT | 257 ms | 313 ms | 381 ms | 533 ms |

Decode is where a ring is the right trade: aggregate scales close to linearly through
eight streams (prose 60.5 → 186.1, code 107.5 → 521.6) while per-stream decay stays
gentle, which is what the 16-slot decoder and DSpark are for. Prefill is where the
bisection shows: ranks on opposite sides of the ring talk through a transit node, so a
four-node ring's bisection is one link, not two. That is the price of having no switch,
not a way to beat one.

### Credits for this half

- The four-Spark ring launcher: [@Saolence](https://github.com/Saolence)
  ([#3](https://github.com/MiaAI-Lab/DeepSeek-v4.1-Flash-DGX-Sparks/pull/3)), carried forward in
  [#19](https://github.com/MiaAI-Lab/DeepSeek-v4.1-Flash-DGX-Sparks/pull/19) by
  [@carlosduque-incoxe](https://github.com/carlosduque-incoxe); the dual-PCI-domain findings and
  the TP4-profile ring measurements above are also @Saolence's.
- RoCEnante on the ring (`DSV41_ROCE_RING`, `b12x.comm.roce_ring`, `scripts/ring_mesh/`): rsync
  ([@rchmagos](https://github.com/rchmagos)), built on
  [FujitsuPolycom/sparkring](https://github.com/FujitsuPolycom/sparkring) (the hardware-forwarded
  opposite-node paths, the fabric planner, the RDMA-TX marker and the path-aware RoCEnante package),
  which derives from RoCEnante by [local-inference-lab/b12x](https://github.com/local-inference-lab/b12x).
- The NCCL transport patch: [FujitsuPolycom/sparkring](https://github.com/FujitsuPolycom/sparkring).
