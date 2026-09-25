"""CPU check for the compact replicated-split gather (adapter/replicated_split.py, DSV41_SPLIT_COMPACT_GATHER).

Four TP ranks are emulated in one process with a fake TP group:
  - the layout: wqkv_a (N=1792, 14 tiles over 4 ranks) -> own widths 512/512/384/384, destination
    columns 0/512/1024/1408, i.e. 64/64/48/48 packs at 0/64/128/176 with a 224-pack row stride;
  - a pure-torch mirror of the RoCE columns kernel's index math (stage the compact own shard from a
    strided view, scatter every source's packs to row * 224 + dst + col) gives the stock [M, n] output,
    window mode (rank 3 reads its 384 columns at column 128 of a 512-wide window) and exact-slice mode;
  - the minimal fallback (dim-0 gather + one cat of views) gives the stock output;
  - gather(): gate off == the old path; gate on runs the first-call self-check once per variant and
    layer, then the compact variant; a variant that differs on any rank is switched off for that layer
    (old path from then on) and the agreement is a MIN over ranks; nothing is checked at capture.
The CuTe kernel itself is covered by tests/test_roce_columns_gpu.py (one GPU, fake peers) and the
4-rank transport by sparks/diagnostics/dsv41-compact-gather/test_compact_gather_4rank.sh.
"""
import os

os.environ["DSV41_REPLICATED_SPLIT"] = "wqkv_a"
os.environ["DSV41_SPLIT_COMPACT_GATHER"] = "1"
import torch  # noqa: E402

import replicated_split as rs  # noqa: E402

W = 4
PACK_ELEMS = 8                                              # bf16 per 16-byte pack


def kernel_mirror(locals_, offs, widths):
    """What the columns kernel writes, pack by pack, from each rank's (strided) GEMM output."""
    packs = [w // PACK_ELEMS for w in widths]
    dst = [sum(packs[:s]) for s in range(W)]
    total = sum(packs)
    m = locals_[0].shape[0]
    # stage: rank s's send slot = its compact shard, row-major, packs[s] per row
    slots = []
    for s in range(W):
        src = locals_[s].contiguous().view(m, -1, PACK_ELEMS)          # [M, in_row_packs, 8]
        o = offs[s] // PACK_ELEMS
        send = torch.empty(m * packs[s], PACK_ELEMS, dtype=src.dtype)
        for i in range(m * packs[s]):
            row, col = divmod(i, packs[s])
            send[i] = src[row, o + col]
        slots.append(send)
    out = torch.empty(m * total, PACK_ELEMS, dtype=locals_[0].dtype)
    for s in range(W):
        for i in range(m * packs[s]):
            row, col = divmod(i, packs[s])
            out[row * total + dst[s] + col] = slots[s][i]
    return out.view(m, total * PACK_ELEMS)


class FakeRoce:
    """The runtime surface gather() uses, over a shared 4-rank mailbox."""
    max_gather_bytes = 2 * 1024 * 1024

    def __init__(self, world, box, corrupt=False):
        self.world, self.box, self.corrupt, self.prepared, self.calls = world, box, corrupt, [], 0

    def should_all_gather_columns(self, inp, columns):
        return inp.dim() == 2 and inp.shape[0] * max(columns) * inp.element_size() <= self.max_gather_bytes

    def prepare_columns(self, columns, dtype):
        self.prepared.append(tuple(columns))

    def all_gather(self, inp, *, dim, out, stream, columns, column_offset):
        rank = self.world.rank
        assert dim == -1 and out.shape == (inp.shape[0], sum(columns))
        assert rs.same_bits(inp, self.box["cols"][rank][0]) and column_offset == self.box["cols"][rank][1]
        self.calls += 1
        locs = [self.box["cols"][r][0] for r in range(W)]           # every rank's shard (pre-filled)
        offs = [self.box["cols"][r][1] for r in range(W)]
        out.copy_(kernel_mirror(locs, offs, list(columns)))
        return out


class World:
    rank = 0


class FakePynccl:
    def __init__(self, roce, disabled):
        self.roce, self.disabled = roce, disabled

    def _resolve_stream(self):
        return None


class FakeGroup:
    """Rank r's view of a TP group whose all_gather returns what the 4 ranks put in the mailbox."""

    def __init__(self, world, box, roce=None, pynccl_disabled=False):
        self.world, self.box = world, box
        self.world_size = W
        self.device_group = None
        self.pynccl_comm = FakePynccl(roce, pynccl_disabled) if roce is not None else None

    @property
    def rank_in_group(self):
        return self.world.rank

    def all_gather(self, t, dim=-1):
        self.box.setdefault("ag", {})[self.world.rank] = t
        parts = [self.box["ag"].get(r, t) for r in range(W)]     # filled by the driver below
        g = torch.cat(parts, dim=0)
        if dim in (-1, 1):
            g = g.reshape((W,) + tuple(t.shape)).movedim(0, 1).reshape(t.shape[0], W * t.shape[1])
        return g


def main():
    ranges, width = rs._ranges(1792, W)
    widths, dst = rs.compact_layout(ranges)
    assert widths == [512, 512, 384, 384] and dst == [0, 512, 1024, 1408]
    assert rs.pack_layout(ranges) == ([64, 64, 48, 48], [0, 64, 128, 176], 224)
    r2, _ = rs._ranges(25600, W)
    assert rs.pack_layout(r2) == ([800] * 4, [0, 800, 1600, 2400], 3200)
    assert rs.pack_layout([(0, 4), (4, 8)]) is None                  # 8-byte widths are not whole packs

    wins = [rs.window(1792, a, width) for a, _ in ranges]
    woffs = [a - w for (a, _), w in zip(ranges, wins)]
    assert woffs == [0, 0, 0, 128]
    g = torch.Generator().manual_seed(3)
    for m in (1, 2, 6, 7, 16, 96):
        bits = torch.randint(-32768, 32767, (m, 1792), generator=g, dtype=torch.int16)
        full = bits.view(torch.bfloat16)                            # every bit pattern, NaNs included
        for mode, offs, locs in (
                ("window", woffs, [full[:, w:w + width].contiguous() for w in wins]),
                ("slice", [0] * W, [full[:, a:b].contiguous() for a, b in ranges])):
            got = kernel_mirror(locs, offs, widths)
            assert torch.equal(got.view(torch.int16), bits), (m, mode, "kernel mirror")
            stacked = torch.stack([rs.padded(x, width) for x in locs])
            got = rs.assemble_stacked(stacked, ranges, offs)
            assert torch.equal(got.view(torch.int16), bits), (m, mode, "minimal")
            old = rs.assemble(torch.cat([rs.padded(x, width) for x in locs], dim=1), ranges, width, offs)
            assert torch.equal(old.view(torch.int16), bits), (m, mode, "old")
    print("  layout, kernel mirror, minimal and old reorder == stock bits at M 1..96 (window + slice)", flush=True)

    # gather() end to end over the fake group: gate off/on, self-check once per variant, fallback
    agrees = []

    def fake_agree(flag, group, dev):
        agrees.append(flag)
        return flag
    rs._agree = fake_agree
    m = 6
    bits = torch.randint(-32768, 32767, (m, 1792), generator=g, dtype=torch.int16)
    full = bits.view(torch.bfloat16)
    locs = [full[:, w:w + width].contiguous() for w in wins]
    st = (None, ranges, width, woffs)

    def run_all(groups, layers):
        """Every rank calls gather() on its shard; returns every rank's result."""
        world = groups[0].world
        box = groups[0].box
        # a collective needs every rank's part before any rank reads it: pre-fill the mailboxes
        box["ag"] = {r: rs.padded(locs[r], width) for r in range(W)}
        box["cols"] = {r: (locs[r], woffs[r]) for r in range(W)}
        outs = []
        for r in range(W):
            world.rank = r
            outs.append(rs.gather(layers[r], groups[r], locs[r], st))
        return outs

    class L:
        _dsv41_prefix = "model.layers.3.self_attn.wqkv_a"

    # 1) minimal (no pynccl / RoCE): first call checks once per layer, result == stock
    world, box = World(), {}
    groups = [FakeGroup(world, box) for _ in range(W)]
    layers = [L() for _ in range(W)]
    for _ in range(3):
        outs = run_all(groups, layers)
        assert all(torch.equal(o.view(torch.int16), bits) for o in outs)
    assert all(ly._dsv41_cg == {("minimal", 6): True} for ly in layers)
    assert agrees == [True] * W, agrees                                  # one check per rank, then none
    # 2) RoCE columns: pynccl enabled + runtime present -> compact kernel path after the check
    agrees.clear()
    world, box = World(), {}
    roce = FakeRoce(world, box)
    groups = [FakeGroup(world, box, roce=roce) for _ in range(W)]
    layers = [L() for _ in range(W)]
    for _ in range(2):
        outs = run_all(groups, layers)
        assert all(torch.equal(o.view(torch.int16), bits) for o in outs)
    assert all(ly._dsv41_cg == {("roce", 6): True} for ly in layers)
    assert roce.calls == 3 * W                                            # check call + 2 served calls
    assert roce.prepared == [tuple(widths)] * W
    assert agrees == [True] * (2 * W)                                    # prepare round + compare round
    # 3) pynccl disabled (eager prefill / non-graph decode) -> minimal, even with RoCE present
    world.rank = 0
    assert rs._roce_columns(FakeGroup(world, box, roce=roce, pynccl_disabled=True), locs[0], widths) is None
    assert rs._roce_columns(groups[0], locs[0], widths) is not None
    big = torch.zeros(4096, 512, dtype=torch.bfloat16)                   # 4 MiB > max_gather_bytes
    assert rs._roce_columns(groups[0], big, widths) is None
    # 4) a variant that differs is switched off for the layer and the old path serves it
    agrees.clear()
    world, box = World(), {}
    groups = [FakeGroup(world, box) for _ in range(W)]
    layers = [L() for _ in range(W)]
    orig = rs.assemble_stacked
    rs.assemble_stacked = lambda stacked, ranges, offs: orig(stacked, ranges, offs).flip(0)
    outs = run_all(groups, layers)
    rs.assemble_stacked = orig
    assert all(ly._dsv41_cg == {("minimal", 6): False} for ly in layers)
    assert all(torch.equal(o.view(torch.int16), bits) for o in outs)     # served by the old path
    # 5) never checked during capture: old path, state stays unchecked
    agrees.clear()
    class CapturingTorch:                    # gather() sees "capturing" (and a CUDA shard) through this
        cuda = type("C", (), {"is_current_stream_capturing": staticmethod(lambda: True)})

        def __getattr__(self, k):
            return getattr(torch, k)
    saved_torch = rs.torch
    rs.torch = CapturingTorch()
    world, box = World(), {}
    groups = [FakeGroup(world, box) for _ in range(W)]
    layers = [L() for _ in range(W)]

    class CudaLike(torch.Tensor):
        is_cuda = True
    saved_locs = list(locs)
    locs[:] = [x.as_subclass(CudaLike) for x in locs]
    outs = run_all(groups, layers)
    locs[:] = saved_locs
    rs.torch = saved_torch
    assert agrees == [] and all(ly._dsv41_cg == {} for ly in layers)
    assert all(torch.equal(o.view(torch.int16), bits) for o in outs)
    # 6) gate off: exactly the old path
    rs.COMPACT = False
    world, box = World(), {}
    groups = [FakeGroup(world, box) for _ in range(W)]
    layers = [L() for _ in range(W)]
    outs = run_all(groups, layers)
    assert all(not hasattr(ly, "_dsv41_cg") for ly in layers)
    assert all(torch.equal(o.view(torch.int16), bits) for o in outs)
    print("  gather(): gate off = old path; self-check once per variant/layer; RoCE only with pynccl on; "
          "a differing variant falls back; no check under capture", flush=True)
    print("test_compact_gather: ok", flush=True)


if __name__ == "__main__":
    main()
