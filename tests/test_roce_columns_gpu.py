"""One-GPU check of the RoCE columns all-gather kernel (b12x .../roce/_allgather_cute.py,
_RoceAllGatherColumnsLaunch, added by scripts/b12x-roce-columns-gather.patch) with the peers faked in
device memory, plus a CUDA-graph microbench.

  docker run --rm --gpus all --network none -v <ds41>:/ds41:ro \
      -e PYTHONPATH=/ds41/runtime/fetched/b12x:/ds41/adapter:/ds41/tests -e B12X_COMPILE_CACHE_DIR=/tmp/b12x-cc \
      --entrypoint python3 <image> /ds41/tests/test_roce_columns_gpu.py [--bench]

No NIC, no proxy, no pinned memory: the kernel only sees addresses. A fake region per rank holds the
receive slots (each peer's payload written there by the test, as the NIC would), the stripe flags
(set to the sequence number, as the NIC's flag write would) and the control record. For each of the
4 rank specializations of wqkv_a's layout (own widths 512/512/384/384 bf16 = 64/64/48/48 packs at
0/64/128/176, row stride 224 packs) and M in 1..96:
  - window mode (every rank's GEMM output is [M, 512], rank 3's own columns start at 128) and exact
    slice mode ([M, 384] on ranks 2/3), random bit patterns (NaN payloads included);
  - the output equals the stock [M, 1792] tensor byte for byte, the send slot holds exactly the compact
    own shard (M * packs[rank] packs, nothing more), the doorbell carries that byte count, and the
    epoch advances with no poison; two consecutive launches exercise both transport slots;
  - the plain (unchanged) gather kernel on the same fake transport still produces the padded dim-0
    layout, and the old reorder of it (movedim copy + cat) equals the columns kernel's output.
--bench: per rank 0 and 3, M = 6 and 96, CUDA graphs of 40 calls: old (plain gather + movedim copy +
cat), minimal (plain gather + one cat of views), compact (columns kernel), each preceded by the same
epoch reset so the faked flags stay valid; reports us per call and the deltas.
"""
import statistics
import sys

import torch

import replicated_split as rs
from b12x.comm.roce import _allgather_cute as ag
from b12x.comm.roce.roce_oneshot import PACK_BYTES, _grid_blocks

torch.cuda.set_per_process_memory_fraction(min(1.0, 1.5e9 / torch.cuda.get_device_properties(0).total_memory))
DEV = torch.device("cuda", 0)
W, SLOTS, HCA, FS, THREADS, BLOCKS = 4, 2, 2, 64, 512, 8
SB = 256 * 1024
CLASSES = BLOCKS.bit_length()
SPIN = 1 << 20
N, WIDTH = 1792, 512


class FakeRank:
    """One rank's transport memory: receive slots, stripe flags, send slots, control record, counters."""

    def __init__(self):
        self.recv = torch.zeros(W * SLOTS * SB, dtype=torch.uint8, device=DEV)
        self.flags = torch.zeros(W * SLOTS * HCA * FS // 4, dtype=torch.int32, device=DEV)
        self.send = torch.zeros(SLOTS * SB, dtype=torch.uint8, device=DEV)
        self.ctrl = torch.zeros(16, dtype=torch.int32, device=DEV)
        self.counters = torch.zeros(2 + 2 * CLASSES, dtype=torch.int32, device=DEV)

    def addrs(self, grid):
        e = self.counters.data_ptr()
        cls = grid.bit_length() - 1
        return dict(recv=self.recv.data_ptr(), flag=self.flags.data_ptr(), send=self.send.data_ptr(),
                    ctrl=self.ctrl.data_ptr(), epoch=e, stage=e + 4 * (1 + cls), tail=e + 4 * (1 + CLASSES + cls),
                    poison=e + 4 * (1 + 2 * CLASSES))

    def post(self, rank, seq, payloads):
        """Peers' payloads land in slot seq & 1 and their stripe flags read seq (what the NIC does)."""
        slot = seq & 1
        for s in range(W):
            if s == rank:
                continue
            b = payloads[s].contiguous().view(torch.uint8).view(-1)
            off = (s * SLOTS + slot) * SB
            self.recv[off:off + b.numel()].copy_(b)
            for h in range(HCA):
                self.flags[((s * SLOTS + slot) * HCA + h) * (FS // 4)] = seq
        self.counters[0] = seq - 1

    def slot_bytes(self, slot):
        return self.send[slot * SB:(slot + 1) * SB]


def launch_columns(fr, rank, local, off, packs, out, rows):
    launcher = ag.get_columns_launcher(W, rank, THREADS, SLOTS, FS, HCA, 0, tuple(packs))
    grid = _grid_blocks(rows * max(packs), THREADS, BLOCKS)
    a = fr.addrs(grid)
    launcher(local.data_ptr(), out.data_ptr(), rows, local.stride(0) * 2 // PACK_BYTES, off * 2 // PACK_BYTES,
             rows * packs[rank] * PACK_BYTES, a["recv"], a["flag"], a["send"], a["ctrl"], SB, a["epoch"],
             a["stage"], a["tail"], a["poison"], SPIN, grid)


def launch_plain(fr, rank, shard, out):
    """The unchanged kernel as roce_gather routes the TP gather: dim 0, row_packs = the shard."""
    launcher = ag.get_launcher(W, rank, THREADS, SLOTS, FS, HCA, 0)
    nbytes = shard.numel() * 2
    grid = _grid_blocks(nbytes // PACK_BYTES, THREADS, BLOCKS)
    a = fr.addrs(grid)
    launcher(shard.data_ptr(), out.data_ptr(), nbytes // PACK_BYTES, nbytes, nbytes // PACK_BYTES, a["recv"],
             a["flag"], a["send"], a["ctrl"], SB, a["epoch"], a["stage"], a["tail"], a["poison"], SPIN, grid)


def layout():
    ranges, width = rs._ranges(N, W)
    wins = [rs.window(N, a, width) for a, _ in ranges]
    woffs = [a - w for (a, _), w in zip(ranges, wins)]
    widths, _ = rs.compact_layout(ranges)
    packs, dst, total = rs.pack_layout(ranges)
    return ranges, width, wins, woffs, widths, packs


def old_reorder(g, m, ranges, offs):
    """GroupCoordinator.all_gather(dim=-1)'s reshape/movedim/reshape (a copy) + the adapter's cat."""
    t = g.reshape(W, m, WIDTH).movedim(0, 1).reshape(m, W * WIDTH)
    return rs.assemble(t, ranges, WIDTH, offs)


def check():
    ranges, width, wins, woffs, widths, packs = layout()
    assert packs == [64, 64, 48, 48]
    g = torch.Generator(device=DEV).manual_seed(11)
    n_ok = 0
    for m in (1, 2, 3, 5, 6, 7, 12, 16, 30, 48, 64, 96):
        for mode in ("window", "slice"):
            bits = torch.randint(-32768, 32767, (m, N), generator=g, device=DEV, dtype=torch.int16)
            full = bits.view(torch.bfloat16)
            if mode == "window":
                locs, offs = [full[:, w:w + width].contiguous() for w in wins], woffs
            else:
                locs, offs = [full[:, a:b].contiguous() for a, b in ranges], [0] * W
            compact = [full[:, a:b].contiguous() for a, b in ranges]            # what each rank sends
            for rank in range(W):
                fr = FakeRank()
                for seq in (1, 2):                                               # both transport slots
                    fr.post(rank, seq, compact)
                    out = torch.full((m, N), -1, dtype=torch.int16, device=DEV).view(torch.bfloat16)
                    launch_columns(fr, rank, locs[rank], offs[rank], packs, out, m)
                    torch.cuda.synchronize()
                    c = fr.counters.tolist()
                    assert c[0] == seq and c[1 + 2 * CLASSES] == 0, (m, mode, rank, seq, c)
                    assert torch.equal(out.view(torch.int16), bits), (m, mode, rank, seq, "output")
                    nb = m * packs[rank] * PACK_BYTES
                    assert fr.ctrl[1].item() == nb and fr.ctrl[4 + (seq & 1)].item() == nb and fr.ctrl[0].item() == seq
                    sent = fr.slot_bytes(seq & 1)
                    assert torch.equal(sent[:nb], compact[rank].view(torch.uint8).view(-1)), (m, mode, rank, "send")
                    assert int(sent[nb:nb + 4096].count_nonzero()) == 0          # nothing past the shard (first use of the slot)
                    n_ok += 1
                # the plain kernel on the same fake transport (padded 512-wide shards), then the old reorder
                fr = FakeRank()
                padded = [rs.padded(x, width) for x in locs]
                fr.post(rank, 1, padded)
                gout = torch.empty(W * m, width, dtype=torch.bfloat16, device=DEV)
                launch_plain(fr, rank, padded[rank], gout)
                torch.cuda.synchronize()
                assert torch.equal(gout.view(W, m, width).view(torch.int16), torch.stack(padded).view(torch.int16))
                assert rs.same_bits(old_reorder(gout, m, ranges, offs), full), (m, mode, rank, "old")
                ag2 = rs.assemble_stacked(gout.view(W, m, width), ranges, offs)
                assert rs.same_bits(ag2, full), (m, mode, rank, "minimal")
    print(f"  columns kernel: {n_ok} launches (4 ranks x 12 M x window/slice x 2 slots) byte-identical to the "
          f"stock layout; send slot = compact shard only; plain kernel + old reorder / minimal cat identical", flush=True)


def bench(reps=50, rounds=7, calls=40):
    ranges, width, wins, woffs, widths, packs = layout()
    g = torch.Generator(device=DEV).manual_seed(5)
    res = {}
    for rank in (0, 3):
        for m in (6, 96):
            bits = torch.randint(-32768, 32767, (m, N), generator=g, device=DEV, dtype=torch.int16)
            full = bits.view(torch.bfloat16)
            locs = [full[:, w:w + width].contiguous() for w in wins]
            compact = [full[:, a:b].contiguous() for a, b in ranges]
            fa, fc = FakeRank(), FakeRank()
            fa.post(rank, 1, locs)            # the plain path sends the whole 512-wide window part
            fc.post(rank, 1, compact)
            outs = {}

            def arm_old():
                fa.counters[0].zero_()
                gout = torch.empty(W * m, width, dtype=torch.bfloat16, device=DEV)
                launch_plain(fa, rank, locs[rank], gout)
                outs["old"] = old_reorder(gout, m, ranges, woffs)

            def arm_min():
                fa.counters[0].zero_()
                gout = torch.empty(W * m, width, dtype=torch.bfloat16, device=DEV)
                launch_plain(fa, rank, locs[rank], gout)
                outs["minimal"] = rs.assemble_stacked(gout.view(W, m, width), ranges, woffs)

            def arm_compact():
                fc.counters[0].zero_()
                out = torch.empty(m, N, dtype=torch.bfloat16, device=DEV)
                launch_columns(fc, rank, locs[rank], woffs[rank], packs, out, m)
                outs["compact"] = out

            def arm_reset():
                fc.counters[0].zero_()

            arms = {"old": arm_old, "minimal": arm_min, "compact": arm_compact, "reset": arm_reset}
            graphs = {}
            for name, fn in arms.items():
                s = torch.cuda.Stream()
                s.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(s):
                    fn()
                torch.cuda.current_stream().wait_stream(s)
                torch.cuda.synchronize()
                gr = torch.cuda.CUDAGraph()
                with torch.cuda.graph(gr):
                    for _ in range(calls):
                        fn()
                graphs[name] = gr
            for gr in graphs.values():
                for _ in range(3):
                    gr.replay()
            torch.cuda.synchronize()
            for name in ("old", "minimal", "compact"):
                assert rs.same_bits(outs[name], full), (rank, m, name, "graph replay")
            assert fa.counters[1 + 2 * CLASSES].item() == 0 and fc.counters[1 + 2 * CLASSES].item() == 0
            t = {k: [] for k in arms}
            for _ in range(rounds):
                for name, gr in graphs.items():
                    a = torch.cuda.Event(enable_timing=True)
                    b = torch.cuda.Event(enable_timing=True)
                    a.record()
                    for _ in range(reps):
                        gr.replay()
                    b.record()
                    b.synchronize()
                    t[name].append(a.elapsed_time(b) * 1000.0 / (reps * calls))
            med = {k: statistics.median(v) for k, v in t.items()}
            res[(rank, m)] = med
            print(f"  rank {rank} M={m:3d}: us/call old {med['old']:.2f}  minimal {med['minimal']:.2f}  "
                  f"compact {med['compact']:.2f}  (epoch reset alone {med['reset']:.2f}; all include it)  "
                  f"old-compact {med['old'] - med['compact']:.2f}  old-minimal {med['old'] - med['minimal']:.2f}",
                  flush=True)
            del graphs
            torch.cuda.synchronize()
    return res


if __name__ == "__main__":
    check()
    if "--bench" in sys.argv:
        bench()
    print("test_roce_columns_gpu: ok", flush=True)
