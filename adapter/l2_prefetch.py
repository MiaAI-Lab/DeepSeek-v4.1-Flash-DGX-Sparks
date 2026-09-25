"""Prefetch the next dense weights into L2 while DRAM is idle in a decode step (decode only).

Every decode step spends ~3.6 ms (c1, rank 1-3 profile 2026-09-24) inside ~89 one-shot
all-reduces and ~49 all-gathers during which no other kernel runs: the RoCE kernel is a 1-8 CTA
grid polling flags, so the SMs and DRAM are idle. The kernel that follows each window is a dense
MXFP8 GEMM (post-MoE AR -> next layer's wqkv_a; all-gather -> wq_b) or the router + shared expert
(post-attention AR), all weight-bound at M<=16. This adapter issues
``cp.async.bulk.prefetch.L2`` for the first DSV41_L2_PREFETCH_MB of those weights right before
the collective is launched, so the GEMM's first K tiles come from L2. The prefetch kernel runs on
a side stream forked right before the collective (issuing 8 MB of bulk prefetches takes ~20 us
because the TMA queue drains at DRAM speed, so it must not sit on the main stream) and is joined
at the next collective or at the end of the forward, long after it finished. Weights are static and a
prefetch is only a cache hint: outputs are bit-identical by construction.

Which weights follow which window is learned, not hard-coded: during the eager warm-up
forwards that SGLang runs before capturing each CUDA graph, the collectives are numbered per
forward and every MXFP8 dense linear records itself in the event sequence. The capture pass
(the same Python sequence) then forks the prefetch right before the collective, so it becomes a
graph branch. A forward whose signature was never learned prefetches nothing; weights are static,
so a stale plan can only waste bandwidth, never change a result.

Gate ``DSV41_L2_PREFETCH=1`` (off by default). ``DSV41_L2_PREFETCH_MB`` budget per window
(default 6), ``DSV41_L2_PREFETCH_AG=0`` skips all-gather windows.

Production: DSV41_L2_PREFETCH=1 DSV41_L2_PREFETCH_WOA=1 (fleet A/B 2026-09-25: WOA -0.4 ms/step,
greedy output byte-identical). Every other sub-gate below (AHEAD, DRAFT, ENGRAM, LMHEAD, SKIP_N,
WOB_MB, MOE) was measured on the fleet the same day, came out flat or slower, and is NOT
recommended; they stay in the code, off by default (0 / unset), for in-boot A/B runs only.

v3 sub-gates (each needs DSV41_L2_PREFETCH=1; all off = the v2 plan, bit for bit):
  DSV41_L2_PREFETCH_WOA=1     wo_a (the fp8 twin of adapter/wo_a_w8.py, or the bf16 weight) is
                              recorded, and the MXFP8 linear right before it (wq_b, or the
                              indexer's last linear in compressed layers) becomes a window: after
                              that linear, a side branch prefetches DSV41_L2_PREFETCH_WOA_MB (6)
                              of wo_a while rope / paging / sparse MLA run (~22-40 us, latency
                              bound). This branch is joined at the end of the forward.
  DSV41_L2_PREFETCH_AHEAD=1   a window whose own weights are smaller than its budget continues
                              into the next window's weights (post-MoE AR: wqkv_a slice 2.6 MB,
                              then the first MB of wq_b); the next window resumes where it stopped.
  DSV41_L2_PREFETCH_DRAFT=1   the DSpark draft forward is bracketed too (its own plans).
  DSV41_L2_PREFETCH_ENGRAM=1  the Engram row join (host-callback wait, 0.2-0.7 ms idle per step)
                              becomes a window of DSV41_L2_PREFETCH_ENGRAM_MB (12) that always
                              looks ahead into the engram.wkv slice read after the row all-reduce.
                              Joined at the end of the forward.
  DSV41_L2_PREFETCH_LMHEAD=1  the target LM head weight is recorded after the last collective.
v4 knobs (read per call through ab_variant.env(); all unset = v3):
  DSV41_L2_PREFETCH_SKIP_N=512  MXFP8 weights with N <= this many output rows leave the plan (the
                              wqkv_a slice, a latency-bound GEMM); a window left empty reaches into
                              the next window (post-MoE AR -> wq_b).
  DSV41_L2_PREFETCH_WOB_MB=4  the attention-core window takes WOA_MB of wo_a, then this much of wo_b.
  DSV41_L2_PREFETCH_MOE=1     (measured NO-GO, off) after the router GEMM, a side branch prefetches
                              DSV41_L2_PREFETCH_MOE_MB (10) of the next layer's dense weights with an
                              L2::evict_last policy while the MoE streams, demoting the previous
                              window's lines first; later windows start after those bytes. Against
                              the real b12x MoE it slows the MoE by ~12.6 us per 10 MB and the lines
                              do not outlive the next layer's dense stream: -13 us per layer.
  (v3's TILED experiment, scales + a K prefix of every row, lost on GB10 and was removed.)

Wiring (all gated by the env flag in sitecustomize.py):
  sglang.srt.layers.quantization.fp8_utils  -> l2_prefetch.install_fp8_utils(module)  (after mxfp8_b12x)
  b12x.comm.roce(_ring).roce_oneshot        -> l2_prefetch.install_roce(module)
  sglang.srt.models.deepseek_v4             -> l2_prefetch.install_model(module)   (after wo_a_w8)
  sglang.srt.models.deepseek_v2             -> l2_prefetch.install_router(module)
  sglang.srt.models.deepseek_v4_dspark      -> l2_prefetch.install_draft(module)   (_DRAFT)
  sglang.srt.layers.engram                  -> l2_prefetch.install_engram(module)  (_ENGRAM, after engram_prefetch)
Microbench and go/no-go: sparks/diagnostics/dsv41-l2-pdl/RESULTS.md, sparks/diagnostics/dsv41-l2-v3/RESULTS.md.
"""
import ctypes as C
import logging
import os
import subprocess
import sys
from pathlib import Path

import torch

try:  # DSV41_AB_VARIANTS in-boot A/B (test only): per-variant gates; plain os.environ when off
    from ab_variant import env as _env, sig_tag as _ab_tag
except ImportError:
    _env, _ab_tag = os.environ.get, tuple

logger = logging.getLogger(__name__)

_SRC = r"""
#include <cuda_runtime.h>
#include <stdint.h>
// segs: [n][2] int64 {address, bytes}; one bulk prefetch per <=chunk piece.
__global__ void l2pf(const long long* __restrict__ segs, int n, long long chunk) {
  for (int s = blockIdx.x; s < n; s += gridDim.x) {
    long long a = segs[2 * s], b = segs[2 * s + 1];
    for (long long off = threadIdx.x * chunk; off < b; off += (long long)blockDim.x * chunk) {
      long long sz = b - off < chunk ? b - off : chunk;
      sz &= ~15LL;
      if (sz > 0)
        asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" :: "l"(a + off), "r"((unsigned)sz) : "memory");
    }
  }
}
extern "C" int dsv41_l2pf(const void* segs, int n, long long chunk, uintptr_t stream) {
  l2pf<<<n < 8 ? n : 8, 64, 0, (cudaStream_t)stream>>>((const long long*)segs, n, chunk);
  return (int)cudaGetLastError();
}
// v4 MOE: the same bulk prefetch with an L2::evict_last cache policy, so lines prefetched during
// the MoE stream survive it; first, the previous MoE window's lines go back to evict_normal
// (applypriority), so at most one window's worth of evict_last lines is ever resident.
__global__ void l2pf_el(const long long* __restrict__ segs, int n, const long long* __restrict__ old, int n_old,
                        long long chunk) {
  long long tid = (long long)blockIdx.x * blockDim.x + threadIdx.x, nth = (long long)gridDim.x * blockDim.x;
  for (int s = 0; s < n_old; ++s)
    for (long long off = tid * 128; off < old[2 * s + 1]; off += nth * 128)
      asm volatile("applypriority.global.L2::evict_normal [%0], 128;" :: "l"(old[2 * s] + off) : "memory");
  uint64_t pol;
  asm volatile("createpolicy.fractional.L2::evict_last.b64 %0, 1.0;" : "=l"(pol));
  for (int s = blockIdx.x; s < n; s += gridDim.x) {
    long long a = segs[2 * s], b = segs[2 * s + 1];
    for (long long off = threadIdx.x * chunk; off < b; off += (long long)blockDim.x * chunk) {
      long long sz = b - off < chunk ? b - off : chunk;
      sz &= ~15LL;
      if (sz > 0)
        asm volatile("cp.async.bulk.prefetch.L2.global.L2::cache_hint [%0], %1, %2;"
                     :: "l"(a + off), "r"((unsigned)sz), "l"(pol) : "memory");
    }
  }
}
extern "C" int dsv41_l2pf_el(const void* segs, int n, const void* old, int n_old, long long chunk, uintptr_t stream) {
  l2pf_el<<<8, 64, 0, (cudaStream_t)stream>>>((const long long*)segs, n, (const long long*)old, n_old, chunk);
  return (int)cudaGetLastError();
}
"""

_state = {"lib": None, "learning": {}, "plans": {}, "sig": None, "idx": 0, "lin": 0, "eng": 0, "moe": 0,
          "side": {}, "side2": {}, "pending": None, "pending2": None, "stats": [0, 0],
          "installed": set(), "logged": set(), "frozen": set(), "keepalive": [], "lm_head": {},
          "woa_depth": 0}
_CHUNK = 16384
# RoCEnante packages whose collectives fork the prefetch: the stock one and the switchless-ring
# one (DSV41_ROCE_RING=1, runtime/b12x/b12x/comm/roce_ring); runtime/roce_tp4_adapt.py picks one.
_ROCE_MODULES = ("b12x.comm.roce.roce_oneshot", "b12x.comm.roce_ring.roce_oneshot")


def enabled():
    return _env("DSV41_L2_PREFETCH", "0").strip() in ("1", "on", "true")


def _sub(name):
    """v3 sub-gate DSV41_L2_PREFETCH_<name> (off by default, needs the main gate)."""
    return enabled() and _env(f"DSV41_L2_PREFETCH_{name}", "0").strip() in ("1", "on", "true")


def _installed(name=None):
    """Install-time gate: always os.environ. Under DSV41_AB_VARIANTS it holds the union of every
    variant (adapter/ab_variant.py), so code a later variant needs is installed; the per-call checks
    (enabled() / _sub()) then pick the variant being captured or replayed."""
    on = os.environ.get("DSV41_L2_PREFETCH", "0").strip() in ("1", "on", "true")
    if name is None or not on:
        return on
    return os.environ.get(f"DSV41_L2_PREFETCH_{name}", "0").strip() in ("1", "on", "true")


def _mb(name, default):
    return int(float(_env(name, default)) * (1 << 20))


def _budget():
    return _mb("DSV41_L2_PREFETCH_MB", "6")


def _lib():
    if _state["lib"] is None:
        cache = Path(os.environ.get("B12X_COMPILE_CACHE_DIR", "/tmp")) / "dsv41_l2pf"
        cache.mkdir(parents=True, exist_ok=True)
        so = cache / "libdsv41_l2pf_v4.so"  # older cached builds lack l2pf_el
        if not so.exists():
            cu = cache / "l2pf.cu"
            cu.write_text(_SRC)
            tmp = cache / f"libdsv41_l2pf_v4.{os.getpid()}.so"
            subprocess.run(["nvcc", "-O3", "-arch=sm_121a", "-shared", "-Xcompiler", "-fPIC",
                            "-o", str(tmp), str(cu)], check=True)
            os.replace(tmp, so)
        lib = C.CDLL(str(so))
        lib.dsv41_l2pf.argtypes = [C.c_void_p, C.c_int, C.c_longlong, C.c_void_p]
        lib.dsv41_l2pf_el.argtypes = [C.c_void_p, C.c_int, C.c_void_p, C.c_int, C.c_longlong, C.c_void_p]
        _state["lib"] = lib
    return _state["lib"]


# -- forward bracketing -------------------------------------------------------------------------

def _begin_forward(sig):
    """Every eager decode/verify forward of a shape that has not been captured yet re-learns its
    plan. The first eager forward of a shape can be a dummy/profiling run outside the graph-capture
    communicator state, whose all-reduces go through NCCL and never reach RoCE (seen 2026-09-25:
    the bs16 verify shape learned 0 collectives and ran uncaptured-plan). The two warm-up forwards
    SGLang runs right before each capture are what the capture replays, so the latest eager
    observation wins, and a shape is frozen once captured."""
    _state["sig"], _state["idx"], _state["lin"], _state["eng"], _state["moe"] = sig, 0, 0, 0, 0
    _state["stats"][1] = 0
    if _sub("WOA"):
        _ensure_wo_a()
    if sig not in _state["frozen"] and not torch.cuda.is_current_stream_capturing():
        _state["learning"][sig] = []


def _join():
    pending = _state["pending"]
    if pending is not None:
        torch.cuda.current_stream().wait_stream(pending)
        _state["pending"] = None


def _join2():
    pending = _state["pending2"]
    if pending is not None:
        torch.cuda.current_stream().wait_stream(pending)
        _state["pending2"] = None


def _side_stream(second=False):
    pool = _state["side2" if second else "side"]
    dev = torch.cuda.current_device()
    if dev not in pool:
        pool[dev] = torch.cuda.Stream(device=dev)
    return pool[dev]


def _counts():
    return (_state["idx"], _state["lin"], _state["eng"], _state["moe"])


def _end_forward():
    _join()
    _join2()
    sig = _state["sig"]
    capturing = torch.cuda.is_current_stream_capturing()
    rec = _state["learning"].pop(sig, None) if sig is not None else None
    if rec is not None and not capturing:
        n = _state["idx"]
        counts = _counts()
        key = (counts, tuple(rec))
        old = _state["plans"].get(sig)
        if n == 0:
            # no RoCE collective seen (NCCL path): keep any earlier plan, never learn "nothing"
            if old is None and ("zero",) + sig[1:] not in _state["logged"]:
                _state["logged"].add(("zero",) + sig[1:])
                logger.warning("DSV41_L2_PREFETCH eager forward tokens=%d mode=%s saw no RoCE collective "
                               "(NCCL path); not learned, waiting for the capture warm-up", sig[1], sig[2])
        elif old is None or old[2] != key:
            plan = _make_plan(rec)
            _state["plans"][sig] = (n, plan, key, counts)
            _side_stream()  # created eagerly, never first inside a graph capture
            _side_stream(True)
            nbytes = sum(int(it[0][1::2].sum()) for it in plan.values())
            extra = sum(1 for k in plan if k[0] != "c")
            logger.warning("DSV41_L2_PREFETCH plan tokens=%d mode=%s collectives=%d windows=%d "
                           "prefetch=%.1f MB/forward%s%s", sig[1], sig[2], n, len(plan), nbytes / 2**20,
                           f" (of which {extra} linear/engram windows)" if extra else "",
                           "" if old is None else f" (relearned; was {old[0]} collectives)")
    elif sig is not None and capturing:
        _state["frozen"].add(sig)
        key = ("capture",) + sig[1:]
        if key not in _state["logged"]:
            _state["logged"].add(key)
            entry = _state["plans"].get(sig)
            learned = entry[0] if entry else 0
            ok = (entry is not None and learned == _state["idx"] and entry[3] == _counts()
                  and _state["stats"][1] > 0)
            logger.warning("DSV41_L2_PREFETCH captured tokens=%d mode=%s branches=%d collectives=%d "
                           "(learned %d)%s", sig[1], sig[2], _state["stats"][1], _state["idx"], learned,
                           "" if ok else " MISMATCH: this graph runs without prefetch")
    _state["sig"] = None


def _take(q, start, budget):
    """Bytes of queue q ([(ptr, nbytes)] in consumption order) from position start=(i, off), at most
    budget -> (segments, end position). From (0, 0) this is the v2 rule: whole weights in order,
    the last one cut at the budget, every piece rounded down to 16 bytes."""
    out, left = [], budget
    i, off = start
    while i < len(q) and left >= 16:
        ptr, nbytes = q[i][:2]
        avail = nbytes - off
        take = min(avail, left) & ~15
        if take > 0:
            out.append((ptr + off, take))
        if avail > left:
            return out, (i, off + take)
        left -= take
        i, off = i + 1, 0
    return out, (i, off)


def _skip_n():
    return int(float(_env("DSV41_L2_PREFETCH_SKIP_N", "0") or 0))


def _windows(events):
    """Event sequence -> [(key, budget, queue, look_ahead)] in forward order. Queue items are
    (ptr, nbytes, rows or None, kind) with kind "r" (linear / router) or "w" (wo_a)."""
    woa, ahead = _sub("WOA"), _sub("AHEAD")
    windows, cur, last_lin = [], None, None
    for ev in events:
        kind = ev[0]
        if kind == "m":  # v4 MOE mark: plans against the windows that follow (see _make_plan)
            windows.append([("m", ev[1]), _mb("DSV41_L2_PREFETCH_MOE_MB", "10"), [], False])
            continue
        if kind == "c":
            cur = [("c", ev[1]), _budget(), [], ahead]
            windows.append(cur)
            last_lin = None
        elif kind == "e":
            cur = [("e", ev[1]), _mb("DSV41_L2_PREFETCH_ENGRAM_MB", "12"), [], True]
            windows.append(cur)
            last_lin = None
        elif kind == "L":
            if cur is not None:
                last_lin = (cur, len(cur[2]), ev[1])
        elif kind == "w":
            if woa and last_lin is not None and last_lin[0] is cur:
                owner, pos, lin = last_lin
                tail = owner[2][pos:]
                del owner[2][pos:]
                cur = [("l", lin), _mb("DSV41_L2_PREFETCH_WOA_MB", "6"), tail, ahead]
                windows.append(cur)
                last_lin = None
            if cur is not None:
                cur[2].append((ev[1], ev[2], None, "w"))
        elif cur is not None:  # "r": a weight read, MXFP8 linears with their output rows N
            cur[2].append((ev[1], ev[2], ev[3] if len(ev) > 3 else None, "r"))
    skip = _skip_n()
    if skip > 0:
        # v4 SKIP_N: MXFP8 weights with N <= skip (the wqkv_a slice: a latency-bound GEMM that
        # prefetching does not speed up) leave the plan; a window left with nothing of its own
        # reaches into the next window's weights (post-MoE AR -> wq_b)
        for w in windows:
            kept = [it for it in w[2] if it[2] is None or it[2] > skip]
            if w[2] and not kept:
                w[3] = True
            w[2] = kept
    return windows


def _make_plan(events):
    """events: the forward's sequence of collectives / windows / weight reads -> {window key:
    device segment table}. Without the v3 sub-gates the windows are the collectives and each takes
    the first budget bytes of the weights read before the next collective (the v2 plan)."""
    windows = _windows(events)
    plan = {}
    start = (0, 0)
    wob = _mb("DSV41_L2_PREFETCH_WOB_MB", "0")
    cover = {}            # window index -> queue position already held (evict_last) by a MOE window
    moe_items = []
    for j, (key, budget, queue, look_ahead) in enumerate(windows):
        if key[0] == "m":
            # v4 MOE: during the MoE phase (DRAM at ~157 of ~250 GB/s) prefetch the next layer's dense
            # weights with evict_last, up to the next MOE mark; later windows start past them
            segs, left = [], budget
            for k in range(j + 1, len(windows)):
                if windows[k][0][0] == "m" or left < 16:
                    break
                more, end = _take(windows[k][2], (0, 0), left)
                segs += more
                left -= sum(x[1] for x in more)
                cover[k] = end
            if segs:
                t = torch.tensor([v for x in segs for v in x[:2]], dtype=torch.int64, device="cuda")
                _state["keepalive"].append(t)
                moe_items.append((key, t, len(segs)))
            continue
        if cover.get(j, (0, 0)) > start:
            start = cover[j]
        if key[0] == "l" and wob > 0:
            # v4 WOB: the attention-core window takes WOA_MB of wo_a, then WOB_MB of wo_b (the
            # weights after wo_a, up to the post-attention AR) instead of the rest of wo_a
            n_w = next((i for i, it in enumerate(queue) if it[3] != "w"), len(queue))
            segs, _ = _take(queue[:n_w], (0, 0), budget)
            more, _ = _take(queue[n_w:], (0, 0), wob)
            segs += more
            start = (0, 0)
            if segs:
                t = torch.tensor([v for s in segs for v in s[:2]], dtype=torch.int64, device="cuda")
                plan[key] = (t, len(segs))
                _state["keepalive"].append(t)
            continue
        segs, end = _take(queue, start, budget)
        start = (0, 0)
        used = sum(s[1] for s in segs)
        if look_ahead and j + 1 < len(windows) and end[0] >= len(queue) and budget - used >= 16:
            more, start = _take(windows[j + 1][2], cover.get(j + 1, (0, 0)), budget - used)
            segs += more
        if segs:
            t = torch.tensor([v for s in segs for v in s], dtype=torch.int64, device="cuda")
            plan[key] = (t, len(segs))
            _state["keepalive"].append(t)  # a captured graph keeps reading this address
    for i, (key, t, n) in enumerate(moe_items):
        _, pt, pn = moe_items[i - 1]  # demote the previous MOE window's lines (wraps to the last)
        plan[key] = (t, n, "el", pt if len(moe_items) > 1 else t, pn if len(moe_items) > 1 else 0)
    return plan


# -- hooks --------------------------------------------------------------------------------------

def _rec(ev):
    sig = _state["sig"]
    if sig is not None and sig in _state["learning"]:
        _state["learning"][sig].append(ev)


def _record_weight(*tensors):
    sig = _state["sig"]
    if sig is None or sig not in _state["learning"]:
        return
    rec = _state["learning"][sig]
    for t in tensors:
        if t is not None and t.is_cuda:
            rec.append(("r", t.data_ptr(), t.numel() * t.element_size()))


def _record_linear(weight, weight_scale):
    """MXFP8 linear: weight and scales, tagged with the weight's output rows N (for SKIP_N)."""
    sig = _state["sig"]
    if sig is None or sig not in _state["learning"]:
        return
    rec = _state["learning"][sig]
    rows = int(weight.shape[0]) if getattr(weight, "dim", lambda: 0)() == 2 else None
    for t in (weight, weight_scale):
        if t is not None and t.is_cuda:
            rec.append(("r", t.data_ptr(), t.numel() * t.element_size(), rows))


def _fork(item, stream=None, second=False):
    t, n = item[:2]
    main = stream if isinstance(stream, torch.cuda.Stream) else torch.cuda.current_stream()
    side = _side_stream(second)
    side.wait_stream(main)
    if len(item) > 2:  # MOE: (segments, n, "el", previous MOE segments, their count)
        rc = _lib().dsv41_l2pf_el(t.data_ptr(), n, item[3].data_ptr(), item[4], _CHUNK, side.cuda_stream)
    else:
        rc = _lib().dsv41_l2pf(t.data_ptr(), n, _CHUNK, side.cuda_stream)
    if rc:
        raise RuntimeError(f"DSV41_L2_PREFETCH launch failed: cuda error {rc}")
    _state["pending2" if second else "pending"] = side
    _state["stats"][0] += 1
    _state["stats"][1] += 1


def _before_collective(kind, stream):
    sig = _state["sig"]
    if sig is None:
        return
    _join()
    _state["idx"] += 1
    idx = _state["idx"]
    _rec(("c", idx))
    if kind == "ag" and _env("DSV41_L2_PREFETCH_AG", "1") == "0":
        return
    entry = _state["plans"].get(sig)
    if entry is None:
        return
    if idx > entry[0]:
        return  # more collectives than learned: sequence differs, launch nothing past the plan
    item = entry[1].get(("c", idx))
    if item is not None:
        _fork(item, stream)


def _after_linear():
    """WOA: numbered MXFP8 linear boundary; forks the wo_a window planned after this linear."""
    sig = _state["sig"]
    if sig is None:
        return
    _state["lin"] += 1
    lin = _state["lin"]
    _rec(("L", lin))
    entry = _state["plans"].get(sig)
    if entry is None or lin > entry[3][1] or _state["idx"] > entry[0]:
        return
    item = entry[1].get(("l", lin))
    if item is not None:
        _fork(item, second=True)


def _after_router():
    """v4 MOE: the MoE phase starts after the router GEMM; fork the evict_last prefetch there."""
    _state["moe"] += 1
    n = _state["moe"]
    _rec(("m", n))
    entry = _state["plans"].get(_state["sig"])
    if entry is None or n > entry[3][3] or _state["idx"] > entry[0]:
        return
    item = entry[1].get(("m", n))
    if item is not None:
        _fork(item, second=True)


def _before_engram():
    sig = _state["sig"]
    if sig is None or not _sub("ENGRAM"):
        return
    _state["eng"] += 1
    n = _state["eng"]
    _rec(("e", n))
    entry = _state["plans"].get(sig)
    if entry is None or n > entry[3][2] or _state["idx"] > entry[0]:
        return
    item = entry[1].get(("e", n))
    if item is not None:
        _fork(item, second=True)


def _record_wo_a(o, wo_a):
    """The bytes the wo_a dispatch will read: the fp8 twin (adapter/wo_a_w8.py) when one exists
    for this weight (2+ rows, or always once the bf16 copy was dropped), else the bf16 weight."""
    sig = _state["sig"]
    if (sig is None or sig not in _state["learning"] or not getattr(wo_a, "is_cuda", False)
            or not _sub("WOA")):
        return
    w8 = sys.modules.get("wo_a_w8")
    twin = None
    if w8 is not None and getattr(w8, "ENABLED", False):
        ptr = wo_a.data_ptr()
        twin = w8._TWINS.get(ptr)
        if twin is not None and o.shape[0] < 2 and ptr not in w8._DROPPED:
            twin = None  # single-row decode keeps the bf16 GEMV
    # twin = (e4m3 [2,1024,4096], exponents [2,32,128]): the 8 KB of exponents first, every tile needs them
    tensors = twin[::-1] if twin is not None else ((wo_a,) if 0 not in wo_a.stride() else ())
    rec = _state["learning"][sig]
    for t in tensors:
        rec.append(("w", t.data_ptr(), t.numel() * t.element_size()))


_WOA_MODULES = ("sglang.srt.models.deepseek_v4", "sglang.srt.models.deepseek_v4_dspark")


def _wrap_wo_a(inner):
    def _apply_wo_a_bf16_matmul(o, wo_a, *args, **kwargs):
        if _state["woa_depth"] == 0:  # outermost wrapper only: one record per call
            _record_wo_a(o, wo_a)
        _state["woa_depth"] += 1
        try:
            return inner(o, wo_a, *args, **kwargs)
        finally:
            _state["woa_depth"] -= 1

    _apply_wo_a_bf16_matmul._dsv41_l2_prefetch = True
    return _apply_wo_a_bf16_matmul


def _ensure_wo_a(modules=None):
    """Keep the wo_a recorder outermost on the dispatch the target and the draft call by name.
    Adapters installed after this one (DSV41_FUSE_QUANT) wrap _apply_wo_a_bf16_matmul again and
    serve M 2..8 without calling the inner function, so it is re-checked at every bracketed
    forward (the draft module holds its own reference, imported by name)."""
    for mod in modules or [sys.modules.get(n) for n in _WOA_MODULES]:
        fn = getattr(mod, "_apply_wo_a_bf16_matmul", None) if mod is not None else None
        if fn is not None and not getattr(fn, "_dsv41_l2_prefetch", False):
            mod._apply_wo_a_bf16_matmul = _wrap_wo_a(fn)


def install_fp8_utils(module):
    """Wrap the MXFP8 dense linear (after mxfp8_b12x) so each call records its weight + scales."""
    if not _installed() or ("fp8", id(module)) in _state["installed"]:
        return
    _state["installed"].add(("fp8", id(module)))
    original = module.flashinfer_mxfp8_blockscaled_linear
    woa = _installed("WOA")

    def linear(input, weight, weight_scale, *args, **kwargs):
        _record_linear(weight, weight_scale)
        out = original(input, weight, weight_scale, *args, **kwargs)
        if woa and _sub("WOA"):
            _after_linear()
        return out

    module.flashinfer_mxfp8_blockscaled_linear = linear
    logger.warning("DSV41_L2_PREFETCH: MXFP8 linears recorded (budget %.1f MB/window)", _budget() / 2**20)


def install_router(module):
    """Record the bf16 router weight (tiny_gemm_bf16 in deepseek_v2, 3.9 MB, read after the
    post-attention all-reduce at ~140 GB/s)."""
    if not _installed() or not hasattr(module, "tiny_gemm_bf16") or ("router", id(module)) in _state["installed"]:
        return
    _state["installed"].add(("router", id(module)))
    original = module.tiny_gemm_bf16

    def tiny_gemm_bf16(hidden_states, weight, *args, **kwargs):
        _record_weight(weight)
        out = original(hidden_states, weight, *args, **kwargs)
        if _state["sig"] is not None and _sub("MOE"):
            _after_router()
        return out

    module.tiny_gemm_bf16 = tiny_gemm_bf16


def install_roce(module):
    """Patch b12x.comm.roce(_ring).roce_oneshot: launch the prefetch right before each collective."""
    if not _installed():
        return
    cls = module.RoceOneshotAllReduce
    if getattr(cls, "_dsv41_l2_prefetch", False):
        return
    cls._dsv41_l2_prefetch = True
    ar, ag = cls.all_reduce, cls.all_gather

    def all_reduce(self, inp, *args, **kwargs):
        if self.should_allreduce(inp):
            _before_collective("ar", kwargs.get("stream"))
        return ar(self, inp, *args, **kwargs)

    def all_gather(self, inp, *args, **kwargs):
        _before_collective("ag", kwargs.get("stream"))
        return ag(self, inp, *args, **kwargs)

    cls.all_reduce, cls.all_gather = all_reduce, all_gather
    _lib()
    logger.warning("DSV41_L2_PREFETCH: RoCE collectives prefetch the next weights into L2")


def _bracket(cls, lm_head=False, gate=None):
    """Wrap cls.forward(self, input_ids, positions, forward_batch, ...): decode / target-verify /
    draft-extend forwards are learned and prefetched, prefill (compute-bound) is left alone."""
    original = cls.forward

    def forward(self, input_ids, positions, forward_batch, *args, **kwargs):
        mode = getattr(forward_batch, "forward_mode", None)
        decode_like = mode is not None and any(
            getattr(mode, f, lambda: False)() for f in ("is_decode", "is_target_verify", "is_draft_extend"))
        if (not decode_like or _state["sig"] is not None or not enabled()
                or (gate is not None and not _sub(gate))):  # per-variant under DSV41_AB_VARIANTS
            return original(self, input_ids, positions, forward_batch, *args, **kwargs)
        sig = (id(self), int(input_ids.shape[0]), str(mode)) + _ab_tag()
        _begin_forward(sig)
        try:
            out = original(self, input_ids, positions, forward_batch, *args, **kwargs)
            if lm_head and _sub("LMHEAD"):
                _record_weight(_state["lm_head"].get(id(self)))
            return out
        finally:
            _end_forward()

    cls.forward = forward


def install_model(module):
    """Bracket DeepseekV4Model.forward (decode / target-verify only; prefill is compute-bound)."""
    if not _installed():
        return
    cls = module.DeepseekV4Model
    if getattr(cls, "_dsv41_l2_prefetch", False):
        return
    cls._dsv41_l2_prefetch = True
    for name in _ROCE_MODULES:
        if name in sys.modules:  # imported before the finder saw it
            install_roce(sys.modules[name])
    lm_head = _installed("LMHEAD")
    _bracket(cls, lm_head=lm_head)
    extras = []
    if _installed("WOA"):
        _ensure_wo_a([module])
        extras.append(f"wo_a windows ({_mb('DSV41_L2_PREFETCH_WOA_MB', '6') / 2**20:.1f} MB)")
    if lm_head:
        causal = module.DeepseekV4ForCausalLM
        orig_causal = causal.forward

        def causal_forward(self, *args, **kwargs):
            head = getattr(getattr(self, "lm_head", None), "weight", None)
            if head is not None and id(self.model) not in _state["lm_head"]:
                _state["lm_head"][id(self.model)] = head
            return orig_causal(self, *args, **kwargs)

        causal.forward = causal_forward
        extras.append("LM head")
    if _installed("AHEAD"):
        extras.append("look-ahead")
    logger.warning("DSV41_L2_PREFETCH: model forward bracketed (decode/verify only)%s",
                   f"; v3: {', '.join(extras)}" if extras else "")


def install_draft(module):
    """DSV41_L2_PREFETCH_DRAFT: bracket the DSpark draft forward (TARGET_VERIFY mode, its own
    CUDA graphs) the same way, so its collectives prefetch the next draft stage's weights."""
    if not _installed("DRAFT"):
        return
    cls = module.DeepseekV4ForCausalLMDSpark
    if getattr(cls, "_dsv41_l2_prefetch", False):
        return
    cls._dsv41_l2_prefetch = True
    for name in _ROCE_MODULES:
        if name in sys.modules:  # imported before the finder saw it
            install_roce(sys.modules[name])
    _bracket(cls, gate="DRAFT")
    logger.warning("DSV41_L2_PREFETCH: DSpark draft forward bracketed")


def install_engram(module):
    """DSV41_L2_PREFETCH_ENGRAM: the Engram row join (EngramEmbedding._owned_rows, which waits for
    the host-side row lookups) forks a prefetch of the engram.wkv slice read right after it."""
    if not _installed("ENGRAM"):
        return
    emb = module.EngramEmbedding
    if getattr(emb, "_dsv41_l2_prefetch", False):
        return
    emb._dsv41_l2_prefetch = True
    original = emb._owned_rows

    def _owned_rows(self, indices, *args, **kwargs):
        _before_engram()
        return original(self, indices, *args, **kwargs)

    emb._owned_rows = _owned_rows
    logger.warning("DSV41_L2_PREFETCH: Engram row joins prefetch %.1f MB of engram.wkv",
                   _mb("DSV41_L2_PREFETCH_ENGRAM_MB", "12") / 2**20)
