"""Routed MoE through b12x main (package ``b12x_next``) instead of FlashInfer CUTLASS. Default OFF.

Gate: ``DSV41_MOE_B12X_NEXT=1``. Purpose: EP_SIZE=1 at TP4 (every rank holds a 576-wide slice of all
384 routed experts), which FlashInfer's SM120 CUTLASS MXFP4 path cannot run (intermediate % 128, and
``block_scale_interleave`` pads 18 scale columns). The same adapter runs EP_SIZE=2 (N=1152) for A/B.
Microbenchmark (diagnostics/dsv41-b12x-moe, 2026-09-24): per-rank MoE -8 % vs FI CUTLASS at equal
experts; at EP1 the rank streams all ~17 experts of a c1 verify step at half width, so there is no
EP-group straggler.

What it patches (all refuse to boot if the engine symbol drifted):

* ``Mxfp4FlashinferCutlassMoEMethod.create_weights``: the ``% 128`` check is lifted (576 needs
  ``% 64``); the checkpoint-layout tensors are created exactly as the stock method does
  (``Fp8MoEMethod.create_weights`` with E8M0 scales). The FusedMoE loader narrows w1/w3 rows by
  ``2304 // moe_tp_size`` and w2 columns / scales by the same factor, and with
  ``load_up_proj_weight_first`` stacks w13 as [w3 (up); w1 (gate)], which is b12x ``W13Layout.W13``
  (verified numerically against an fp32 reference built from the checkpoint slices).
* ``...process_weights_after_loading``: no FlashInfer interleave; the checkpoint bytes are repacked
  in place into b12x W4A8 (MXFP4 weights, in-kernel MXFP8 activations) ``PreparedExperts``, one
  shared ``WeightPlan`` per geometry.
* ``FusedOpPool[("none", "flashinfer_mxfp4")]``: replaced by ``_fused_b12x_next``. A layer that
  reaches it without having been converted refuses to run: the patched ``process_weights_after_loading``
  left its weights in checkpoint layout, which the stock FlashInfer kernel would read as garbage.

Execution plans are per geometry (experts, K, N, top-k, EP rank, swiglu limit), prepared once at
load on the first layer and shared by every layer of that geometry (the binding carries each layer's
own prepared experts; see ``_Geometry._run``). Capacities: every exact decode/verify M of the CUDA
graph list (verify rows B+1 and draft rows B per request, B = DSPARK_BLOCK_SIZE) plus a bounded
ladder (128 .. max(4096, CHUNKED_PREFILL_SIZE)); a batch uses the smallest capacity >= M, larger
batches run in chunks of the top capacity. A capacity above b12x's per-launch token limit for the
geometry is dropped (warning), so the chunk is always a prepared capacity one launch can run.
Scratch: one arena per geometry sized for the largest plan. EP: a Triton kernel maps global top-k
ids to local ids in-graph; slots owned by another EP group and CUDA-graph padding ids (-1) get
weight 0 and point at an expert the batch already reads.

Autotune: the exact-M plans of the primary row count are raced at load (b12x PreparationSession),
the rest use b12x's heuristic. Selections persist in ``$B12X_NEXT_COMPILE_CACHE_DIR/preparation``
(default /state/b12x-next-compile), separate from the SG17 b12x cache RoCEnante uses. Tuning is
rank-local (no collectives); ranks may pick different tactics, and every rank prepares the same set
of plans, so the M -> plan mapping is rank-invariant. Race buffers are released after preparation.

Knobs: DSV41_MOE_B12X_NEXT_TUNE (1), _TUNE_ROWS ("6:6,3:5" top-k:rows raced), _GRAPH_BS
("1,2,3,4,5,6,7,8,10,12,14,16"), _LADDER ("128,256,512,1024,2048,4096"), _CACHE_ONLY (0),
_WARM (1: run every plan once at load), _DETERMINISTIC (0; EP1 only, b12x rejects it at N=1152;
nothing is raced: b12x's race cannot compile the deterministic top-k sum kernel), _DET_TRITON (1:
under _DETERMINISTIC the pinned plans keep the Triton route planner, which needs the runtime patch
scripts/b12x_next-det-triton-planner.patch, else internal with a warning; 0: internal planner),
_MEMSTATS (0: 1 logs load-time peak transients, resetting torch's peak counters),
_SMALL_PLAN ("triton:48:16": route planner, max active clusters, tile rows of the dynamic plan pinned
at <= 8 rows for compact N64), _PLAN_TABLE ("1-8=<_SMALL_PLAN>": per-capacity compact-N64 plans,
``lo-hi[@k<top-k>]=<planner>:<mac|none>:<tile>`` or ``=heur``, first match wins; replaces the whole
table, unlisted capacities run b12x's heuristic), _M64_MIN_CAP (2048: compact-N64 ladder capacities >= this run the
M64 tile; 0 = off; needs the runtime patch scripts/b12x_next-compact-n64-m64.patch, else M16 with a
warning),
_DIRECT_IDS (1: at EP_SIZE=1 the router's int32 ids / fp32 weights go straight to b12x, no remap
kernel; b12x skips ids outside [0, E), so CUDA-graph padding rows (-1) contribute nothing and their
output rows are 0. Off, or at EP>1, or with other dtypes / _DETERMINISTIC, the remap kernel runs).

Pre-quantized input (prefill sequence parallel, DSV41_PREFILL_SP_FP8_MOE=1; needs the runtime patch
scripts/b12x_next-prequant-input.patch): at prefill capacities b12x's front-end stores one MXFP8 row
per token (E4M3 [token, K], UE8M0 [token, K/32]) in the plan's packed_input / packed_input_scale
and the FC1 phase reads it from there. ``prequant_views(layer, M)`` returns those two views, the
caller writes the rows (an all-gather of per-shard MXFP8 rows), and a dispatch carrying
``hidden_states_pre_quant`` with the ``_dsv41_b12x_rows`` marker launches the front-end variant
that skips its own quantization. Same bytes -> same FC1/FC2 inputs; whether the caller's quantizer
gives b12x's bytes is prefill_sp's runtime check, not assumed here. The variant is compiled at load
for the prefill capacities of the top-k _PREQUANT_TOPK (6) geometry when prefill SP FP8-MoE is on.
"""
import bisect
import dataclasses
import inspect
import logging
import os
import sys
import time

import torch

logger = logging.getLogger(__name__)

ENABLED = os.environ.get("DSV41_MOE_B12X_NEXT", "0").strip() not in ("0", "", "off", "false")
PINNED_COMMIT = "a7d7d29b2ef8869086e0ceaa787321f17544e3c9"
_FUSED_KEY = ("none", "flashinfer_mxfp4")
_TAG = "[moe_b12x_next]"


def _env(name, default):
    return os.environ.get(name, default).strip()


def _ints(text):
    return [int(v) for v in text.replace(" ", "").split(",") if v]


TUNE = _env("DSV41_MOE_B12X_NEXT_TUNE", "1") not in ("0", "off", "false", "")
CACHE_ONLY = _env("DSV41_MOE_B12X_NEXT_CACHE_ONLY", "0") not in ("0", "off", "false", "")
WARM = _env("DSV41_MOE_B12X_NEXT_WARM", "1") not in ("0", "off", "false", "")
# 1 = b12x's deterministic reduction (per-slot route buffer + top-k sum) instead of atomics: bit-stable
# run to run; the atomic default differs by bf16 ulps between runs.
DETERMINISTIC = _env("DSV41_MOE_B12X_NEXT_DETERMINISTIC", "0") not in ("0", "off", "false", "")
# 1 = measure load-time peaks with torch's peak counters (resets them); off in serving by default.
MEMSTATS = _env("DSV41_MOE_B12X_NEXT_MEMSTATS", "0") not in ("0", "off", "false", "")
# 1 = EP1 skips the ep_remap launch (3 us/layer): b12x's dynamic/micro kernels and the triton route
# planner all bounds-check expert ids, so -1 padding needs no rewrite (verified: numerics_noremap.py).
DIRECT_IDS = _env("DSV41_MOE_B12X_NEXT_DIRECT_IDS", "1") not in ("0", "off", "false", "")
# Pre-quantized prefill input (see the docstring). Warm (compile at load) only when prefill SP will
# use it; the capacities from the SP row floor up.
_ON = ("1", "on", "true")
PREQUANT_WARM = (_env("DSV41_PREFILL_SP_FP8_MOE", "0").lower() in _ON
                 and _env("DSV41_PREFILL_SP", "0").lower() in _ON + ("shard",))
PREQUANT_MIN_ROWS = int(_env("DSV41_PREFILL_SP_MIN_ROWS", "2048") or 2048)
PREQUANT_TOPK = int(_env("DSV41_MOE_B12X_NEXT_PREQUANT_TOPK", "6") or 6)
PREQUANT_STATS = {"calls": 0, "warm_caps": {}}


def _peak_begin(dev):
    torch.cuda.synchronize(dev)
    base = torch.cuda.memory_allocated(dev)
    if MEMSTATS:
        torch.cuda.reset_peak_memory_stats(dev)
    return base


def _peak_mb(dev, base):
    torch.cuda.synchronize(dev)
    return round((torch.cuda.max_memory_allocated(dev) - base) / 2**20, 1) if MEMSTATS else None



GRAPH_BS = _ints(_env("DSV41_MOE_B12X_NEXT_GRAPH_BS", "1,2,3,4,5,6,7,8,10,12,14,16"))
_MAX_BS = int(_env("CUDA_GRAPH_MAX_BS_DECODE", "16") or 16)
GRAPH_BS = [b for b in GRAPH_BS if b <= _MAX_BS] or [1]
BLOCK = int(_env("DSPARK_BLOCK_SIZE", "5") or 5)
LADDER = _ints(_env("DSV41_MOE_B12X_NEXT_LADDER", "128,256,512,1024,2048,4096"))
# Prefill capacities of the compact N64 geometry (EP1, N=576) on the M64 tile instead of b12x's M16 pin.
M64_MIN_CAP = int(_env("DSV41_MOE_B12X_NEXT_M64_MIN_CAP", "2048") or 0)
# Decode plan pinned for compact N64 at <= 8 rows: planner:max_active_clusters:tile_m (dynamic, grouped).
_SMALL_SPEC = _env("DSV41_MOE_B12X_NEXT_SMALL_PLAN", "triton:48:16")
_SMALL = _SMALL_SPEC.split(":")
SMALL_PLANNER, SMALL_MAC, SMALL_TILE = _SMALL[0], (int(_SMALL[1]) if _SMALL[1] != "none" else None), int(_SMALL[2])
# 1 = keep the Triton route planner of the pinned small plans under DETERMINISTIC. Needs the runtime
# patch scripts/b12x_next-det-triton-planner.patch (stock b12x refuses the combination); without it
# _b12x() falls back to the internal planner. The planner only writes per-expert row counts, the tile
# prefix and the barrier words (order-free integers), so the output is bit-identical to the internal
# planner; it replaces 2 zero-fill launches + 2 grid barriers + the one-warp prefix scan of the
# front-end with one 1-CTA Triton launch. Measured (diagnostics/dsv41-moe-launch): -3.5 us per call at
# E=128, bit-identical on 9530 cases (M 1-8, 12-64, eager and graph).
DET_TRITON = _env("DSV41_MOE_B12X_NEXT_DET_TRITON", "1") not in ("0", "off", "false", "")
if DETERMINISTIC and SMALL_PLANNER == "triton" and not DET_TRITON:
    # b12x's Triton route planner refuses deterministic compact queries (_tuning validation)
    SMALL_PLANNER = "internal"


def parse_plan_table(text, deterministic):
    """``lo-hi[@k<top-k>]=<planner>:<mac|none>:<tile>`` or ``...=heur``, comma-separated; first match wins.

    Returns [(lo, hi, top-k or None, (planner, mac, tile) or None)]. ``heur`` = no override (b12x's
    heuristic plan). Under determinism a ``triton`` planner becomes ``internal`` (b12x refuses it)."""
    table = []
    for item in text.replace(" ", "").split(","):
        if not item:
            continue
        try:
            rng, spec = item.split("=")
            topk = None
            if "@k" in rng:
                rng, k = rng.split("@k")
                topk = int(k)
            lo, hi = (int(v) for v in rng.split("-")) if "-" in rng else (int(rng), int(rng))
            if spec == "heur":
                plan = None
            else:
                planner, mac, tile = spec.split(":")
                if planner not in ("internal", "triton"):
                    raise ValueError(planner)
                if deterministic and planner == "triton":
                    planner = "internal"
                plan = (planner, None if mac == "none" else int(mac), int(tile))
        except ValueError as exc:
            raise RuntimeError(f"DSV41_MOE_B12X_NEXT_PLAN_TABLE: bad entry {item!r} ({exc})") from exc
        if lo < 1 or hi < lo:
            raise RuntimeError(f"DSV41_MOE_B12X_NEXT_PLAN_TABLE: bad range in {item!r}")
        table.append((lo, hi, topk, plan))
    return table


# Per-capacity decode plans of the compact N64 geometry (EP1, N=576). The default is the <= 8 row pin
# above; every other capacity runs b12x's heuristic (dynamic, internal planner, mac none, M16, grouped).
# Measured 2026-09-25 on the engine's path under determinism (diagnostics/dsv41-moe-plan-table): at
# M = 10..96 (target top-6) and 10..80 (draft top-3) no max_active_clusters / tile choice beat the
# heuristic beyond noise; every candidate was bit-identical to it; the calls run at the ~214 GB/s MoE
# read ceiling, so the per-call cost is set by the distinct experts' bytes, not the plan.
_PLAN_TABLE_SPEC = _env("DSV41_MOE_B12X_NEXT_PLAN_TABLE", f"1-8={_SMALL_SPEC}")
PLAN_TABLE = parse_plan_table(_PLAN_TABLE_SPEC, DETERMINISTIC and not DET_TRITON)


def _det_triton_supported(fimpl):
    """True when the b12x_next runtime admits the Triton route planner with deterministic output."""
    try:
        from b12x_next.moe.fused_moe import _tuning
        gate = inspect.getsource(fimpl._dynamic_external_route_plan_supported)
        val = inspect.getsource(_tuning.validate_moe_decode_config)
    except Exception:  # noqa: BLE001
        return False
    return ("not deterministic_output" not in gate
            and "_compact_w4a8_query(query) and not query.deterministic_output" not in val)


def table_plan(cap, topk):
    """(planner, mac, tile) pinned for this capacity / top-k, or None (b12x heuristic)."""
    for lo, hi, k, plan in PLAN_TABLE:
        if lo <= cap <= hi and (k is None or k == topk):
            return plan
    return None
_CHUNK_ENV = int(_env("CHUNKED_PREFILL_SIZE", "0") or 0)
if _CHUNK_ENV > max(LADDER):
    LADDER.append(_CHUNK_ENV)
TUNE_ROWS = {int(k): int(v) for k, v in (p.split(":") for p in
             _env("DSV41_MOE_B12X_NEXT_TUNE_ROWS", "6:6,3:5").split(",") if p)}
EXACT_MAX = 96 if not GRAPH_BS else max(96, (BLOCK + 1) * max(GRAPH_BS))

_state = {"method_installed": False, "runner_installed": False, "orig_fused": None, "b12x_ready": False,
          "ep_logged": False}
_LAYERS = {}        # w13_weight.data_ptr() -> _LayerState
_GEOMS = {}         # geometry key -> _Geometry
_SESSION = [None]
_PREP_PEAK_MB = []    # per-layer peak transient of prepare_weights
_B = {}             # b12x_next modules, filled only by _b12x() after the pin/drift checks
_ENG = {}           # engine helpers of the fused function (install_runner)


def _die(msg):
    raise RuntimeError(f"DSV41_MOE_B12X_NEXT: {msg} -- refusing to boot (unset DSV41_MOE_B12X_NEXT "
                       f"to run the stock FlashInfer CUTLASS path)")


# --------------------------------------------------------------------------------------------
# b12x_next import and drift checks
# --------------------------------------------------------------------------------------------
def _b12x():
    if _state["b12x_ready"]:
        return _B
    if not os.environ.get("B12X_NEXT_COMPILE_CACHE_DIR"):
        state = os.environ.get("STATE_PATH", "/state")
        root = os.path.join(state, "b12x-next-compile")
        try:
            os.makedirs(root, exist_ok=True)
            os.environ["B12X_NEXT_COMPILE_CACHE_DIR"] = root
        except OSError:
            pass
    if os.environ.get("B12X_NEXT_COMPILE_CACHE_DIR") == os.environ.get("B12X_COMPILE_CACHE_DIR"):
        _die("B12X_NEXT_COMPILE_CACHE_DIR must differ from the SG17 B12X_COMPILE_CACHE_DIR")
    try:
        import b12x_next
        from b12x_next import preparation as prep
        from b12x_next.moe import fused_moe as fm
        from b12x_next.moe.fused_moe import _preparation as fprep
        from b12x_next.moe.fused_moe import api as fapi
    except Exception as exc:
        _die(f"cannot import b12x_next ({exc!r}); put runtime/b12x_next on PYTHONPATH")
    commit_file = os.path.join(os.path.dirname(b12x_next.__file__), "SOURCE_COMMIT")
    try:
        commit = open(commit_file).read().strip()
    except OSError:
        commit = "unknown"
    if commit != PINNED_COMMIT:
        _die(f"b12x_next is built from {commit}, the adapter is pinned to {PINNED_COMMIT}")
    src = inspect.getsource(fprep._FusedMoeState.bind)
    for needle in ('kwargs["experts"] = self.experts._impl', "self.scratch.bind(**kwargs)",
                   "compact_launches=self.compact_launches", "unit_scale_contract",
                   "_w4a16_launches"):
        if needle not in src:
            _die(f"b12x_next _FusedMoeState.bind drifted (missing {needle!r})")
    if "replace(state.bind(**kwargs), plan=plan)" not in inspect.getsource(fapi.bind):
        _die("b12x_next fused_moe.bind drifted")
    try:        # the prequant-input runtime patch (optional: without it prefill SP keeps bf16 gathers)
        from b12x_next.moe.fused_moe import _impl as fimpl
    except ImportError:
        fimpl = None
    if DETERMINISTIC and DET_TRITON and not _det_triton_supported(fimpl):
        global PLAN_TABLE
        PLAN_TABLE = parse_plan_table(_PLAN_TABLE_SPEC, True)
        print(f"{_TAG} WARNING: DSV41_MOE_B12X_NEXT_DET_TRITON=1 but this b12x_next lacks the "
              f"det-triton-planner patch; deterministic plans use the internal planner", flush=True)
    _B.update(fm=fm, prep=prep, pkg=b12x_next,
              impl=fimpl if callable(getattr(fimpl, "prequantized_input", None)) else None)
    _state["b12x_ready"] = True
    return _B


# --------------------------------------------------------------------------------------------
# EP remap kernel: global top-k ids -> local ids; foreign / padding slots -> weight 0 on an
# expert the batch already reads (the row's lowest local expert, else the batch's lowest).
# --------------------------------------------------------------------------------------------
_REMAP = [None]


def _remap_kernel():
    if _REMAP[0] is None:
        import triton
        import triton.language as tl

        # Input strides are runtime arguments (no specialization, so a strided view never JITs a
        # new variant inside graph capture); the outputs are contiguous [M, K].
        @triton.jit(do_not_specialize=["M", "OFFSET", "NUM_LOCAL", "SI0", "SI1", "SW0", "SW1"])
        def kernel(IDS, W, OUT_IDS, OUT_W, M, OFFSET, NUM_LOCAL, SI0, SI1, SW0, SW1,
                   K: tl.constexpr, KP: tl.constexpr, BR: tl.constexpr):
            cols = tl.arange(0, KP)
            cmask = cols < K
            best = tl.full((), 2147483647, tl.int32)
            for r0 in range(0, M, BR):
                rows = r0 + tl.arange(0, BR)
                m = (rows[:, None] < M) & cmask[None, :]
                lid = tl.load(IDS + rows[:, None] * SI0 + cols[None, :] * SI1, mask=m, other=-1).to(tl.int32) - OFFSET
                ok = m & (lid >= 0) & (lid < NUM_LOCAL)
                best = tl.minimum(best, tl.min(tl.min(tl.where(ok, lid, 2147483647), axis=1), axis=0))
            fb = tl.where(best < NUM_LOCAL, best, 0)
            for r0 in range(0, M, BR):
                rows = r0 + tl.arange(0, BR)
                m = (rows[:, None] < M) & cmask[None, :]
                offs = rows[:, None] * K + cols[None, :]
                lid = tl.load(IDS + rows[:, None] * SI0 + cols[None, :] * SI1, mask=m, other=-1).to(tl.int32) - OFFSET
                w = tl.load(W + rows[:, None] * SW0 + cols[None, :] * SW1, mask=m, other=0.0).to(tl.float32)
                ok = m & (lid >= 0) & (lid < NUM_LOCAL)
                row_min = tl.min(tl.where(ok, lid, 2147483647), axis=1)
                fill = tl.where(row_min < NUM_LOCAL, row_min, fb)
                tl.store(OUT_IDS + offs, tl.where(ok, lid, fill[:, None]), mask=m)
                tl.store(OUT_W + offs, tl.where(ok, w, 0.0), mask=m)

        _REMAP[0] = (kernel, triton.next_power_of_2)
    return _REMAP[0]


def ep_remap(topk_ids, topk_weights, out_ids, out_w, offset, num_local):
    """Writes out_ids (int32) / out_w (fp32), contiguous [M, K]; one CTA, no host sync, graph-capturable.

    ``topk_ids`` / ``topk_weights`` may be any strided [M, K] views (each read by its own strides)."""
    kernel, np2 = _remap_kernel()
    m, k = topk_ids.shape
    if m == 0:
        return
    if tuple(topk_weights.shape) != (m, k) or not (out_ids.is_contiguous() and out_w.is_contiguous()) \
            or out_ids.shape[0] < m or out_w.shape[0] < m:
        raise RuntimeError(f"DSV41_MOE_B12X_NEXT: ep_remap shapes ids {tuple(topk_ids.shape)} weights "
                           f"{tuple(topk_weights.shape)} out {tuple(out_ids.shape)}/{tuple(out_w.shape)}")
    kernel[(1,)](topk_ids, topk_weights, out_ids, out_w, m, offset, num_local,
                 topk_ids.stride(0), topk_ids.stride(1), topk_weights.stride(0), topk_weights.stride(1),
                 K=k, KP=np2(k), BR=32, num_warps=4)


def ep_remap_reference(topk_ids, topk_weights, offset, num_local):
    """torch version of ep_remap (tests)."""
    lid = topk_ids.to(torch.int64) - offset
    ok = (lid >= 0) & (lid < num_local)
    big = torch.full_like(lid, 2**31 - 1)
    best = torch.where(ok, lid, big).min()
    fb = best if int(best) < num_local else torch.zeros_like(best)
    row_min = torch.where(ok, lid, big).min(dim=1).values
    fill = torch.where(row_min < num_local, row_min, fb)
    ids = torch.where(ok, lid, fill[:, None].expand_as(lid)).to(torch.int32)
    w = torch.where(ok, topk_weights.float(), torch.zeros_like(topk_weights, dtype=torch.float32))
    return ids, w


# --------------------------------------------------------------------------------------------
# per-geometry plans
# --------------------------------------------------------------------------------------------
def _session():
    if _SESSION[0] is None:
        B = _b12x()
        _SESSION[0] = B["prep"].PreparationSession(
            device=torch.device("cuda", torch.cuda.current_device()), autotune=TUNE,
            compile_workers=0, cache_only=CACHE_ONLY)
    return _SESSION[0]


@dataclasses.dataclass
class _PlanEntry:
    cap: int
    plan: object
    state: object
    specs: tuple
    views: tuple = ()
    tuned: bool = False
    source: str = ""
    config: str = ""
    max_per_launch: int = 0


class _Geometry:
    def __init__(self, key, num_experts, hidden, inter, topk, ep_rank, limit, device, ep_size=1):
        B = _b12x()
        fm = B["fm"]
        self.key = key
        self.E, self.K, self.N, self.topk = num_experts, hidden, inter, topk
        self.offset = ep_rank * num_experts
        self.direct = DIRECT_IDS and int(ep_size) == 1 and not DETERMINISTIC
        self.device = device
        self.limit = limit
        self.weight_plan = fm.plan_weights(
            source=fm.PackedSource(format=fm.PackedSourceFormat.MXFP4_E8M0_K32, w13_layout=fm.W13Layout.W13),
            activation=fm.ActivationSpec(mode=fm.ActivationMode.A8, nonlinearity="silu",
                                         io_dtype=torch.bfloat16, swiglu_limit=limit),
            geometry=fm.MoEGeometry(num_experts=num_experts, hidden_size=hidden, intermediate_size=inter))
        self.plans = {}
        self.caps = []
        self.table = []
        self.chunk = 0
        self.arena = None
        self.ids_buf = None
        self.w_buf = None
        self.ready = False
        self.report = {}
        self.impl0 = None           # any layer's prepared experts (binding views do not depend on it)
        self.pq_views = {}          # capacity -> (packed_input [rows, K] u8, packed_input_scale flat u8)

    def capacities(self):
        rows = sorted({r * b for r in (BLOCK, BLOCK + 1) for b in GRAPH_BS})
        exact = [m for m in rows if m <= EXACT_MAX]
        return sorted(set(exact) | {c for c in LADDER if c > max(exact)})

    def tuned_caps(self):
        # DETERMINISTIC: no race. b12x does not declare the deterministic top-k sum kernel among a
        # plan's programs, so its race refuses to compile it ("unplanned CuTe program"); the
        # heuristic plans compile it in _warm instead.
        if not TUNE or DETERMINISTIC:
            return set()
        r = TUNE_ROWS.get(self.topk)
        if r is None:
            return set()
        return {r * b for b in GRAPH_BS}

    def build(self, experts):
        """Prepare every capacity on the first layer's experts; allocate the shared arena."""
        B = _b12x()
        fm, prep = B["fm"], B["prep"]
        dev = self.device
        base = _peak_begin(dev)
        t0 = time.time()
        session = _session()
        tuned = self.tuned_caps()

        def noop(state):
            return prep.PreparedCall(run=lambda: None)

        for cap in self.capacities():
            routing = fm.RoutingSpec(deterministic_output=True) if DETERMINISTIC else None
            kw = {}
            pinned = table_plan(cap, self.topk) if self.N % 128 == 64 else None
            if pinned is not None:
                # <= 8 rows (default table): b12x's heuristic (and a race on uniform routing) picks the
                # `micro` plan, which streams one expert slice per (token, top-k slot): 36 at a c1
                # verify step, where `dynamic` streams each distinct expert once (~16 in real decode,
                # fewer with dead verify rows). Measured on the engine's path: 523 -> 349 us per call
                # at EP1, M=6. Other capacities: DSV41_MOE_B12X_NEXT_PLAN_TABLE.
                planner, mac, tile = pinned
                kw["override"] = fm.MoeDecodeConfig(
                    backend="dynamic", route_planner=planner, max_active_clusters=mac,
                    dynamic_tile_m=tile, dynamic_route_mode="grouped")
            elif M64_MIN_CAP and cap >= M64_MIN_CAP and cap > EXACT_MAX and self.N % 128 == 64 \
                    and _m64_admitted():
                # b12x pins compact N64 to M16 tiles at every capacity: each 16-row tile re-stages the
                # expert's weight slice, 4x the M64 traffic. Prefill caps run M64 (runtime patch admits
                # it; M32 is wrong on n64 and stays rejected). Numerics equal the M16 path (rel-L2 vs
                # fp32 4.67 % both). Measured on the engine's path, EP1 N=576, per call, M16 -> M64:
                # 6 experts x M rows: -34 % at 2048, -28 % at 4096; spread routing -1 % / -4 %, but
                # +0.5..3 % at caps <= 1024, hence the 2048 floor.
                kw["override"] = fm.MoeDecodeConfig(
                    backend="dynamic", route_planner="internal", max_active_clusters=None,
                    dynamic_tile_m=64, dynamic_route_mode="grouped")
            plan = fm.plan_execution(experts=experts, routing=routing,
                                     capacity=fm.ExecutionCapacity(max_tokens=cap, top_k=self.topk), **kw)
            race = self._race_call(cap) if cap in tuned and not kw else None
            req = plan.request(name=f"dsv41-moe-E{self.E}-N{self.N}-k{self.topk}-M{cap}",
                               prepare_call=noop, benchmark_call=race)
            session.prepare((req,), autotune=race is not None and TUNE)
            state = plan.prepared.state
            sel = plan.prepared.selection
            lp = state.scratch.launch_plan
            self.plans[cap] = _PlanEntry(
                cap=cap, plan=plan, state=state, specs=tuple(plan.scratch_specs()), tuned=race is not None,
                source=str(getattr(sel, "source", "")), config=str(getattr(sel, "config", "")),
                max_per_launch=int(getattr(lp, "max_tokens_per_launch", cap) or cap))
        self.chunk, self.caps, dropped = select_chunk({c: e.max_per_launch for c, e in self.plans.items()})
        if dropped:
            limit = min(self.plans[c].max_per_launch for c in dropped)
            print(f"{_TAG} WARNING: E={self.E} K={self.K} N={self.N} top-k {self.topk}: capacities {dropped} "
                  f"exceed b12x's per-launch limit of {limit} tokens and are dropped; larger batches run in "
                  f"chunks of {self.chunk} (set CHUNKED_PREFILL_SIZE <= {limit} to prepare a larger plan)",
                  flush=True)
            for c in dropped:
                del self.plans[c]
        self.table = [None] + [self._lookup(m) for m in range(1, EXACT_MAX + 1)]
        # one arena per geometry, each plan's specs carved from its start (plans never overlap in time)
        need = max(sum(_align(s.nbytes) for s in self.plans[c].specs) for c in self.caps)
        self.arena = torch.empty(max(need, 256), dtype=torch.uint8, device=dev)
        for c in self.caps:
            off, views = 0, []
            for s in self.plans[c].specs:
                nb = s.nbytes
                views.append(self.arena[off:off + nb].view(s.dtype).view(tuple(s.shape)))
                off += _align(nb)
            self.plans[c].views = tuple(views)
        self.ids_buf = torch.empty(self.chunk, self.topk, dtype=torch.int32, device=dev)
        self.w_buf = torch.empty(self.chunk, self.topk, dtype=torch.float32, device=dev)
        self.ready = True
        self.impl0 = experts._impl
        t_prep = time.time() - t0
        if WARM:
            self._warm(experts)
        if PREQUANT_WARM and self.topk == PREQUANT_TOPK:
            self._warm_prequant(experts)
        peak = _peak_mb(dev, base)
        torch.cuda.empty_cache()
        persistent = self.arena.numel() + self.ids_buf.numel() * 8
        self.report = {
            "geometry": dict(E=self.E, K=self.K, N=self.N, topk=self.topk, offset=self.offset),
            "capacities": self.caps, "chunk": self.chunk, "direct_ids": self.direct,
            "tuned": sorted(c for c in self.caps if self.plans[c].tuned),
            "sources": {c: self.plans[c].source for c in self.caps},
            "prepare_s": round(t_prep, 1), "total_s": round(time.time() - t0, 1),
            "peak_transient_mb": peak, "persistent_mb": round(persistent / 2**20, 1),
        }
        print(f"{_TAG} geometry ready {self.report}", flush=True)

    def _race_call(self, cap):
        B = _b12x()
        prep = B["prep"]
        dev, K, E, topk = self.device, self.K, self.E, self.topk

        def race(state):
            g = torch.Generator(device=dev)
            g.manual_seed(1234 + cap)
            sc = tuple(torch.empty(tuple(s.shape), dtype=s.dtype, device=dev) for s in state.scratch.scratch_specs())
            x = torch.randn(cap, K, device=dev, generator=g).to(torch.bfloat16)
            src = x.clone()
            ids = torch.rand(cap, E, device=dev, generator=g).argsort(dim=1)[:, :topk].to(torch.int32).contiguous()
            w = torch.softmax(torch.randn(cap, topk, device=dev, generator=g), dim=1).float()
            out = torch.empty(cap, K, dtype=torch.bfloat16, device=dev)
            b = state.bind(scratch=sc, a=x, topk_ids=ids, topk_weights=w, output=out)
            return prep.PreparedCall(run=lambda: state.run(b), output=out, produce=lambda: x.copy_(src),
                                     owners=(sc, x, src, ids, w, out, b))
        return race

    def _lookup(self, m):
        i = bisect.bisect_left(self.caps, m)
        return self.caps[i] if i < len(self.caps) else self.caps[-1]

    def cap_for(self, m):
        return self.table[m] if m <= EXACT_MAX else self._lookup(m)

    def _warm(self, experts):
        x = torch.zeros(self.chunk, self.K, dtype=torch.bfloat16, device=self.device)
        out = torch.empty_like(x)
        ids = torch.zeros(self.chunk, self.topk, dtype=torch.int32, device=self.device)
        w = torch.zeros(self.chunk, self.topk, dtype=torch.float32, device=self.device)
        for c in self.caps:
            m = min(c, self.chunk)
            self._run(c, x[:m], ids[:m], w[:m], out[:m], experts._impl)
            ep_remap(ids[:m], w[:m], self.ids_buf[:m], self.w_buf[:m], 0, self.E)
        torch.cuda.synchronize(self.device)
        del x, out, ids, w

    # ---------------------------------------------------------------------------------------
    # pre-quantized input (prefill SP)
    # ---------------------------------------------------------------------------------------
    def prequant_caps(self, min_rows):
        """Capacities a pre-quantized batch of min_rows .. chunk rows can land on."""
        if not self.ready or min_rows > self.chunk:
            return []
        return sorted({self.cap_for(min_rows)} | {c for c in self.caps if c >= min_rows})

    def prequant_views(self, m):
        """(q [m, K] uint8, scales [m, K/32] uint8): where the front-end of the plan that runs m
        rows reads its MXFP8 input rows. None when m needs more than one launch, the runtime lacks
        the pre-quantized patch, or the plan does not expose token-major input storage."""
        if _B.get("impl") is None or not self.ready or not 1 <= m <= self.chunk:
            return None
        cap = self.cap_for(m)
        v = self.pq_views.get(cap)
        if v is None:
            v = self.pq_views[cap] = self._bind_views(cap)
        if v is False:
            return None
        pin, psc = v
        kb = self.K // 32
        return pin[:m], psc[: m * kb].view(m, kb)

    def _bind_views(self, cap):
        e = self.plans[cap]
        st = e.state
        dev = self.device
        a = torch.empty((1, self.K), dtype=torch.bfloat16, device=dev)
        ids = torch.zeros((1, self.topk), dtype=torch.int32, device=dev)
        w = torch.zeros((1, self.topk), dtype=torch.float32, device=dev)
        out = torch.empty((1, self.K), dtype=torch.bfloat16, device=dev)
        kw = dict(scratch=e.views, a=a, topk_ids=ids, topk_weights=w, output=out,
                  fast_math=st.scratch.caps.w4a16_fast_math, experts=self.impl0, unit_scale_contract=False)
        if st.w4a16_launches is not None:
            kw["_w4a16_launches"] = st.w4a16_launches
        try:
            b = st.scratch.bind(**kw)
            pin, psc = getattr(b, "packed_input", None), getattr(b, "packed_input_scale", None)
            kb = self.K // 32
            ok = (isinstance(pin, torch.Tensor) and isinstance(psc, torch.Tensor)
                  and pin.dtype == torch.uint8 and psc.dtype == torch.uint8
                  and pin.dim() == 3 and pin.shape[0] == 1 and pin.shape[1] >= cap and pin.shape[2] == self.K
                  and pin.is_contiguous() and psc.is_contiguous() and psc.numel() >= cap * kb)
        except Exception as exc:  # noqa: BLE001
            print(f"{_TAG} WARNING: capacity {cap}: cannot bind the pre-quantized input views ({exc!r})",
                  flush=True)
            return False
        if not ok:
            print(f"{_TAG} WARNING: capacity {cap}: packed_input {getattr(pin, 'shape', None)} / scale "
                  f"{getattr(psc, 'shape', None)} are not token-major MXFP8 storage; no pre-quantized "
                  f"input at this capacity", flush=True)
            return False
        return pin[0], psc.view(-1)

    def forward_prequant(self, impl, m, topk_ids, topk_weights, out):
        """Routed MoE of m rows whose MXFP8 input is already in prequant_views(m)."""
        if _B.get("impl") is None:
            raise RuntimeError("DSV41_MOE_B12X_NEXT: pre-quantized input needs the patched b12x_next runtime")
        if not 1 <= m <= self.chunk:
            raise RuntimeError(f"DSV41_MOE_B12X_NEXT: pre-quantized input of {m} rows (one launch holds "
                               f"<= {self.chunk})")
        direct = (self.direct and topk_ids.dtype == torch.int32 and topk_weights.dtype == torch.float32
                  and topk_ids.is_contiguous() and topk_weights.is_contiguous())
        if direct:
            ids, w = topk_ids, topk_weights
        else:
            ids, w = self.ids_buf[:m], self.w_buf[:m]
            ep_remap(topk_ids, topk_weights, ids, w, self.offset, self.E)
        # [m, K] bf16 shape carrier: the pre-quantized front-end never reads it (no kernel, no bytes)
        a = torch.empty((m, self.K), dtype=torch.bfloat16, device=self.device)
        with _B["impl"].prequantized_input():
            self._run(self.cap_for(m), a, ids, w, out, impl)
        PREQUANT_STATS["calls"] += 1
        return out

    def _warm_prequant(self, experts):
        """Compile the pre-quantized front-end variant of every prefill capacity at load."""
        if _B.get("impl") is None:
            print(f"{_TAG} WARNING: DSV41_PREFILL_SP_FP8_MOE=1 but b12x_next lacks the prequant-input "
                  f"patch; prefill SP keeps the bf16 MoE gather", flush=True)
            return
        t0 = time.time()
        done = []
        for cap in self.prequant_caps(PREQUANT_MIN_ROWS):
            v = self.prequant_views(cap)
            if v is None:
                continue
            q, sf = v
            q.zero_()
            sf.zero_()
            ids = torch.zeros((cap, self.topk), dtype=torch.int32, device=self.device)
            w = torch.zeros((cap, self.topk), dtype=torch.float32, device=self.device)
            out = torch.empty((cap, self.K), dtype=torch.bfloat16, device=self.device)
            self.forward_prequant(experts._impl, cap, ids, w, out)
            done.append(cap)
            del ids, w, out
        torch.cuda.synchronize(self.device)
        PREQUANT_STATS["calls"] = 0
        PREQUANT_STATS["warm_caps"][self.key] = done
        print(f"{_TAG} pre-quantized prefill input ready at capacities {done} "
              f"({time.time() - t0:.1f} s, E={self.E} top-k {self.topk})", flush=True)

    def _run(self, cap, x, ids, w, out, impl):
        if x.shape[0] > cap:
            raise RuntimeError(f"DSV41_MOE_B12X_NEXT: {x.shape[0]} rows on the capacity-{cap} plan "
                               f"(chunk {self.chunk}, capacities {self.caps})")
        e = self.plans[cap]
        st = e.state
        kw = dict(scratch=e.views, a=x, topk_ids=ids, topk_weights=w, output=out,
                  fast_math=st.scratch.caps.w4a16_fast_math, experts=impl, unit_scale_contract=False)
        if st.w4a16_launches is not None:
            kw["_w4a16_launches"] = st.w4a16_launches
        b = dataclasses.replace(st.scratch.bind(**kw), compact_launches=st.compact_launches, plan=e.plan)
        _B["fm"].run(binding=b)

    def forward(self, impl, x, topk_ids, topk_weights, out):
        m_all = x.shape[0]
        # EP1: every expert is local, the ids need no translation, and b12x ignores ids outside
        # [0, E) (padding -1). Row slices of the router's contiguous [M, top-k] buffers stay
        # contiguous. Any other dtype / layout takes the remap kernel (it also casts).
        direct = (self.direct and topk_ids.dtype == torch.int32 and topk_weights.dtype == torch.float32
                  and topk_ids.is_contiguous() and topk_weights.is_contiguous())
        s = 0
        while s < m_all:
            e = min(m_all, s + self.chunk)
            m = e - s
            if direct:
                ids, w = topk_ids[s:e], topk_weights[s:e]
            else:
                ids = self.ids_buf[:m]
                w = self.w_buf[:m]
                ep_remap(topk_ids[s:e], topk_weights[s:e], ids, w, self.offset, self.E)
            self._run(self.cap_for(m), x[s:e], ids, w, out[s:e], impl)
            s = e
        return out


def select_chunk(limits):
    """``limits``: {prepared capacity: that plan's max tokens per launch}.

    Returns ``(chunk, kept capacities, dropped capacities)``. A capacity is kept only if its plan
    runs it in one launch (capacity <= its launch limit), and the chunk is the largest kept one, so
    every chunk ``forward`` issues has a prepared plan whose scratch holds it.
    """
    kept = sorted(c for c, lim in limits.items() if c <= lim)
    if not kept:
        _die(f"no prepared capacity fits b12x's per-launch limit ({limits})")
    return kept[-1], kept, sorted(set(limits) - set(kept))


_M64_STATE = []


def _m64_admitted():
    """True when the b12x_next runtime carries the compact-N64 M64 admission patch."""
    if not _M64_STATE:
        try:
            from b12x_next.moe.fused_moe import _tuning
            ok = hasattr(_tuning, "_compact_n64_tiles") and 64 in _tuning._compact_n64_tiles(
                type("Q", (), {"num_tokens": 1 << 20})())
        except Exception:  # noqa: BLE001
            ok = False
        if not ok:
            print(f"{_TAG} WARNING: b12x_next lacks the compact-N64 M64 patch; prefill stays on M16",
                  flush=True)
        _M64_STATE.append(ok)
    return _M64_STATE[0]


def _align(n, a=256):
    return (n + a - 1) // a * a


@dataclasses.dataclass
class _LayerState:
    geom: _Geometry
    experts: object          # PreparedExperts (owns the repacked checkpoint storage)
    impl: object             # experts._impl, what the binding consumes
    hidden: int


# --------------------------------------------------------------------------------------------
# engine hooks
# --------------------------------------------------------------------------------------------
def _check_loader():
    """The FusedMoE loader must narrow by moe_tp_size (576 at TP4/EP1) and stack [up; gate]."""
    mod = sys.modules.get("sglang.srt.layers.moe.fused_moe_triton.layer")
    if mod is None:
        import importlib
        mod = importlib.import_module("sglang.srt.layers.moe.fused_moe_triton.layer")
    w13 = inspect.getsource(mod.FusedMoE._load_w13)
    w2 = inspect.getsource(mod.FusedMoE._load_w2)
    for needle in ('getattr(self.quant_method, "load_up_proj_weight_first", False)',
                   "loaded_weight.shape[shard_dim] // self.moe_tp_size", "start = shard_size"):
        if needle not in w13:
            _die(f"FusedMoE._load_w13 drifted (missing {needle!r})")
    if "loaded_weight.shape[shard_dim] // self.moe_tp_size" not in w2:
        _die("FusedMoE._load_w2 drifted")


def install_method(module):
    """sglang.srt.layers.quantization.mxfp4_flashinfer_cutlass_moe"""
    if not ENABLED or _state["method_installed"]:
        return
    for other in ("DSV41_MOE_SM12X", "DSV41_MOE_PAD"):
        if _env(other, "0") not in ("0", "", "off", "false"):
            _die(f"{other} patches the same method; enable only one")
    cls = getattr(module, "Mxfp4FlashinferCutlassMoEMethod", None)
    if cls is None:
        _die("Mxfp4FlashinferCutlassMoEMethod is gone")
    for name in ("create_weights", "process_weights_after_loading", "create_moe_runner", "apply"):
        if not callable(getattr(cls, name, None)):
            _die(f"Mxfp4FlashinferCutlassMoEMethod.{name} is gone")
    cw = inspect.getsource(cls.create_weights)
    pw = inspect.getsource(cls.process_weights_after_loading)
    if "% 128" not in cw or "self._fp8.create_weights(" not in cw or "fp4_scale_dtype=torch.float8_e8m0fnu" not in cw:
        _die("Mxfp4FlashinferCutlassMoEMethod.create_weights drifted")
    if "block_scale_interleave" not in pw or "self._fp8.process_weights_after_loading(layer)" not in pw:
        _die("Mxfp4FlashinferCutlassMoEMethod.process_weights_after_loading drifted")
    if "fused_experts_none_to_flashinfer_mxfp4" not in inspect.getsource(module) and \
            "flashinfer_cutlass" not in inspect.getsource(cls.create_moe_runner):
        _die("create_moe_runner no longer builds the flashinfer_mxfp4 runner")
    _check_loader()
    _b12x()                                       # import + pin/drift checks at boot, not at first use
    orig_process = cls.process_weights_after_loading

    def create_weights(self, layer, num_experts, hidden_size, intermediate_size_per_partition,
                       params_dtype, **extra_weight_attrs):
        n = intermediate_size_per_partition
        if hidden_size % 128 != 0 or n % 64 != 0:
            raise ValueError(f"DSV41_MOE_B12X_NEXT: unsupported routed-expert shape hidden={hidden_size} "
                             f"intermediate_per_partition={n} (needs hidden % 128 == 0, n % 64 == 0)")
        if not self.load_up_proj_weight_first:
            _die("load_up_proj_weight_first is no longer True; w13 layout would be [gate; up]")
        self._fp8.create_weights(layer, num_experts, hidden_size, n, params_dtype,
                                 fp4_scale_dtype=torch.float8_e8m0fnu, **extra_weight_attrs)

    def process_weights_after_loading(self, layer):
        self._fp8.process_weights_after_loading(layer)
        if getattr(layer, "_mega_moe_weights_built", False):
            return
        _convert_layer(self, layer)

    cls.create_weights = create_weights
    cls.process_weights_after_loading = process_weights_after_loading
    cls._dsv41_b12x_next_orig_process = orig_process
    _state["method_installed"] = True
    print(f"{_TAG} armed: routed MoE on b12x_next {PINNED_COMMIT[:8]} (W4A8, in-place repack); "
          f"graph bs {GRAPH_BS}, rows {BLOCK}/{BLOCK + 1}, ladder {LADDER}, tune={TUNE}", flush=True)
    if DETERMINISTIC:
        print(f"{_TAG} deterministic reduction: plan table {PLAN_TABLE} (det_triton={DET_TRITON}), no "
              f"autotune race (heuristic plans elsewhere)", flush=True)
    rows = uncovered_rows()
    if rows:
        print(f"{_TAG} WARNING: CUDA_GRAPH_MAX_BS_DECODE={_MAX_BS} exceeds the adapter's graph bs list (max "
              f"{max(GRAPH_BS)}); decode/verify row counts {rows} (bs {max(GRAPH_BS) + 1}..{_MAX_BS} x "
              f"{BLOCK}/{BLOCK + 1}) have no exact plan and run on the next capacity up: correct, but slower. "
              f"Set DSV41_MOE_B12X_NEXT_GRAPH_BS to the engine's capture list to cover them.", flush=True)


def uncovered_rows():
    """Draft/verify row counts of decode batch sizes above the adapter's graph list, up to
    CUDA_GRAPH_MAX_BS_DECODE, that have no exact-M plan (the engine's own capture list is not known
    here, so every batch size in that range is named)."""
    top = max(GRAPH_BS)
    if _MAX_BS <= top:
        return []
    exact = {r * b for r in (BLOCK, BLOCK + 1) for b in GRAPH_BS if r * b <= EXACT_MAX}
    return sorted({r * b for r in (BLOCK, BLOCK + 1) for b in range(top + 1, _MAX_BS + 1)} - exact)


def _convert_layer(method, layer):
    B = _b12x()
    fm = B["fm"]
    w13, w2 = layer.w13_weight, layer.w2_weight
    s13, s2 = layer.w13_weight_scale_inv, layer.w2_weight_scale_inv
    for name, t in (("w13_weight_scale_inv", s13), ("w2_weight_scale_inv", s2)):
        if t.dtype != torch.float8_e8m0fnu:
            _die(f"{name} must stay native E8M0, got {t.dtype}")
    E, two_n, k_half = w13.shape
    n, k = two_n // 2, k_half * 2
    if tuple(w2.shape) != (E, k, n // 2) or tuple(s13.shape) != (E, two_n, k // 32) \
            or tuple(s2.shape) != (E, k, n // 32):
        _die(f"unexpected expert tensor shapes w13 {tuple(w13.shape)} w2 {tuple(w2.shape)} "
             f"s13 {tuple(s13.shape)} s2 {tuple(s2.shape)}")
    if int(getattr(layer, "num_fused_shared_experts", 0) or 0):
        _die("fused shared experts are not supported")
    for t in (w13, w2, s13, s2):
        if not t.is_contiguous():
            _die("expert tensors must be contiguous")
    cfg = method.moe_runner_config if hasattr(method, "moe_runner_config") else layer.moe_runner_config
    limit = getattr(cfg, "swiglu_limit", None)
    if getattr(cfg, "gemm1_clamp_limit", None) is not None or getattr(cfg, "gemm1_alpha", None) is not None:
        _die("gemm1 clamp/alpha variants are not supported")
    topk = int(layer.top_k)
    ep_rank, ep_size = int(layer.moe_ep_rank), int(layer.moe_ep_size)
    dev = w13.device
    key = (E, k, n, topk, ep_rank, ep_size, limit, dev.index)
    geom = _GEOMS.get(key)
    new_geom = geom is None
    if new_geom:
        geom = _Geometry(key, E, k, n, topk, ep_rank, limit, dev, ep_size=ep_size)
        _GEOMS[key] = geom
    if not _state["ep_logged"]:
        _state["ep_logged"] = True
        print(f"{_TAG} INFO: routed MoE at EP_SIZE={ep_size} (this rank: EP rank {ep_rank}, {E} experts x "
              f"N={n}){'' if ep_size == 1 else '; the production profile is EP_SIZE=1, EP>1 is an A/B setup'}",
              flush=True)
    base = _peak_begin(dev)
    ones = torch.ones(E, dtype=torch.float32, device=dev)
    ptrs = tuple(t.untyped_storage().data_ptr() for t in (w13, w2, s13, s2))
    experts = fm.prepare_weights(plan=geom.weight_plan, weights=fm.PackedWeights(
        w13=w13.data.view(torch.uint8), w2=w2.data.view(torch.uint8),
        w13_block_scales=s13.data.view(torch.uint8), w2_block_scales=s2.data.view(torch.uint8),
        w13_global_scales=ones, w2_global_scales=ones))
    try:
        rep = experts._impl.representation_for("w4a8_mx")
        now = tuple(t.untyped_storage().data_ptr() for t in (rep.w13_rp, rep.w2_rp, rep.w13_sfb, rep.w2_sfb))
    except Exception as exc:  # noqa: BLE001
        _die(f"cannot inspect the prepared W4A8 representation ({exc!r})")
    peak = _peak_mb(dev, base)
    if now != ptrs:
        # a second allocation next to the checkpoint tensors would double the expert memory per rank
        _die(f"b12x_next did not repack E={E} K={k} N={n} in place")
    if peak is not None:
        _PREP_PEAK_MB.append(peak)
    if new_geom:
        geom.build(experts)
    st = _LayerState(geom=geom, experts=experts, impl=experts._impl, hidden=k)
    _LAYERS[w13.data_ptr()] = st
    layer._dsv4_mxfp4_backend = "b12x_next"
    layer._dsv41_b12x_next = st
    torch.cuda.empty_cache()
    if new_geom or len(_LAYERS) <= 1:
        print(f"{_TAG} layer ready E={E} K={k} N={n} topk={topk} ep={ep_rank}/{ep_size} "
              f"in_place=True prepare_peak_mb={peak}", flush=True)


def install_runner(module):
    """sglang.srt.layers.moe.moe_runner.flashinfer_cutlass (after its fused funcs registered)."""
    if not ENABLED or _state["runner_installed"]:
        return
    _b12x()                                       # pin/drift checks whichever hook runs first
    from sglang.srt.layers.moe.moe_runner.base import FusedOpPool
    orig = FusedOpPool._fused_funcs.get(_FUSED_KEY)
    if orig is None or getattr(orig, "__name__", "") != "fused_experts_none_to_flashinfer_mxfp4":
        _die(f"FusedOpPool{_FUSED_KEY} is {orig!r}, not the stock flashinfer_mxfp4 dispatcher")
    try:
        from sglang.srt.distributed import get_tp_group
        from sglang.srt.distributed.device_communicators.pynccl_allocator import use_symmetric_memory
        from sglang.srt.layers.dp_attention import is_allocation_symmetric
        from sglang.srt.layers.moe.token_dispatcher.standard import StandardCombineInput
        from sglang.srt.layers.moe.topk import TopKOutputChecker
    except Exception as exc:
        _die(f"engine helper import failed: {exc!r}")
    _ENG.update(get_tp_group=get_tp_group, use_symmetric_memory=use_symmetric_memory,
              is_allocation_symmetric=is_allocation_symmetric,
              StandardCombineInput=StandardCombineInput, TopKOutputChecker=TopKOutputChecker)
    _state["orig_fused"] = orig
    FusedOpPool._fused_funcs[_FUSED_KEY] = _fused_b12x_next
    _state["runner_installed"] = True
    print(f"{_TAG} FusedOpPool{_FUSED_KEY} -> b12x_next", flush=True)


def _alloc_out(m, k, dev):
    # Same allocation as the stock path: the output feeds the TP all-reduce.
    if _ENG["is_allocation_symmetric"]():
        with _ENG["use_symmetric_memory"](_ENG["get_tp_group"]()):
            return torch.empty(m, k, dtype=torch.bfloat16, device=dev)
    return torch.empty(m, k, dtype=torch.bfloat16, device=dev)


def _fused_b12x_next(dispatch_output, quant_info, runner_config):
    st = _LAYERS.get(quant_info.w13_weight.data_ptr())
    if st is None:
        # process_weights_after_loading is patched class-wide, so this layer's weights are in
        # checkpoint layout (not FlashInfer-interleaved): the stock kernel would return garbage.
        _die(f"a flashinfer_mxfp4 MoE layer reached the fused function without a b12x_next conversion "
             f"(w13 {tuple(quant_info.w13_weight.shape)}; e.g. _mega_moe_weights_built skipped it)")
    if getattr(quant_info, "padded_hidden", None) not in (None, st.hidden):
        raise RuntimeError("DSV41_MOE_B12X_NEXT: padded hidden size is not supported")
    x = dispatch_output.hidden_states
    topk = dispatch_output.topk_output
    if _ENG["TopKOutputChecker"].format_is_bypassed(topk):
        topk = topk.to_standard()
    out = _alloc_out(x.shape[0], x.shape[1], x.device)
    pre = getattr(dispatch_output, "hidden_states_pre_quant", None)
    rows = getattr(pre, "_dsv41_b12x_rows", None) if pre is not None else None
    if rows is not None:
        # prefill SP wrote this chunk's MXFP8 rows into the plan's input storage; x is a
        # shape-only placeholder. Any mismatch would compute on stale bytes: refuse.
        m = x.shape[0]
        views = st.geom.prequant_views(m)
        if (views is None or rows[0] is not st.geom or rows[1] != m
                or rows[2] != views[0].data_ptr() or rows[3] != views[1].data_ptr()):
            raise RuntimeError(f"DSV41_MOE_B12X_NEXT: pre-quantized input marker {rows[1:]} does not "
                               f"match this layer's plan for {m} rows")
        st.geom.forward_prequant(st.impl, m, topk.topk_ids, topk.topk_weights, out)
        return _ENG["StandardCombineInput"](hidden_states=out)
    if x.shape[0]:
        st.geom.forward(st.impl, x, topk.topk_ids, topk.topk_weights, out)
    return _ENG["StandardCombineInput"](hidden_states=out)


def prequant_target(layer, m):
    """For prefill SP: (marker, q [m, K] u8, scales [m, K/32] u8) of FusedMoE ``layer`` (the
    DeepseekV2MoE's ``experts``) when its routed MoE can take m pre-quantized rows, else None."""
    st = getattr(layer, "_dsv41_b12x_next", None)
    if st is None or not ENABLED:
        return None
    views = st.geom.prequant_views(m)
    if views is None:
        return None
    q, sf = views
    return (st.geom, m, q.data_ptr(), sf.data_ptr()), q, sf


def geometry_reports():
    return {"geometries": [g.report for g in _GEOMS.values()], "layers": len(_LAYERS),
            "prepare_weights_peak_mb_max": max(_PREP_PEAK_MB, default=0.0)}
