"""Eager GPU work between the DSpark draft and target-verify CUDA graphs, cut down. Default OFF.

At c1 the host runs about one step (~35 ms) ahead of the GPU (overlap scheduler), so nothing the
host does between the two graph replays stalls the GPU. What costs is the GPU side of the eager
glue itself: ~54 tiny kernels / memcpys per step on the forward stream, each paying ~2.7 us of
dispatch gap on top of its ~1.3 us run (in-graph kernels: ~0.5 us gap). Profile 2026-09-25, all four
ranks alike: ~85 us draft -> verify + ~145 us verify -> draft = ~0.23 ms/step
(diagnostics/dsv41-eager-glue/RESULTS.md). Every part below keeps the outputs bit-identical.

DSV41_EAGER_GLUE is a comma list of parts (1 / on / all = every part that is not dropped):
  fence   adapter/folded_result_fence.py: the six clones of the folded verify results become ONE
          copy kernel (torch.cat of their int32 views into one fresh buffer, sliced back into
          tensors of the same dtype and shape). Same bytes, still off the persistent buffers.
  stage   DsparkDraftSampler.stage_sampling_params (+ draft_tau's multiply) is skipped while its
          inputs are unchanged: the same sampling_info temperature / top_k tensor objects at the
          same in-place version, the same bs, and the staged buffers untouched since.
  vcap    verify_cap's live-length update (DSV41_VERIFY_CAP=conf:T: cumprod, threshold, count,
          copy into the live buffer, rank-0 broadcast) is captured at the end of the draft graph,
          on the same confidence tensor, instead of running eagerly before the verify. Only for
          graph sizes no smaller batch is padded into (so the graph and the eager code write the
          same rows); the eager call is skipped only on steps whose draft replayed such a graph.
  vcapk   the same live-length update as ONE Triton kernel instead of ~10 torch ops (in the draft
          graph with vcap, eagerly otherwise). Bit-identical to torch: the running product is
          formed in the association torch's cumprod uses (sequential for one row, which goes
          through cub; the Sklansky tree of scan_innermost_dim for more rows), then the same fp32
          threshold compares, leading-run count and clamp. Checked against the torch ops on the
          first DSV41_EAGER_GLUE_CHECKS eager calls; any difference turns it off for good.
  tvglue, dglue: DROPPED after the 2026-09-25 fleet boot (see DROPPED below); the text is history.
  tvglue  the target-verify replay prep (DSV4 init_forward_metadata_out_graph) goes through
          SGLang's own MetadataGlueGraph (SGLANG_ENABLE_METADATA_GLUE_GRAPH), which SGLang forces
          off for every DFlash-family algorithm. Only when the prep is device-only here: DSV4
          backend, needs_cpu_seq_lens False (DSpark target), static verify layout, online c128 off.
  dglue   the same for the DSpark draft's replay prep. Its one host-fed input is the uniform
          extend length list [gamma] * bs, uploaded per step: served from a cached device tensor
          while the glue runs (same values), so the captured graph reads nothing from the host.
          Both glues compare the glued metadata with a fresh eager prep, bit for bit, on the first
          DSV41_EAGER_GLUE_CHECKS (8) replays of every key, and fall back to eager for good on
          any difference.
In-boot A/B (adapter/ab_variant.py, test only): DSV41_EAGER_GLUE is a union-parts gate. The code
of every part in the union is installed; fence / stage / tvglue / dglue dispatch per call on the
replayed variant, vcap per capture (each variant's draft graphs contain it or not) and its eager
skip per replayed variant.
"""
import logging
import os

import torch

logger = logging.getLogger(__name__)

PARTS = ("fence", "stage", "vcap", "vcapk", "tvglue", "dglue")
_OFF = ("", "0", "off", "false", "no")
CHECKS = int(os.environ.get("DSV41_EAGER_GLUE_CHECKS", "8") or 8)

try:  # DSV41_AB_VARIANTS in-boot A/B (test only): the env holds the union, each variant its own parts
    import ab_variant as _ab
except ImportError:
    _ab = None


# DROPPED 2026-09-25 after the fleet boot's CHECK FAILED (diagnostics/dsv41-eager-glue/RESULTS.md §6): the
# DSV4 replay prep reads forward_batch.out_cache_loc, a fresh tensor every step, not a runner static
# buffer, so a glue graph replays the capture-time address; a bounded bit-check cannot guard that. The
# names still parse (existing env strings keep working) but install nothing; "all" leaves them out.
DROPPED = frozenset({"tvglue", "dglue"})
_warned = set()


def _parse(value):
    raw = (value or "").strip().lower()
    if raw in _OFF:
        return frozenset()
    if raw in ("1", "on", "true", "all"):
        return frozenset(PARTS) - DROPPED
    parts = frozenset(p.strip() for p in raw.replace(";", ",").split(",") if p.strip())
    bad = parts - set(PARTS)
    if bad:
        raise ValueError(f"DSV41_EAGER_GLUE: unknown part(s) {sorted(bad)}; known {PARTS}")
    dropped = parts & DROPPED
    if dropped and not dropped <= _warned:
        _warned.update(dropped)
        print(f"[eager_glue] {sorted(dropped)} dropped (stale out_cache_loc pointer in a glue graph); ignored",
              flush=True)
    return parts - DROPPED


def installed():
    """Parts whose code is installed: os.environ, which holds the union under the A/B harness."""
    return _parse(os.environ.get("DSV41_EAGER_GLUE", ""))


def on(part):
    """Part enabled for the variant being captured or replayed (plain env when not under A/B)."""
    if _ab is not None and _ab.ACTIVE:
        return part in _ab.parts("DSV41_EAGER_GLUE", PARTS)
    return part in installed()


_stats = {"stage_hit": 0, "stage_miss": 0, "vcap_skip": 0, "vcap_eager": 0, "fence": 0}


def _tick(key, every=4096):
    _stats[key] += 1
    n = sum(_stats[k] for k in ("stage_hit", "stage_miss"))
    if key.startswith("stage") and n % every == 0:
        print(f"[eager_glue] stats {_stats}", flush=True)


# -------------------------------------------------------------------------------------- fence --
def fused_clone(tensors):
    """Fresh copies of small contiguous int32 / int64 tensors in ONE kernel, or None (caller clones).
    The int64 tensors go first so every int64 slice starts on an 8-byte boundary."""
    if not tensors or not on("fence"):
        return None
    dev = tensors[0].device
    for t in tensors:
        if (t.device != dev or t.dtype not in (torch.int32, torch.int64) or not t.is_contiguous()
                or t.numel() == 0):
            return None
    order = sorted(range(len(tensors)), key=lambda i: tensors[i].dtype != torch.int64)
    words = [tensors[i].reshape(-1).view(torch.int32) for i in order]
    flat = torch.cat(words)
    out = [None] * len(tensors)
    pos = 0
    for i, w in zip(order, words):
        t = tensors[i]
        seg = flat[pos:pos + w.numel()]
        out[i] = (seg.view(torch.int64) if t.dtype == torch.int64 else seg).view(t.shape)
        pos += w.numel()
    _stats["fence"] += 1
    return out


# -------------------------------------------------------------------------------------- stage --
def install_sampler(module):
    """sglang.srt.speculative.dspark_components.dspark_draft_sampler. Install AFTER draft_tau and
    spec_sync_free: this wrapper must be the outermost stage_sampling_params (a hit skips the whole
    chain, draft_tau's multiply included)."""
    parts = installed()
    cls = module.DsparkDraftSampler
    if "stage" in parts and not getattr(cls, "_dsv41_eager_glue_stage", False):
        cls._dsv41_eager_glue_stage = True
        orig_stage = cls.stage_sampling_params

        def stage_sampling_params(self, *, bs, sampling_info):
            if not on("stage") or not self.folded_sampling or self.temperatures is None:
                self.__dict__.pop("_dsv41_stage_key", None)
                return orig_stage(self, bs=bs, sampling_info=sampling_info)
            if sampling_info is None:
                refs = ()
                key = (bs, None)
            else:
                t, k = sampling_info.temperatures, sampling_info.top_ks
                refs = (t, k)   # held: keeps id() and data_ptr() of the keyed tensors unique
                key = (bs, id(t), t._version, t.data_ptr(), tuple(t.shape),
                       id(k), k._version, k.data_ptr(), tuple(k.shape))
            mine = (self.temperatures._version, self.greedy_mask._version)
            last = self.__dict__.get("_dsv41_stage_key")
            if last is not None and last[0] == key and last[1] == mine:
                _tick("stage_hit")
                ssf = __import__("sys").modules.get("spec_sync_free")
                if ssf is not None and getattr(ssf, "AUDIT", False):
                    ssf.audit_tick()        # its stage wrapper is skipped with the rest of the chain
                return None
            _tick("stage_miss")
            orig_stage(self, bs=bs, sampling_info=sampling_info)
            self._dsv41_stage_key = (key, (self.temperatures._version, self.greedy_mask._version), refs)
            return None

        cls.stage_sampling_params = stage_sampling_params
        print("[eager_glue] stage: draft sampling params staged only when their inputs change", flush=True)
    if parts & {"vcap", "vcapk"} and not getattr(cls, "_dsv41_eager_glue_vcap", False):
        _install_vcap_sampler(cls)


# --------------------------------------------------------------------------------------- vcap --
_vcap = {"captured": set(), "step": None, "orig": None, "eligible": None, "armed": False}


def _graph_variant(bs):
    """Which A/B graph set a replay at this size comes from (ab_variant.install_graph_backend)."""
    if _ab is None or not _ab.ACTIVE:
        return None
    limit = getattr(_ab, "_MAX_BS", None)
    return _ab.current() if limit is None or bs <= limit else 0


def _capture_tag():
    return _ab.current() if _ab is not None and _ab.ACTIVE else None


def _eligible_sizes():
    """Decode graph sizes no smaller batch is padded into: bs itself and bs - 1 are captured (or bs
    is the smallest). For those the graph's rows are exactly the eager code's rows."""
    got = _vcap["eligible"]
    if got is None:
        try:
            from sglang.srt.runtime_context import get_exec
            sizes = sorted(set(int(b) for b in get_exec().graph.cuda_graph_config.decode.bs))
        except Exception as exc:   # engine drift: vcap stays eager (safe), say so once
            print(f"[eager_glue] vcap: decode graph sizes unavailable ({exc!r}); vcap stays eager", flush=True)
            sizes = []
        got = frozenset(b for i, b in enumerate(sizes) if i == 0 or sizes[i - 1] == b - 1)
        _vcap["eligible"] = got
    return got


def _vcap_module():
    import sys
    vc = sys.modules.get("verify_cap")
    if vc is None:
        import verify_cap as vc
    return vc


try:
    import triton
    import triton.language as tl

    @triton.jit
    def _clamp01(x):
        return tl.where(x != x, x, tl.minimum(tl.maximum(x, 0.0), 1.0))     # NaN stays NaN, as torch

    @triton.jit
    def _thr_at(THR, thr, i: tl.constexpr, THR_T: tl.constexpr):
        if THR_T:
            return tl.load(THR + i)
        return thr

    @triton.jit
    def _vcap_kernel(CONF, stride, LIVE, THR, thr, bs, kmin, kmax,
                     ORDER: tl.constexpr, THR_T: tl.constexpr, BLOCK: tl.constexpr):
        """live[r] = clamp(#leading j with prod_{i<=j} clamp(conf[r, i], 0, 1) >= thr_j, kmin, kmax) + 1
        for 5 positions; ORDER 0 = ((((x0 x1) x2) x3) x4) (cub, one row), 1 = Sklansky (x0 x1),
        x2 (x0 x1), (x2 x3)(x0 x1), x4 [(x2 x3)(x0 x1)] (scan_innermost_dim)."""
        r = tl.arange(0, BLOCK)
        m = r < bs
        base = CONF + r * stride
        x0 = _clamp01(tl.load(base + 0, m, 0.0).to(tl.float32))
        x1 = _clamp01(tl.load(base + 1, m, 0.0).to(tl.float32))
        x2 = _clamp01(tl.load(base + 2, m, 0.0).to(tl.float32))
        x3 = _clamp01(tl.load(base + 3, m, 0.0).to(tl.float32))
        x4 = _clamp01(tl.load(base + 4, m, 0.0).to(tl.float32))
        c1 = x0 * x1
        if ORDER == 0:
            c2 = c1 * x2
            c3 = c2 * x3
        else:
            c2 = x2 * c1
            c3 = (x2 * x3) * c1
        c4 = x4 * c3
        run = (x0 >= _thr_at(THR, thr, 0, THR_T)).to(tl.int64)
        k = run
        run = run * (c1 >= _thr_at(THR, thr, 1, THR_T)).to(tl.int64)
        k += run
        run = run * (c2 >= _thr_at(THR, thr, 2, THR_T)).to(tl.int64)
        k += run
        run = run * (c3 >= _thr_at(THR, thr, 3, THR_T)).to(tl.int64)
        k += run
        run = run * (c4 >= _thr_at(THR, thr, 4, THR_T)).to(tl.int64)
        k += run
        k = tl.minimum(tl.maximum(k, kmin), kmax)
        tl.store(LIVE + r, k + 1, m)
except Exception:                                    # no triton: vcapk stays off
    _vcap_kernel = None

_vcapk = {"checks": 0, "off": False}


def _vcapk_update(vc, confidence, bs):
    """The fused live-length update; False when it does not apply (caller runs the torch ops)."""
    if (_vcap_kernel is None or _vcapk["off"] or not on("vcapk") or confidence is None
            or confidence.dim() != 2 or confidence.shape[1] != 5 or confidence.stride(1) != 1
            or not confidence.is_cuda or confidence.shape[0] != bs or bs > 1024 or vc.STRIDE != 6):
        return False
    capturing = torch.cuda.is_current_stream_capturing()
    if not capturing and _vcapk["checks"] < CHECKS:
        # reference first (it writes and shares the live buffer), then the kernel into a scratch row
        _vcap["orig"](confidence, bs)
        ref = vc.live_buf()[:bs].clone()
        got = torch.empty_like(ref)
        _launch_vcapk(vc, confidence, bs, got)
        _vcapk["checks"] += 1
        if not torch.equal(got, ref):
            _vcapk["off"] = True
            print(f"[eager_glue] vcapk: CHECK FAILED at bs={bs} ({got.tolist()} vs torch {ref.tolist()}); "
                  "torch ops for good", flush=True)
        return True
    buf = vc.live_buf(confidence.device)
    vc._state["last_conf"] = confidence
    _launch_vcapk(vc, confidence, bs, buf)
    vc.share_live(buf, bs)
    return True


def _launch_vcapk(vc, confidence, bs, out):
    thr = vc._state["thr"]
    thr_t = None
    if isinstance(thr, list):
        thr_t = vc._state.get("thr_t")
        if thr_t is None or thr_t.device != confidence.device:
            t = (thr + [thr[-1]] * 5)[:5]
            thr_t = vc._state["thr_t"] = torch.tensor(t, dtype=torch.float32, device=confidence.device)
    _vcap_kernel[(1,)](confidence, confidence.stride(0), out, thr_t if thr_t is not None else out,
                       0.0 if thr_t is not None else float(thr), bs, int(vc._state["kmin"]), vc.STRIDE - 1,
                       ORDER=0 if bs == 1 else 1, THR_T=thr_t is not None,
                       BLOCK=max(16, triton.next_power_of_2(bs)))


def _live_update(vc, confidence, bs):
    if not _vcapk_update(vc, confidence, bs):
        _vcap["orig"](confidence, bs)


def _install_vcap_sampler(cls):
    vc = _vcap_module()
    if not vc.ENABLED or vc._state.get("thr") is None:
        print("[eager_glue] vcap: DSV41_VERIFY_CAP is not a conf:T policy; nothing to capture", flush=True)
        return
    cls._dsv41_eager_glue_vcap = True
    if _vcap["orig"] is None:
        _vcap["orig"] = vc.set_live_from_confidence
    orig_live = _vcap["orig"]
    orig_call = cls.__call__

    def __call__(self, hidden_states, input_ids):
        out = orig_call(self, hidden_states, input_ids)
        if self.confidence_out is not None and on("vcap"):
            bs = hidden_states.shape[0] // self.query_token_num
            if bs in _eligible_sizes():
                # the eager call (or its fused twin), on the tensor the proposer hands the verify step
                _live_update(vc, self.confidence_out[:bs], bs)
                if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
                    _vcap["captured"].add((_capture_tag(), bs))
        return out

    cls.__call__ = __call__

    def set_live_from_confidence(confidence, bs):
        step, _vcap["step"] = _vcap["step"], None       # consumed once per verify
        if (step == (True, bs) and on("vcap") and (_graph_variant(bs), bs) in _vcap["captured"]
                and confidence is not None):
            vc._state["last_conf"] = confidence          # what the eager call would have kept
            _stats["vcap_skip"] += 1
            return None
        _stats["vcap_eager"] += 1
        if confidence is None:
            return orig_live(confidence, bs)
        return _live_update(vc, confidence, bs)

    vc.set_live_from_confidence = set_live_from_confidence
    _vcap["armed"] = True
    print("[eager_glue] vcap: verify_cap live length captured into the draft graph "
          "(padding-free sizes), eager call skipped on those replays; vcapk "
          f"{'fused kernel' if 'vcapk' in installed() else 'off'}", flush=True)


def install_draft(module):
    """sglang.srt.speculative.dspark_components.dspark_draft: note, per step, whether the proposal
    came out of a draft graph replay (folded) and at which bs."""
    if not installed() & {"vcap", "vcapk"}:
        return
    cls = module.DraftBlockProposer
    if getattr(cls, "_dsv41_eager_glue", False):
        return
    cls._dsv41_eager_glue = True
    orig = cls.propose

    def propose(self, *a, **kw):
        out = orig(self, *a, **kw)
        _vcap["step"] = (bool(out.folded and out.confidence is not None), kw.get("bs"))
        return out

    cls.propose = propose


# ------------------------------------------------------------------------------ tvglue / dglue --
def _tensors(obj, out, seen, depth=0):
    """Every tensor reachable from a metadata object (dataclass / msgspec / plain attributes)."""
    if depth > 6 or obj is None or isinstance(obj, (int, float, str, bool, bytes)):
        return
    if isinstance(obj, torch.Tensor):
        if id(obj) not in seen:
            seen.add(id(obj))
            out.append(obj)
        return
    if id(obj) in seen:
        return
    seen.add(id(obj))
    if isinstance(obj, (list, tuple)):
        for x in obj:
            _tensors(x, out, seen, depth + 1)
        return
    if isinstance(obj, dict):
        for x in obj.values():
            _tensors(x, out, seen, depth + 1)
        return
    names = list(getattr(obj, "__dict__", {}).keys()) + list(getattr(obj, "__struct_fields__", ()) or ())
    for n in names:
        try:
            _tensors(getattr(obj, n), out, seen, depth + 1)
        except Exception:
            pass


def _bits(t):
    t = t.detach()
    if t.dim() == 0:
        t = t.reshape(1)
    return t.contiguous().view(torch.uint8)


class _MoveCache:
    """dglue: DSV4 _move_to_device([gamma] * bs) served from a cached device tensor (same values)."""

    def __init__(self, backend):
        self.backend = backend
        self.cache = {}

    def __enter__(self):
        be = self.backend
        orig = be._move_to_device
        cache = self.cache

        def _move_to_device(x):
            key = tuple(int(v) for v in x)
            if len(key) <= 256 and len(set(key)) <= 1:
                got = cache.get(key)
                if got is None:
                    got = cache[key] = orig(list(key))
                return got
            return orig(x)

        be._move_to_device = _move_to_device
        self._orig = orig
        return self

    def __exit__(self, *exc):
        self.backend.__dict__.pop("_move_to_device", None)
        return False


def _make_glue(base_cls, device, kind):
    class CheckedGlue(base_cls):
        """SGLang's MetadataGlueGraph, per-call switchable, captured thread-local (a serving-time
        capture must not trip other threads' CUDA calls), and checked bit for bit against a fresh
        eager prep on its first CHECKS replays of every key."""

        def __init__(self, device):
            super().__init__(device)
            self.kind = kind
            self._move = None

        def _eager(self, attn_backend, fb_view):
            attn_backend.init_forward_metadata_out_graph(fb_view)

        def run(self, attn_backend, fb_view, key):
            if not on(self.kind):
                return self._eager(attn_backend, fb_view)
            if self.kind == "dglue":
                if self._move is None or self._move.backend is not attn_backend:
                    self._move = _MoveCache(attn_backend)
                with self._move:
                    return self._run(attn_backend, fb_view, key)
            return self._run(attn_backend, fb_view, key)

        def _run(self, attn_backend, fb_view, key):
            st = self._states.get(key)
            if st is None:
                st = self._states[key] = {"warmups": 0, "graph": None, "meta": None, "checks": 0}
            if st["graph"] is not None:
                for backend, metadata in st["meta"]:
                    backend.forward_metadata = metadata
                st["graph"].replay()
                if st["checks"] < CHECKS:
                    st["checks"] += 1
                    self._check(attn_backend, fb_view, key)
                return None
            if st["warmups"] < self.NUM_WARMUP:
                st["warmups"] += 1
                return self._eager(attn_backend, fb_view)
            if self._capture_stream is None:
                self._capture_stream = torch.cuda.Stream()
            graph = torch.cuda.CUDAGraph()
            try:
                with torch.cuda.graph(graph, stream=self._capture_stream, capture_error_mode="thread_local"):
                    self._eager(attn_backend, fb_view)
            except Exception as exc:
                print(f"[eager_glue] {self.kind}: capture failed for {key} ({exc!r}); eager prep for good",
                      flush=True)
                self.disabled = True
                return self._eager(attn_backend, fb_view)
            st["meta"] = [(b, b.forward_metadata) for b in self._leaves(attn_backend)]
            st["graph"] = graph
            graph.replay()
            st["checks"] = 1
            self._check(attn_backend, fb_view, key)
            if not self.disabled:
                print(f"[eager_glue] {self.kind}: replay prep captured for key {key}", flush=True)
            return None

        def _check(self, attn_backend, fb_view, key):
            """The glued state vs a fresh eager prep into the same persistent metadata."""
            meta = attn_backend.forward_metadata
            ts = []
            _tensors(meta, ts, set())
            glued = [(t, _bits(t).clone()) for t in ts]
            self._eager(attn_backend, fb_view)
            if attn_backend.forward_metadata is not meta:
                bad = "the eager prep installed another metadata object"
            else:
                ts2 = []
                _tensors(meta, ts2, set())
                bad = None
                if len(ts2) != len(glued):
                    bad = f"{len(ts2)} tensors after eager vs {len(glued)} glued"
                else:
                    for i, (t, g) in enumerate(glued):
                        if ts2[i] is not t or not torch.equal(_bits(t), g):
                            bad = f"tensor {i} {tuple(t.shape)} {t.dtype} differs"
                            break
            if bad is not None:
                print(f"[eager_glue] {self.kind}: CHECK FAILED for {key}: {bad}; eager prep for good",
                      flush=True)
                self.disabled = True

    return CheckedGlue(device)


def _glue_kind(runner):
    """'tvglue' / 'dglue' when this runner's replay prep is device-only (see module doc), else None."""
    mr = runner.model_runner
    algo = getattr(mr, "spec_algorithm", None)
    if algo is None or not algo.is_dspark():
        return None
    if getattr(runner, "ragged_verify_mode", False) or getattr(runner, "enable_two_batch_overlap", False):
        return None
    be = mr.attn_backend
    if type(be).__name__ != "DeepseekV4AttnBackend":
        return None
    mtp = getattr(be, "online_c128_mtp", None)
    if mtp is not None and mtp.enabled():
        return None
    if mr.is_draft_worker:
        return "dglue" if getattr(be, "is_dspark_draft", False) and hasattr(be, "_move_to_device") else None
    return "tvglue" if not getattr(be, "needs_cpu_seq_lens", True) else None


def install_runner(module):
    """sglang.srt.model_executor.runner.decode_cuda_graph_runner. tvglue / dglue are DROPPED (see DROPPED):
    installed() never returns them, so this installs nothing. The checked glue above is kept only for
    diagnostics/dsv41-eager-glue/glue_repro.py."""
    parts = installed() & {"tvglue", "dglue"}
    cls = module.DecodeCudaGraphRunner
    if not parts or getattr(cls, "_dsv41_eager_glue", False):
        return
    cls._dsv41_eager_glue = True
    base = module.MetadataGlueGraph
    orig_init = cls.__init__

    def __init__(self, *a, **kw):
        orig_init(self, *a, **kw)
        kind = _glue_kind(self)
        if kind in parts and self._metadata_glue is None:
            self._metadata_glue = _make_glue(base, self.device, kind)
            print(f"[eager_glue] {kind}: {'draft' if kind == 'dglue' else 'target verify'} replay prep "
                  "through the metadata glue graph", flush=True)
        elif kind is None and self.model_runner.spec_algorithm.is_dspark():
            print("[eager_glue] glue: this runner's replay prep is not provably device-only; kept eager",
                  flush=True)

    cls.__init__ = __init__
