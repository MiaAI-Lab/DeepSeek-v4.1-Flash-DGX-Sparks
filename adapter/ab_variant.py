"""In-boot A/B of adapter flags: one CUDA graph set per variant, switched at runtime (TEST ONLY).

Off unless DSV41_AB_VARIANTS >= 2. Never set it in production: it roughly doubles decode graph
memory and adds a file poll to rank 0's scheduler loop.

  DSV41_AB_VARIANTS=2
  DSV41_AB_V0="DSV41_L2_PREFETCH_WOA=0"          spell the baseline out: empty = whatever the base env says
  DSV41_AB_V1="DSV41_L2_PREFETCH_WOA=1;DSV41_L2_PREFETCH_AHEAD=1"
  DSV41_AB_FILE=/state/ab_variant   (rank 0 reads "<variant> [token]"; writes <file>.ack JSON; a flag
                                     file older than the process is ignored, a stale ack is removed)
  DSV41_AB_MAX_BS=<n>               (optional: graph shapes above n are captured once, variant 0)
  DSV41_AB_ALLOW_AA=1               (allow variants whose effective configs are identical: A/A runs)

configure() refuses variants that resolve to the same effective config (after defaults and the
L2 main gate), unless DSV41_AB_ALLOW_AA=1: an override that repeats the base value silently
turns an intended A/B into an A/A. The config hash covers the variant specs, the pre-union base,
DSV41_AB_MAX_BS and the sha256 of ab_variant / l2_prefetch / fuse_quant / sitecustomize as found
on sys.path, so two nodes with different override files refuse to run together.

Flags. An adapter reads its gate through env(name) instead of os.environ. For the variant being
captured (or, at runtime, the variant being replayed) env() returns that variant's override, else
the ORIGINAL base value. Gates that an adapter or sitecustomize.py reads once at install time are
switched ON in os.environ for the whole process when any variant enables them (the "union"), so
the code is installed and then dispatches per call on env(). Keys outside the registry below are
refused: a flag that is not read through env() at call time would silently be the same in every
variant.

Graphs. FullCudaGraphBackend.capture_one (target verify and DSpark draft both use it) captures
each shape once per variant with that variant current; replay picks the set of the runtime
variant. Everything outside capture_one (static input buffers, attention metadata per bs, the
verify epilogue and draft sampler buffers) is shared by the variants.

Rank invariance. Every TP rank must replay the same variant on the same step. Only rank 0 reads
the flag file (stat-cached, DSV41_AB_POLL_S=0.5); on a change it appends a marker tuple to the
requests it pulled from the tokenizer, which SGLang broadcasts to every TP rank in the same
scheduler iteration. Every rank strips the marker from recv_requests' result and switches
before that iteration's batch runs. The scheduler is single-threaded, so the forward launched in
that iteration and all later ones see the new variant on every rank. Rank 0 then writes the ack.
"""
import contextlib
import hashlib
import importlib.abc
import importlib.machinery
import importlib.util
import json
import logging
import os
import re
import sys
import time

logger = logging.getLogger(__name__)

TAG = "__dsv41_ab_variant__"

# gates read at install time (by sitecustomize.py or an install_* function): unioned into os.environ
UNION_BOOL = ("DSV41_L2_PREFETCH", "DSV41_L2_PREFETCH_WOA", "DSV41_L2_PREFETCH_DRAFT",
              "DSV41_L2_PREFETCH_ENGRAM", "DSV41_L2_PREFETCH_LMHEAD")
UNION_PARTS = {"DSV41_FUSE_QUANT": ("qnorm", "wo_a", "hcpad"),
               # adapter/spec_sync_free.py: per-step rank-0 broadcasts kept / dropped per variant
               "DSV41_SPEC_SYNC_FREE": ("draft", "accept", "vcap", "merge"),
               # adapter/eager_glue.py: glue between the draft and verify graphs (vcap per capture)
               "DSV41_EAGER_GLUE": ("fence", "stage", "vcap", "vcapk", "tvglue", "dglue")}
# read through env() at call time only: never unioned
RUNTIME = ("DSV41_L2_PREFETCH_AHEAD", "DSV41_L2_PREFETCH_MB", "DSV41_L2_PREFETCH_WOA_MB",
           "DSV41_L2_PREFETCH_ENGRAM_MB", "DSV41_L2_PREFETCH_AG",
           # l2_prefetch v4 (plan-time, per variant)
           "DSV41_L2_PREFETCH_SKIP_N", "DSV41_L2_PREFETCH_WOB_MB", "DSV41_L2_PREFETCH_MOE",
           "DSV41_L2_PREFETCH_MOE_MB")
KNOWN = frozenset(UNION_BOOL) | frozenset(UNION_PARTS) | frozenset(RUNTIME)
# numeric knobs with their defaults (the name is historical: SKIP_N is a row count, not MB)
_MB_DEFAULTS = {"DSV41_L2_PREFETCH_MB": 6.0, "DSV41_L2_PREFETCH_WOA_MB": 6.0, "DSV41_L2_PREFETCH_ENGRAM_MB": 12.0,
                "DSV41_L2_PREFETCH_SKIP_N": 0.0, "DSV41_L2_PREFETCH_WOB_MB": 0.0, "DSV41_L2_PREFETCH_MOE_MB": 10.0}
HASHED_SOURCES = ("ab_variant", "l2_prefetch", "fuse_quant", "sitecustomize")

MODULES = ("sglang.srt.model_executor.runner_backend.full_cuda_graph_backend",
           "sglang.srt.managers.scheduler_components.request_receiver")

_OFF = ("", "0", "off", "false", "no")

ACTIVE = False
N = 1
CONFIG_HASH = ""
SOURCES = {}
_FILE = "/state/ab_variant"
_POLL_S = 0.5
_MAX_BS = None
_START_NS = 0
_REVERSE = False       # DSV41_AB_CAPTURE_ORDER=reverse: capture the sets last-to-first (diagnostic)
_DUMP_DIR = None       # DSV41_AB_DUMP_DIR + DSV41_AB_DUMP_SIZES=1,2: DOT dump of those shapes per set
_DUMP_SIZES = frozenset()
_specs = [{}]
_base = {}
_capture = None
_runtime = 0
_parts_cache = {}
_poll = {"checked": -1e9, "key": None, "file": None, "announced": None, "bad": None}
_state = {"seq": 0, "token": None, "leader": False, "dp_warned": False, "replays": {},
          "installed": set(), "ranks": None, "stale_checked": False}


def truthy(value):
    return value is not None and str(value).strip().lower() not in _OFF


def parse_spec(text):
    """'K=V K2=V2' (whitespace or ';' separated) -> {K: V}. Refuses unknown keys."""
    out = {}
    for item in re.split(r"[;\s]+", (text or "").strip()):
        if not item:
            continue
        if "=" not in item:
            raise RuntimeError(f"DSV41_AB: '{item}' is not KEY=VALUE")
        key, value = item.split("=", 1)
        key = key.strip()
        if key not in KNOWN:
            raise RuntimeError(f"DSV41_AB: {key} is not switchable in-boot; known: {sorted(KNOWN)}")
        out[key] = value.strip()
    return out


def _parse_parts(value, all_parts):
    raw = (value or "0").strip().lower()
    if raw in _OFF:
        return frozenset()
    if raw in ("1", "on", "true", "all"):
        return frozenset(all_parts)
    parts = frozenset(p.strip() for p in raw.split(",") if p.strip())
    bad = parts - set(all_parts)
    if bad:
        raise RuntimeError(f"DSV41_AB: unknown part(s) {sorted(bad)}; known {all_parts}")
    return parts


def _norm(cfg):
    """Effective behaviour of one variant's registry values (defaults applied; L2 knobs are moot
    while the L2 main gate is off)."""
    out = {}
    l2 = truthy(cfg.get("DSV41_L2_PREFETCH"))
    for key in sorted(KNOWN):
        value = cfg.get(key)
        if key.startswith("DSV41_L2_PREFETCH_") and not l2:
            out[key] = None
        elif key in UNION_PARTS:
            out[key] = sorted(_parse_parts(value, UNION_PARTS[key]))
        elif key in _MB_DEFAULTS:
            out[key] = float(value) if value not in (None, "") else _MB_DEFAULTS[key]
        elif key == "DSV41_L2_PREFETCH_AG":
            out[key] = value is None or value.strip() != "0"
        else:
            out[key] = truthy(value)
    return out


def _source_hashes():
    """sha256[:16] of the adapter files this harness relies on, as this process would import them."""
    out = {}
    for name in HASHED_SOURCES:
        path = __file__ if name == "ab_variant" else getattr(sys.modules.get(name), "__file__", None)
        if path is None:
            try:
                spec = importlib.util.find_spec(name)
                path = spec.origin if spec is not None else None
            except (ImportError, ValueError):
                path = None
        try:
            with open(path, "rb") as f:
                out[name] = hashlib.sha256(f.read()).hexdigest()[:16]
        except (OSError, TypeError):
            out[name] = "missing"
    return out


def configure(environ=None):
    """Parse DSV41_AB_*; returns True when armed. Idempotent across processes: the pre-union base
    values travel to child processes in DSV41_AB_BASE."""
    global ACTIVE, N, CONFIG_HASH, SOURCES, _specs, _base, _runtime, _capture, _FILE, _POLL_S, _MAX_BS, _START_NS
    global _REVERSE, _DUMP_DIR, _DUMP_SIZES
    environ = os.environ if environ is None else environ
    raw_n = environ.get("DSV41_AB_VARIANTS", "").strip()
    n = int(raw_n) if raw_n else 1
    if n < 2:
        ACTIVE, N, _specs, _base = False, 1, [{}], {}
        return False
    if n > 8:
        raise RuntimeError(f"DSV41_AB_VARIANTS={n}: at most 8 graph sets")
    extra = sorted(k for k in environ if re.fullmatch(r"DSV41_AB_V\d+", k) and int(k[10:]) >= n)
    if extra:
        raise RuntimeError(f"DSV41_AB: {extra} set but DSV41_AB_VARIANTS={n}")
    specs = [parse_spec(environ.get(f"DSV41_AB_V{i}", "")) for i in range(n)]
    base = (json.loads(environ["DSV41_AB_BASE"]) if "DSV41_AB_BASE" in environ
            else {k: environ.get(k) for k in sorted(KNOWN)})
    eff = [_norm({k: s.get(k, base.get(k)) for k in KNOWN}) for s in specs]
    same = [(i, j) for i in range(n) for j in range(i + 1, n) if eff[i] == eff[j]]
    if same and not truthy(environ.get("DSV41_AB_ALLOW_AA")):
        raise RuntimeError(f"DSV41_AB: variants {same} resolve to the same effective config (an A/A); spell "
                           "the baseline out in DSV41_AB_V0, or set DSV41_AB_ALLOW_AA=1 for a deliberate A/A")
    max_bs = environ.get("DSV41_AB_MAX_BS", "").strip()
    max_bs = int(max_bs) if max_bs else None
    poll_s = float(environ.get("DSV41_AB_POLL_S", "0.5"))
    flag = environ.get("DSV41_AB_FILE", "").strip() or "/state/ab_variant"
    reverse = environ.get("DSV41_AB_CAPTURE_ORDER", "").strip().lower() == "reverse"
    sources = _source_hashes()
    environ.setdefault("DSV41_AB_BASE", json.dumps(base, sort_keys=True))
    # union of the install-time gates over the base and every variant
    for key in UNION_BOOL:
        if any(truthy(s.get(key, base.get(key))) for s in specs):
            environ[key] = "1"
    for key, all_parts in UNION_PARTS.items():
        union = set()
        for s in specs:
            union |= _parse_parts(s.get(key, base.get(key)), all_parts)
        if union:
            environ[key] = "1" if union == set(all_parts) else ",".join(p for p in all_parts if p in union)
    ACTIVE, N, _specs, _base = True, n, specs, base
    _runtime, _capture = 0, None
    _FILE, _POLL_S, _MAX_BS, _START_NS, SOURCES = flag, poll_s, max_bs, time.time_ns(), sources
    _REVERSE = reverse
    _DUMP_DIR = environ.get("DSV41_AB_DUMP_DIR", "").strip() or None
    _DUMP_SIZES = frozenset(int(x) for x in environ.get("DSV41_AB_DUMP_SIZES", "1").split(",") if x.strip())
    _parts_cache.clear()
    CONFIG_HASH = hashlib.sha256(json.dumps([n, specs, base, max_bs, sources, reverse], sort_keys=True)
                                 .encode()).hexdigest()[:16]
    if not any(isinstance(f, _Finder) for f in sys.meta_path):
        sys.meta_path.insert(0, _Finder())
    return True


def describe():
    lines = [f"DSV41_AB armed (TEST ONLY): {N} variants, config {CONFIG_HASH}, sources {SOURCES}"]
    for i, s in enumerate(_specs):
        eff = {k: env_for(i, k) for k in sorted(KNOWN) if env_for(i, k) is not None}
        lines.append(f"  v{i}: overrides {s or '{}'} -> effective {eff}")
    return "\n".join(lines)


# -- flag reads --------------------------------------------------------------------------------

def current():
    return _capture if _capture is not None else _runtime


def env_for(variant, name, default=None):
    spec = _specs[variant]
    if name in spec:
        return spec[name]
    value = _base.get(name)
    return default if value is None else value


def env(name, default=None):
    """os.environ.get(name, default), except for registry keys while armed: the value of the
    variant being captured or replayed."""
    if not ACTIVE or name not in KNOWN:
        return os.environ.get(name, default)
    return env_for(current(), name, default)


def parts(name, all_parts):
    """Parts enabled for the current variant, for comma-list gates (DSV41_FUSE_QUANT)."""
    value = env(name, "0")
    key = (name, value)
    got = _parts_cache.get(key)
    if got is None:
        got = _parts_cache[key] = _parse_parts(value, all_parts)
    return got


def sig_tag():
    """() when off (callers' cache keys stay unchanged), else (current variant,)."""
    return (current(),) if ACTIVE else ()


@contextlib.contextmanager
def capturing(variant):
    global _capture
    prev, _capture = _capture, variant
    try:
        yield
    finally:
        _capture = prev


def set_runtime(variant):
    global _runtime
    _runtime = int(variant)


# -- graph backend -----------------------------------------------------------------------------

def _variants_for(shape_key):
    limit = _MAX_BS
    size = getattr(shape_key, "size", None)
    return N if limit is None or size is None or size <= limit else 1


def _kind(backend):
    runner = getattr(backend, "_cuda_graph_runner", None)
    mr = getattr(runner, "model_runner", None)
    return "draft" if getattr(mr, "is_draft_worker", False) else "target"


@contextlib.contextmanager
def _debug_graphs(enabled):
    """Diagnostic: every torch.cuda.CUDAGraph created inside is in debug mode (keeps its cudaGraph_t,
    so debug_dump can print it)."""
    if not enabled:
        yield
        return
    import torch
    base = torch.cuda.CUDAGraph

    class _DebugGraph(base):
        def __init__(self, *a, **k):
            k.setdefault("keep_graph", True)       # torch >= 2.9: keep the cudaGraph_t for debug_dump
            super().__init__(*a, **k)
            self.enable_debug_mode()

    torch.cuda.CUDAGraph = _DebugGraph
    try:
        yield
    finally:
        torch.cuda.CUDAGraph = base


def _dump(backend, shape_key, variant):
    import torch
    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    os.makedirs(_DUMP_DIR, exist_ok=True)
    path = os.path.join(_DUMP_DIR, f"{_kind(backend)}-size{shape_key.size}-set{variant}-rank{rank}.dot")
    graph = backend._graphs[shape_key]
    graph.debug_dump(path)
    graph.instantiate()                          # keep_graph=True defers it to here (or the first replay)
    logger.warning("DSV41_AB: dumped %s (%s)", path, "ok" if os.path.exists(path) else "NOT WRITTEN")


def _check_ranks(tp_group):
    """All TP ranks must capture the same variant list, or capture deadlocks / replays diverge."""
    group = getattr(tp_group, "cpu_group", None)
    if group is None or getattr(tp_group, "world_size", 1) <= 1:
        return
    import torch
    import torch.distributed as dist
    h = int(CONFIG_HASH, 16) & ((1 << 62) - 1)
    t = torch.tensor([h, -h], dtype=torch.int64)
    dist.all_reduce(t, op=dist.ReduceOp.MAX, group=group)
    if int(t[0]) != h or -int(t[1]) != h:
        raise RuntimeError("DSV41_AB: TP ranks disagree on DSV41_AB_* (config hash differs); "
                           "use the same env on every node")


def install_graph_backend(module):
    cls = module.FullCudaGraphBackend
    if getattr(cls, "_dsv41_ab", False):
        return
    cls._dsv41_ab = True
    orig_capture, orig_replay, orig_cleanup = cls.capture_one, cls.replay, cls.cleanup

    def _sets(self):
        sets = self.__dict__.get("_dsv41_ab_sets")
        if sets is None:
            _check_ranks(getattr(self, "_tp_group", None))
            sets = [(self._graphs, self._outputs)] + [({}, {}) for _ in range(1, N)]
            self._dsv41_ab_sets = sets
            self._dsv41_ab_shared = set()
        return sets

    def capture_one(self, shape_key, forward_fn, capture_inputs=None, post_warmup_hook=None):
        if not ACTIVE:
            return orig_capture(self, shape_key, forward_fn, capture_inputs, post_warmup_hook)
        sets = _sets(self)
        n = _variants_for(shape_key)
        order = list(range(n))[::-1] if _REVERSE and n == N else range(n)
        try:
            for i, v in enumerate(order):
                if i and post_warmup_hook is not None:
                    # The previous capture left the attention backend holding metadata built
                    # INSIDE that graph (never executed: garbage on the host side). Stock SGLang
                    # rebuilds it per shape outside capture_one; the warm-up reset restores the
                    # pre-capture metadata the same way it does between the two warm-ups.
                    post_warmup_hook()
                self._graphs, self._outputs = sets[v]
                dump = _DUMP_DIR is not None and getattr(shape_key, "size", None) in _DUMP_SIZES
                with capturing(v), _debug_graphs(dump):
                    orig_capture(self, shape_key, forward_fn, capture_inputs, post_warmup_hook)
                if dump:
                    _dump(self, shape_key, v)
        finally:
            self._graphs, self._outputs = sets[0]
        if n < N:
            self._dsv41_ab_shared.add(shape_key)

    def replay(self, shape_key, static_forward_batch, **kwargs):
        sets = self.__dict__.get("_dsv41_ab_sets")
        if sets is None:
            return orig_replay(self, shape_key, static_forward_batch, **kwargs)
        v = _runtime if shape_key in sets[_runtime][0] else 0
        counts = _state["replays"].setdefault(_kind(self), [0] * (N + 1))
        counts[v if v == _runtime else N] += 1
        if v == 0:
            return orig_replay(self, shape_key, static_forward_batch, **kwargs)
        self._graphs, self._outputs = sets[v]
        try:
            return orig_replay(self, shape_key, static_forward_batch, **kwargs)
        finally:
            self._graphs, self._outputs = sets[0]

    def cleanup(self):
        for graphs, outputs in self.__dict__.pop("_dsv41_ab_sets", None) or ():
            graphs.clear()
            outputs.clear()
        return orig_cleanup(self)

    cls.capture_one, cls.replay, cls.cleanup = capture_one, replay, cleanup
    logger.warning("DSV41_AB: FullCudaGraphBackend captures %d graph sets per shape%s (TEST ONLY)", N,
                   f" up to size {_MAX_BS}" if _MAX_BS is not None else "")


# -- rank-0 decision, broadcast with the pulled requests ---------------------------------------

def _file():
    return _FILE


def _drop_stale_ack():
    """Leader, first iteration: an ack left by an earlier boot must not satisfy a driver."""
    _state["stale_checked"] = True
    try:
        if os.stat(_FILE + ".ack").st_mtime_ns < _START_NS:
            os.unlink(_FILE + ".ack")
            logger.warning("DSV41_AB: removed a stale %s.ack from an earlier boot", _FILE)
    except OSError:
        pass


def _read_file():
    """Rank 0 only. Returns (variant, token) from the flag file, stat-cached; a file written
    before this process started (an earlier boot's) is ignored."""
    now = time.monotonic()
    if now - _poll["checked"] < _POLL_S:
        return _poll["file"]
    _poll["checked"] = now
    path = _FILE
    try:
        st = os.stat(path)
    except OSError:
        return _poll["file"]
    key = (st.st_mtime_ns, st.st_size, st.st_ino)
    if key == _poll["key"]:
        return _poll["file"]
    _poll["key"] = key
    if st.st_mtime_ns < _START_NS:
        if _poll["bad"] != key:
            _poll["bad"] = key
            logger.warning("DSV41_AB: ignoring %s: written before this boot; rewrite it", path)
        return _poll["file"]
    try:
        with open(path) as f:
            fields = f.read().split()
        v = int(fields[0])
        if not 0 <= v < N:
            raise ValueError(f"variant {v} not in [0, {N})")
        _poll["file"] = (v, fields[1] if len(fields) > 1 else None)
    except (OSError, ValueError, IndexError) as exc:
        if _poll["bad"] != key:
            _poll["bad"] = key
            logger.error("DSV41_AB: ignoring %s: %s", path, exc)
    return _poll["file"]


def _write_ack():
    path = _FILE + ".ack"
    rec = {"variant": _runtime, "token": _state["token"], "seq": _state["seq"], "time": time.time(),
           "config": CONFIG_HASH, "variants": N, "replays": _state["replays"], "ranks": _state["ranks"]}
    tmp = f"{path}.{os.getpid()}.tmp"
    try:
        with open(tmp, "w") as f:
            json.dump(rec, f)
        os.replace(tmp, path)
    except OSError as exc:
        logger.error("DSV41_AB: cannot write %s: %s", path, exc)


def _apply(marker):
    _, variant, token, config = marker
    if config != CONFIG_HASH:
        raise RuntimeError(f"DSV41_AB: rank 0 config {config} != this rank's {CONFIG_HASH}")
    prev = _runtime
    set_runtime(variant)
    _state["seq"] += 1
    _state["token"] = token
    logger.warning("DSV41_AB: variant %d -> %d (seq %d)", prev, variant, _state["seq"])


def _check_ranks_agree(group):
    """After a switch, every TP rank (all of them saw the same broadcast, so all get here in the
    same iteration) MAX-reduces [v, -v, seq, -seq]: max == min on both proves they agree."""
    if group is None:
        return None
    import torch
    import torch.distributed as dist
    if not dist.is_initialized() or dist.get_world_size(group) <= 1:
        return None
    t = torch.tensor([_runtime, -_runtime, _state["seq"], -_state["seq"]], dtype=torch.int64)
    dist.all_reduce(t, op=dist.ReduceOp.MAX, group=group)
    v_max, v_min, s_max, s_min = int(t[0]), -int(t[1]), int(t[2]), -int(t[3])
    return {"world": dist.get_world_size(group), "variant": [v_min, v_max], "seq": [s_min, s_max],
            "agree": v_min == v_max == _runtime and s_min == s_max == _state["seq"]}


def install_receiver(module):
    cls = module.SchedulerRequestReceiver
    if getattr(cls, "_dsv41_ab", False):
        return
    cls._dsv41_ab = True
    orig_pull, orig_recv = cls._pull_raw_reqs, cls.recv_requests

    def _pull_raw_reqs(self):
        reqs = orig_pull(self)
        if reqs is None or not ACTIVE:
            return reqs              # not the rank that reads the tokenizer socket
        _state["leader"] = True
        if not _state["stale_checked"]:
            _drop_stale_ack()
        try:
            from sglang.srt.runtime_context import get_parallel
            dp = get_parallel().enable_dp_attention
        except Exception:
            dp = False
        if dp or getattr(self.ps, "pp_size", 1) > 1:
            if not _state["dp_warned"]:
                _state["dp_warned"] = True
                logger.error("DSV41_AB: DP attention / PP not supported; staying on variant 0")
            return reqs
        want = _read_file()
        if want is not None and want != _poll["announced"]:
            _poll["announced"] = want
            reqs = list(reqs) + [(TAG, want[0], want[1], CONFIG_HASH)]
        return reqs

    def recv_requests(self, *args, **kwargs):
        reqs = orig_recv(self, *args, **kwargs)
        if not ACTIVE or not reqs:
            return reqs
        markers = [r for r in reqs if isinstance(r, tuple) and len(r) == 4 and r[0] == TAG]
        if not markers:
            return reqs
        for m in markers:
            _apply(m)
        _state["ranks"] = _check_ranks_agree(getattr(self, "tp_cpu_group", None))
        if _state["leader"]:
            _write_ack()
        if _state["ranks"] is not None and not _state["ranks"]["agree"]:
            raise RuntimeError(f"DSV41_AB: TP ranks disagree after a switch: {_state['ranks']}")
        return [r for r in reqs if not (isinstance(r, tuple) and len(r) == 4 and r[0] == TAG)]

    cls._pull_raw_reqs, cls.recv_requests = _pull_raw_reqs, recv_requests
    logger.warning("DSV41_AB: rank 0 polls %s; the switch rides the request broadcast", _FILE)


def install(module):
    name = module.__name__
    if name in _state["installed"]:
        return
    _state["installed"].add(name)
    if name.endswith("full_cuda_graph_backend"):
        install_graph_backend(module)
    elif name.endswith("request_receiver"):
        install_receiver(module)


class _Loader(importlib.abc.Loader):
    def __init__(self, original):
        self.original = original

    def create_module(self, spec):
        return self.original.create_module(spec)

    def exec_module(self, module):
        self.original.exec_module(module)
        install(module)


class _Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname not in MODULES:
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        if spec is not None:
            spec.loader = _Loader(spec.loader)
        return spec
