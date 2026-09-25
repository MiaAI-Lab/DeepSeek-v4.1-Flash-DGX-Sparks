"""Pure-Python checks for adapter/ab_variant.py (no torch, no GPU; runs on the Mac too):

1. off by default: env() is os.environ.get, sig_tag() is (), nothing is unioned;
2. variant specs: union of install-time gates, per-variant values from the pre-union base,
   refusal of unknown keys / stray DSV41_AB_Vn, the base surviving a child re-configure;
3. graph backend: every shape captured once per variant with that variant current, the warm-up
   reset before each extra variant, replay of the runtime variant's set, DSV41_AB_MAX_BS sharing;
4. rank invariance: rank 0 alone reads the flag file (stat-cached) and appends a marker to the
   requests it pulled; every rank (two module instances here) switches on the same broadcast,
   the marker never reaches the scheduler, only the leader writes the ack, a config mismatch raises.
  python3 tests/test_ab_variant.py        (PYTHONPATH=adapter or run from the repo root)
"""
import importlib.util
import json
import os
import sys
import tempfile
import types

# TP4-line adapters: armed only under start-tp4.sh, which sets DSV41_LAUNCHER=tp4 in every rank.
os.environ["DSV41_LAUNCHER"] = "tp4"

ADAPTER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "adapter")
sys.path.insert(0, ADAPTER)
KEYS = [k for k in os.environ if k.startswith(("DSV41_AB", "DSV41_L2_PREFETCH", "DSV41_FUSE_QUANT"))]
for k in KEYS:
    del os.environ[k]


def fresh(name="ab_variant_t", environ=None):
    """A separate module instance = a separate rank's process state."""
    spec = importlib.util.spec_from_file_location(name, os.path.join(ADAPTER, "ab_variant.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def clean_meta_path():
    sys.meta_path[:] = [f for f in sys.meta_path if type(f).__name__ != "_Finder"]


def test_off_by_default():
    ab = fresh()
    env = {"DSV41_L2_PREFETCH": "1"}
    assert ab.configure(env) is False and not ab.ACTIVE
    assert env == {"DSV41_L2_PREFETCH": "1"}
    os.environ["DSV41_L2_PREFETCH_WOA"] = "1"
    try:
        assert ab.env("DSV41_L2_PREFETCH_WOA") == "1" and ab.env("NOPE", "d") == "d"
    finally:
        del os.environ["DSV41_L2_PREFETCH_WOA"]
    assert ab.sig_tag() == ()
    assert ab.configure({"DSV41_AB_VARIANTS": "1"}) is False


def test_specs_and_union():
    ab = fresh()
    env = {"DSV41_AB_VARIANTS": "2", "DSV41_L2_PREFETCH": "1", "DSV41_L2_PREFETCH_MB": "6",
           "DSV41_AB_V0": "", "DSV41_AB_V1": "DSV41_L2_PREFETCH_WOA=1;DSV41_L2_PREFETCH_ENGRAM=1 "
                                            "DSV41_L2_PREFETCH_MB=8 DSV41_FUSE_QUANT=qnorm,wo_a"}
    try:
        assert ab.configure(env) and ab.ACTIVE and ab.N == 2
        # install-time gates unioned on, runtime-only ones untouched
        assert env["DSV41_L2_PREFETCH_WOA"] == "1" and env["DSV41_L2_PREFETCH_ENGRAM"] == "1"
        assert env["DSV41_FUSE_QUANT"] == "qnorm,wo_a" and env["DSV41_L2_PREFETCH_MB"] == "6"
        assert "DSV41_L2_PREFETCH_AHEAD" not in env
        base = json.loads(env["DSV41_AB_BASE"])
        assert base["DSV41_L2_PREFETCH_WOA"] is None and base["DSV41_L2_PREFETCH"] == "1"
        # runtime variant 0: base values (pre-union), variant 1: overrides
        assert ab.current() == 0 and ab.sig_tag() == (0,)
        assert ab.env("DSV41_L2_PREFETCH_WOA", "0") == "0" and ab.env("DSV41_L2_PREFETCH_MB") == "6"
        assert ab.parts("DSV41_FUSE_QUANT", ("qnorm", "wo_a", "hcpad")) == frozenset()
        with ab.capturing(1):
            assert ab.current() == 1 and ab.sig_tag() == (1,)
            assert ab.env("DSV41_L2_PREFETCH_WOA") == "1" and ab.env("DSV41_L2_PREFETCH_MB") == "8"
            assert ab.parts("DSV41_FUSE_QUANT", ("qnorm", "wo_a", "hcpad")) == {"qnorm", "wo_a"}
        assert ab.current() == 0
        ab.set_runtime(1)
        assert ab.env("DSV41_L2_PREFETCH_ENGRAM") == "1"
        with ab.capturing(0):
            assert ab.env("DSV41_L2_PREFETCH_ENGRAM", "0") == "0"
        # keys outside the registry are never per-variant
        os.environ["DSV41_WO_A_W8"] = "1"
        assert ab.env("DSV41_WO_A_W8") == "1"
        del os.environ["DSV41_WO_A_W8"]
        # a child process inherits the unioned env + DSV41_AB_BASE: same variants, same hash
        h = ab.CONFIG_HASH
        child = fresh("ab_child")
        assert child.configure(dict(env)) and child.CONFIG_HASH == h
        assert child.env_for(0, "DSV41_L2_PREFETCH_WOA", "0") == "0"
        assert "DSV41_AB armed" in ab.describe() and "v1:" in ab.describe()
        # all parts -> "1"
        env2 = {"DSV41_AB_VARIANTS": "2", "DSV41_AB_V1": "DSV41_FUSE_QUANT=1"}
        assert fresh("ab2").configure(env2) and env2["DSV41_FUSE_QUANT"] == "1"
        # refusals
        for bad in ({"DSV41_AB_VARIANTS": "2", "DSV41_AB_V1": "DSV41_WO_A_W8=0"},
                    {"DSV41_AB_VARIANTS": "2", "DSV41_AB_V2": "DSV41_L2_PREFETCH_WOA=1"},
                    {"DSV41_AB_VARIANTS": "2", "DSV41_AB_V1": "DSV41_L2_PREFETCH_WOA"},
                    {"DSV41_AB_VARIANTS": "2", "DSV41_AB_V1": "DSV41_L2_PREFETCH_TILED=1"},
                    {"DSV41_AB_VARIANTS": "2", "DSV41_AB_V1": "DSV41_FUSE_QUANT=qnorm,bogus"},
                    {"DSV41_AB_VARIANTS": "9"}):
            try:
                fresh("ab_bad").configure(bad)
            except RuntimeError:
                continue
            raise AssertionError(f"accepted {bad}")
    finally:
        clean_meta_path()


class ShapeKey(tuple):
    @property
    def size(self):
        return self[0]


def fake_backend_module(log):
    class FullCudaGraphBackend:
        def __init__(self):
            self._graphs, self._outputs = {}, {}
            self._tp_group = types.SimpleNamespace(world_size=1)
            self._cuda_graph_runner = types.SimpleNamespace(model_runner=types.SimpleNamespace(is_draft_worker=False))

        def capture_one(self, shape_key, forward_fn, capture_inputs=None, post_warmup_hook=None):
            for _ in range(2):
                forward_fn()
                if post_warmup_hook:
                    post_warmup_hook()
            out = forward_fn()
            self._graphs[shape_key] = ("graph", out)
            self._outputs[shape_key] = out

        def can_run(self, fb, shape_key):
            return shape_key in self._graphs

        def replay(self, shape_key, static_forward_batch, **kwargs):
            log.append(("replay", self._graphs[shape_key][1]))
            return self._outputs[shape_key]

        def cleanup(self):
            self._graphs.clear()
            self._outputs.clear()

    return types.SimpleNamespace(__name__="sglang.srt.model_executor.runner_backend.full_cuda_graph_backend",
                                 FullCudaGraphBackend=FullCudaGraphBackend)


def test_graph_backend():
    ab = fresh()
    try:
        ab.configure({"DSV41_AB_VARIANTS": "2", "DSV41_L2_PREFETCH": "1", "DSV41_AB_V1": "DSV41_L2_PREFETCH_WOA=1"})
        log, hooks = [], []
        mod = fake_backend_module(log)
        ab.install(mod)
        ab.install(mod)                                    # idempotent
        be = mod.FullCudaGraphBackend()

        def fwd():
            return (ab.current(), ab.env("DSV41_L2_PREFETCH_WOA", "0"))

        for bs in (4, 2, 1):
            be.capture_one(ShapeKey((bs,)), fwd, None, lambda: hooks.append(ab.current()))
        # 2 warm-up resets per capture + 1 extra reset before variant 1, per shape
        assert len(hooks) == 3 * (2 + 2 + 1)
        assert set(be._graphs) == {ShapeKey((4,)), ShapeKey((2,)), ShapeKey((1,))}   # can_run = set 0
        assert be.can_run(None, ShapeKey((2,)))
        assert be.replay(ShapeKey((2,)), None) == (0, "0")
        ab.set_runtime(1)
        assert be.replay(ShapeKey((2,)), None) == (1, "1")
        assert be._graphs is be._dsv41_ab_sets[0][0]      # restored after the swap
        ab.set_runtime(0)
        assert be.replay(ShapeKey((4,)), None) == (0, "0")
        assert ab._state["replays"]["target"] == [2, 1, 0]

        # DSV41_AB_MAX_BS: bigger shapes captured once (variant 0) and shared
        ab._MAX_BS = 2                                     # parsed once by configure (DSV41_AB_MAX_BS)
        be2 = mod.FullCudaGraphBackend()
        for bs in (4, 2):
            be2.capture_one(ShapeKey((bs,)), fwd)
        ab._MAX_BS = None
        assert ShapeKey((4,)) not in be2._dsv41_ab_sets[1][0] and ShapeKey((2,)) in be2._dsv41_ab_sets[1][0]
        ab.set_runtime(1)
        assert be2.replay(ShapeKey((4,)), None) == (0, "0") and be2.replay(ShapeKey((2,)), None) == (1, "1")
        assert ab._state["replays"]["target"][2] == 1        # one shared-shape replay counted
        be2.cleanup()
        assert not be2._graphs and "_dsv41_ab_sets" not in be2.__dict__
        ab.set_runtime(0)
    finally:
        clean_meta_path()


def fake_receiver_module(pull_result, bus):
    """recv_requests = pull (rank 0 only) + broadcast; `bus` carries rank 0's list to the others."""
    class SchedulerRequestReceiver:
        def __init__(self, leader):
            self.leader = leader
            self.ps = types.SimpleNamespace(pp_size=1)

        def _pull_raw_reqs(self):
            return list(pull_result) if self.leader else None

        def recv_requests(self, local_reqs=None):
            reqs = self._pull_raw_reqs()
            if self.leader:
                bus[:] = [reqs]
                return reqs
            return list(bus[0])

    return types.SimpleNamespace(__name__="sglang.srt.managers.scheduler_components.request_receiver",
                                 SchedulerRequestReceiver=SchedulerRequestReceiver)


def put(path, text):
    """What the driver does: a fresh inode per write (os.replace), so the stat key always moves."""
    with open(path + ".tmp", "w") as f:
        f.write(text)
    os.replace(path + ".tmp", path)


def test_rank_invariant_switch():
    tmp = tempfile.mkdtemp()
    flag = os.path.join(tmp, "ab_variant")
    cfg = {"DSV41_AB_VARIANTS": "2", "DSV41_L2_PREFETCH": "1", "DSV41_AB_V1": "DSV41_L2_PREFETCH_AHEAD=1",
           "DSV41_AB_FILE": flag, "DSV41_AB_POLL_S": "0"}
    try:
        r0, r1 = fresh("rank0"), fresh("rank1")
        r0.configure(dict(cfg))
        r1.configure(dict(cfg))
        bus = []
        m0, m1 = fake_receiver_module(["req-a"], bus), fake_receiver_module(["req-a"], bus)
        r0.install(m0)
        r1.install(m1)
        lead, other = m0.SchedulerRequestReceiver(True), m1.SchedulerRequestReceiver(False)

        def iteration():
            a = lead.recv_requests()
            b = other.recv_requests()
            return a, b

        # no file: nothing broadcast, both on variant 0
        assert iteration() == (["req-a"], ["req-a"]) and r0._runtime == r1._runtime == 0
        assert not os.path.exists(flag + ".ack")
        put(flag, "1 tok1\n")
        a, b = iteration()
        assert a == ["req-a"] and b == ["req-a"]                     # marker stripped on every rank
        assert r0._runtime == r1._runtime == 1
        assert bus[0][-1][0] == r0.TAG                               # it rode rank 0's broadcast
        ack = json.load(open(flag + ".ack"))
        assert ack["variant"] == 1 and ack["token"] == "tok1" and ack["seq"] == 1
        assert r1._state["leader"] is False and r0._state["leader"] is True
        # unchanged file: no further marker
        iteration()
        assert r0._state["seq"] == r1._state["seq"] == 1
        # same variant, new token: re-ack (the driver's per-block counter snapshot)
        put(flag, "1 tok2\n")
        iteration()
        assert json.load(open(flag + ".ack"))["token"] == "tok2" and r1._state["seq"] == 2
        # invalid content is ignored, variant kept
        put(flag, "7 tok3\n")
        iteration()
        assert r0._runtime == r1._runtime == 1
        # stat cache: within DSV41_AB_POLL_S (parsed once) a change is not seen yet
        r0._POLL_S = 3600.0
        put(flag, "0 tok4 \n")
        iteration()
        assert r0._runtime == 1
        r0._poll["checked"] = -1e9
        iteration()
        assert r0._runtime == r1._runtime == 0
        r0._POLL_S = 0.0
        # a rank configured differently refuses instead of silently diverging
        r2 = fresh("rank2")
        r2.configure(dict(cfg, DSV41_AB_V1="DSV41_L2_PREFETCH_MB=8"))
        m2 = fake_receiver_module([], bus)
        r2.install(m2)
        put(flag, "1 tok5\n")
        lead.recv_requests()
        try:
            m2.SchedulerRequestReceiver(False).recv_requests()
        except RuntimeError as exc:
            assert "config" in str(exc)
        else:
            raise AssertionError("config mismatch accepted")
    finally:
        clean_meta_path()


def test_stale_files_from_an_earlier_boot():
    tmp = tempfile.mkdtemp()
    flag = os.path.join(tmp, "ab_variant")
    put(flag, "1 old\n")
    put(flag + ".ack", '{"token": "old", "variant": 1}')
    past = 1_000_000_000
    os.utime(flag, (past, past))
    os.utime(flag + ".ack", (past, past))
    try:
        r0 = fresh("rank0s")
        r0.configure({"DSV41_AB_VARIANTS": "2", "DSV41_L2_PREFETCH": "1", "DSV41_AB_V1": "DSV41_L2_PREFETCH_AHEAD=1",
                      "DSV41_AB_FILE": flag, "DSV41_AB_POLL_S": "0"})
        bus = []
        m0 = fake_receiver_module(["r"], bus)
        r0.install(m0)
        lead = m0.SchedulerRequestReceiver(True)
        assert lead.recv_requests() == ["r"] and r0._runtime == 0      # the old flag is ignored
        assert not os.path.exists(flag + ".ack")                       # the old ack is gone
        put(flag, "1 new\n")
        lead.recv_requests()
        assert r0._runtime == 1 and json.load(open(flag + ".ack"))["token"] == "new"
    finally:
        clean_meta_path()


def test_aa_guard_and_explicit_v0():
    base = {"DSV41_AB_VARIANTS": "2", "DSV41_L2_PREFETCH": "1", "DSV41_L2_PREFETCH_MB": "6"}
    try:
        refused = [
            dict(base, DSV41_AB_V1="DSV41_L2_PREFETCH_MB=6"),            # repeats the base value
            dict(base, DSV41_AB_V1="DSV41_L2_PREFETCH_MB=6.0"),          # same number
            dict(base, DSV41_L2_PREFETCH_WOA="1", DSV41_AB_V1="DSV41_L2_PREFETCH_WOA=1"),
            dict(base, DSV41_AB_V0="DSV41_L2_PREFETCH=0;DSV41_L2_PREFETCH_WOA=1",
                 DSV41_AB_V1="DSV41_L2_PREFETCH=0"),                      # L2 off: its sub-gates are moot
            dict(base, DSV41_AB_V1="DSV41_L2_PREFETCH_AG=1"),            # AG defaults to on
            dict(base, DSV41_AB_V1="DSV41_FUSE_QUANT=0"),
            {"DSV41_AB_VARIANTS": "3", "DSV41_AB_V1": "DSV41_FUSE_QUANT=1", "DSV41_AB_V2": "DSV41_FUSE_QUANT=all"},
        ]
        for env in refused:
            try:
                fresh("ab_aa").configure(dict(env))
            except RuntimeError as exc:
                assert "A/A" in str(exc)
                continue
            raise AssertionError(f"A/A accepted: {env}")
        assert fresh("ab_aa").configure(dict(base, DSV41_AB_V1="DSV41_L2_PREFETCH_MB=6", DSV41_AB_ALLOW_AA="1"))
        # an explicit baseline: base has WOA on, V0 turns it off, V1 keeps it
        ab = fresh("ab_v0")
        env = dict(base, DSV41_L2_PREFETCH_WOA="1", DSV41_AB_V0="DSV41_L2_PREFETCH_WOA=0",
                   DSV41_AB_V1="DSV41_L2_PREFETCH_WOA=1")
        assert ab.configure(env) and env["DSV41_L2_PREFETCH_WOA"] == "1"
        assert ab.env_for(0, "DSV41_L2_PREFETCH_WOA") == "0" and ab.env_for(1, "DSV41_L2_PREFETCH_WOA") == "1"
    finally:
        clean_meta_path()


def test_config_hash_inputs():
    env = {"DSV41_AB_VARIANTS": "2", "DSV41_L2_PREFETCH": "1", "DSV41_AB_V1": "DSV41_L2_PREFETCH_WOA=1"}
    try:
        a, b, c = fresh("h1"), fresh("h2"), fresh("h3")
        a.configure(dict(env))
        b.configure(dict(env))
        c.configure(dict(env, DSV41_AB_MAX_BS="4"))
        assert a.CONFIG_HASH == b.CONFIG_HASH != c.CONFIG_HASH and c._MAX_BS == 4
        assert set(a.SOURCES) == {"ab_variant", "l2_prefetch", "fuse_quant", "sitecustomize"}
        assert a.SOURCES["ab_variant"] != "missing" and a.SOURCES["l2_prefetch"] != "missing"
        # a different adapter file on one node changes that node's hash
        saved = sys.modules.get("fuse_quant")
        tmp = tempfile.mkdtemp()
        with open(os.path.join(tmp, "fuse_quant.py"), "w") as f:
            f.write("# another version\n")
        sys.modules["fuse_quant"] = types.SimpleNamespace(__file__=os.path.join(tmp, "fuse_quant.py"))
        try:
            d = fresh("h4")
            d.configure(dict(env))
            assert d.CONFIG_HASH != a.CONFIG_HASH
        finally:
            if saved is None:
                del sys.modules["fuse_quant"]
            else:
                sys.modules["fuse_quant"] = saved
    finally:
        clean_meta_path()


def test_autotune_fingerprint_is_the_base_config():
    import autotune_keep as ak
    keys = [k for k in os.environ if k.startswith(("DSV41_AB", "DSV41_L2_PREFETCH", "DSV41_FUSE_QUANT"))]
    assert not keys, keys
    os.environ["DSV41_L2_PREFETCH"] = "1"
    try:
        fp_base = ak.launch_fingerprint()
        os.environ.update({"DSV41_AB_VARIANTS": "2", "DSV41_AB_V1": "DSV41_L2_PREFETCH_WOA=1;DSV41_FUSE_QUANT=1"})
        ab = fresh("ab_fp")
        assert ab.configure() and os.environ["DSV41_L2_PREFETCH_WOA"] == "1"   # unioned into os.environ
        assert ak.launch_fingerprint() == fp_base
    finally:
        for k in [k for k in os.environ if k.startswith(("DSV41_AB", "DSV41_L2_PREFETCH", "DSV41_FUSE_QUANT"))]:
            del os.environ[k]
        clean_meta_path()


def test_finder_wraps_only_its_modules():
    ab = fresh()
    try:
        ab.configure({"DSV41_AB_VARIANTS": "2", "DSV41_AB_ALLOW_AA": "1"})
        finder = next(f for f in sys.meta_path if type(f).__name__ == "_Finder")
        assert finder.find_spec("json") is None
        assert sum(type(f).__name__ == "_Finder" for f in sys.meta_path) == 1
        ab.configure({"DSV41_AB_VARIANTS": "2", "DSV41_AB_ALLOW_AA": "1"})
        assert sum(type(f).__name__ == "_Finder" for f in sys.meta_path) == 1
    finally:
        clean_meta_path()


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("PASS", name, flush=True)
