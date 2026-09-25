"""CPU checks for adapter/eager_glue.py (no GPU, no engine):

  PYTHONPATH=adapter python3 tests/test_eager_glue.py
  docker run --rm --network none -v <ds41>:/ds41:ro -e PYTHONPATH=/ds41/adapter \
      --entrypoint python3 <image> /ds41/tests/test_eager_glue.py

1. part parsing (all / 1 / lists, unknown parts refused) and per-variant dispatch under ab_variant;
2. fence: fused_clone returns fresh tensors with the same bytes, dtypes and shapes (int64 slices
   8-byte aligned whatever the order and bs), refuses what it cannot copy, and is off with the part;
3. stage: over a random walk of steps (same tensors, in-place edits, new tensors, bs changes,
   sampling_info None, foreign writes to the staged buffers) the cached wrapper leaves the staged
   buffers exactly as always-staging would, with draft_tau's multiply applied once per staging;
4. vcap: eligible graph sizes; the in-graph call writes what the eager call writes; the eager call
   is skipped only for a folded step at a captured, padding-free size of the replayed variant, and
   the skip is consumed once;
5. glue kind selection: only the device-only DSV4 preps qualify.
"""
import importlib
import os
import random
import sys
import types

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "adapter"))


def load(spec, vcap_spec="conf:0.1"):
    os.environ["DSV41_EAGER_GLUE"] = spec
    os.environ["DSV41_VERIFY_CAP"] = vcap_spec
    for m in ("eager_glue", "verify_cap"):
        sys.modules.pop(m, None)
    vc = importlib.import_module("verify_cap")
    eg = importlib.import_module("eager_glue")
    return eg, vc


def test_parse():
    eg, _ = load("all")
    assert eg.installed() == {"fence", "stage", "vcap", "vcapk"}          # tvglue / dglue dropped
    eg, _ = load("vcapk,dglue")
    assert eg.installed() == {"vcapk"}
    eg, _ = load("tvglue,dglue")
    assert eg.installed() == frozenset()
    eg, _ = load("fence,stage")
    assert eg.installed() == {"fence", "stage"} and eg.on("fence") and not eg.on("vcap")
    eg, _ = load("")
    assert eg.installed() == frozenset() and not eg.on("fence")
    try:
        load("fence,glue")
        eg.installed()
    except ValueError:
        pass
    else:
        raise AssertionError("unknown part accepted")


def test_ab_dispatch():
    eg, _ = load("fence,vcap")
    fake = types.SimpleNamespace(ACTIVE=True, _MAX_BS=4, cur=0)
    fake.current = lambda: fake.cur
    fake.parts = lambda name, all_parts: (frozenset({"fence"}) if fake.cur == 0 else frozenset({"vcap"}))
    eg._ab = fake
    assert eg.on("fence") and not eg.on("vcap")
    fake.cur = 1
    assert eg.on("vcap") and not eg.on("fence")
    assert eg._graph_variant(3) == 1 and eg._graph_variant(8) == 0   # above MAX_BS: set 0's graph
    assert eg._capture_tag() == 1


def test_fence():
    eg, _ = load("fence")
    for bs in (1, 2, 3, 5, 16, 64):
        g = torch.Generator().manual_seed(bs)
        bufs = {
            "correct_len": torch.randint(0, 6, (64,), generator=g),
            "bonus": torch.randint(0, 129280, (64,), generator=g),
            "cap_trim_lens": torch.randint(0, 6, (64,), generator=g, dtype=torch.int32),
            "commit_lens": torch.randint(1, 7, (64,), generator=g, dtype=torch.int32),
            "new_seq_lens": torch.randint(0, 1 << 40, (64,), generator=g),
            "out_tokens": torch.randint(-1, 129280, (64, 6), generator=g),
        }
        fields = list(bufs)
        random.Random(bs).shuffle(fields)                 # any order: int64 slices stay aligned
        src = [bufs[f][:bs] for f in fields]
        out = eg.fused_clone(src)
        assert out is not None
        base = out[0].untyped_storage().data_ptr()
        for s, o in zip(src, out):
            assert o.dtype == s.dtype and o.shape == s.shape and torch.equal(o, s)
            assert o.untyped_storage().data_ptr() == base                     # one fresh buffer
            assert o.untyped_storage().data_ptr() != s.untyped_storage().data_ptr()
            if o.dtype == torch.int64:
                assert o.data_ptr() % 8 == 0
        for s in src:                                     # later writes to the persistent buffers
            s.add_(1)                                     # do not reach the copies
        assert all(torch.equal(o, s - 1) for s, o in zip(src, out))
    assert eg.fused_clone([torch.zeros(4)]) is None                          # float
    assert eg.fused_clone([torch.zeros(4, 6, dtype=torch.int64)[:, 1]]) is None  # strided
    assert eg.fused_clone([torch.zeros(0, dtype=torch.int64)]) is None
    eg, _ = load("stage")
    assert eg.fused_clone([torch.zeros(4, dtype=torch.int64)]) is None      # part off


def _fence_module_roundtrip():
    """folded_result_fence with DSV41_EAGER_GLUE=fence returns the fused copies."""
    os.environ["DSV41_FOLDED_FENCE"] = "1"
    sys.modules.pop("folded_result_fence", None)
    ff = importlib.import_module("folded_result_fence")

    class AcceptOuts:
        def __init__(self, **kw):
            self.__dict__.update(kw)

    class TargetVerifyExecutor:
        def accept_and_finalize(self, *, folded_accept, bs):
            return self.outs

    mod = types.SimpleNamespace(TargetVerifyExecutor=TargetVerifyExecutor, AcceptOuts=AcceptOuts)
    ff.install(mod)
    ex = TargetVerifyExecutor()
    ex.outs = AcceptOuts(correct_len=torch.arange(3), bonus=torch.arange(3) + 5,
                         cap_trim_lens=torch.arange(3, dtype=torch.int32),
                         commit_lens=torch.ones(3, dtype=torch.int32),
                         new_seq_lens=torch.arange(3) + 100, out_tokens=torch.arange(18).view(3, 6))
    got = ex.accept_and_finalize(folded_accept=True, bs=3)
    for f in ff.FIELDS:
        a, b = getattr(got, f), getattr(ex.outs, f)
        assert torch.equal(a, b) and a.dtype == b.dtype and a.data_ptr() != b.data_ptr()
    return got


def test_fence_module():
    eg, _ = load("fence")
    before = eg._stats["fence"]
    got = _fence_module_roundtrip()
    assert eg._stats["fence"] == before + 1
    # one storage for all six: the single cat
    assert len({getattr(got, f).untyped_storage().data_ptr() for f in ("bonus", "commit_lens", "out_tokens")}) == 1


def make_sampler_module(calls):
    class DsparkDraftSampler:
        def __init__(self, max_bs=16):
            self.folded_sampling = True
            self.temperatures = torch.ones(max_bs)
            self.greedy_mask = torch.ones(max_bs, dtype=torch.bool)

        def stage_sampling_params(self, *, bs, sampling_info):        # the engine's body
            calls.append(bs)
            if sampling_info is None:
                self.temperatures[:bs].fill_(1.0)
                self.greedy_mask[:bs].fill_(True)
                return
            torch.clamp(sampling_info.temperatures.view(-1)[:bs].to(torch.float32), min=1e-5,
                        out=self.temperatures[:bs])
            self.greedy_mask[:bs].copy_((sampling_info.top_ks <= 1).view(-1)[:bs])

    return types.SimpleNamespace(DsparkDraftSampler=DsparkDraftSampler)


def tau_wrap(cls, tau=0.7):                               # adapter/draft_tau.py's wrapper
    orig = cls.stage_sampling_params

    def stage(self, *, bs, sampling_info):
        orig(self, bs=bs, sampling_info=sampling_info)
        if sampling_info is not None:
            self.temperatures[:bs].mul_(tau)

    cls.stage_sampling_params = stage


def test_stage():
    rng = random.Random(0)
    for trial in range(20):
        eg, _ = load("stage", vcap_spec="")
        calls_a, calls_b = [], []
        mod_a, mod_b = make_sampler_module(calls_a), make_sampler_module(calls_b)
        tau_wrap(mod_a.DsparkDraftSampler)
        tau_wrap(mod_b.DsparkDraftSampler)
        eg.install_sampler(mod_a)                          # mod_b: the stock, always-staging chain
        a, b = mod_a.DsparkDraftSampler(), mod_b.DsparkDraftSampler()
        info = None
        bs = 1
        for step in range(200):
            r = rng.random()
            if r < 0.05:
                info = None
            elif r < 0.12 or info is None:
                n = rng.randint(1, 16)
                info = types.SimpleNamespace(temperatures=torch.rand(n, 1) + 0.1,
                                             top_ks=torch.randint(0, 3, (n,)))
                bs = n
            elif r < 0.17:
                info.temperatures.mul_(0.5)                   # in place: version bump
            elif r < 0.20:
                info.top_ks[rng.randrange(info.top_ks.numel())] = rng.randint(0, 3)
            elif r < 0.23:
                a.temperatures.fill_(3.0)                      # a foreign write to the staged buffer
                b.temperatures.fill_(3.0)
            elif r < 0.26 and info is not None:
                bs = rng.randint(1, info.top_ks.numel())
            if info is None:
                bs = rng.randint(1, 16) if r < 0.05 else bs
            a.stage_sampling_params(bs=bs, sampling_info=info)
            b.stage_sampling_params(bs=bs, sampling_info=info)
            assert torch.equal(a.temperatures, b.temperatures), (trial, step)
            assert torch.equal(a.greedy_mask, b.greedy_mask), (trial, step)
        assert len(calls_a) < len(calls_b), (len(calls_a), len(calls_b))
    # a steady c1 decode stages once
    eg, _ = load("stage", vcap_spec="")
    calls = []
    mod = make_sampler_module(calls)
    tau_wrap(mod.DsparkDraftSampler)
    eg.install_sampler(mod)
    s = mod.DsparkDraftSampler()
    info = types.SimpleNamespace(temperatures=torch.full((1, 1), 0.6), top_ks=torch.ones(1, dtype=torch.int64))
    for _ in range(50):
        s.stage_sampling_params(bs=1, sampling_info=info)
    assert calls == [1] and abs(float(s.temperatures[0]) - 0.6 * 0.7) < 1e-7


class FakeGroup:
    world_size = 1

    def broadcast(self, t, src=0):
        return t


def make_vcap_modules(eg, vc, sizes):
    rc = types.ModuleType("sglang.srt.runtime_context")
    rc.get_exec = lambda: types.SimpleNamespace(graph=types.SimpleNamespace(
        cuda_graph_config=types.SimpleNamespace(decode=types.SimpleNamespace(bs=sizes))))
    for name in ("sglang", "sglang.srt"):
        sys.modules.setdefault(name, types.ModuleType(name))
    sys.modules["sglang.srt.runtime_context"] = rc
    vc._state["tp_group"] = FakeGroup()

    class DsparkDraftSampler:
        query_token_num = 6
        folded_sampling = False
        temperatures = None

        def __init__(self):
            self.confidence_out = torch.zeros(16, 5)

        def __call__(self, hidden_states, input_ids):
            bs = hidden_states.shape[0] // self.query_token_num
            self.confidence_out[:bs].copy_(hidden_states.view(bs, 6, -1)[:, 1:, 0])
            return "draft-out"

        def stage_sampling_params(self, *, bs, sampling_info):
            pass

    class DraftBlockProposer:
        def propose(self, *, bs, folded):
            return types.SimpleNamespace(folded=folded, confidence=torch.zeros(bs, 5))

    return (types.SimpleNamespace(DsparkDraftSampler=DsparkDraftSampler),
            types.SimpleNamespace(DraftBlockProposer=DraftBlockProposer))


def test_vcap():
    eg, vc = load("vcap")
    sizes = [1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16]
    smod, dmod = make_vcap_modules(eg, vc, sizes)
    assert eg._eligible_sizes() == frozenset(range(1, 9))
    eg.install_sampler(smod)
    eg.install_draft(dmod)
    assert vc.set_live_from_confidence is not eg._vcap["orig"]
    s = smod.DsparkDraftSampler()
    g = torch.Generator().manual_seed(1)
    for bs in (1, 3, 8, 10):
        conf = torch.rand(bs * 6, 1, generator=g)
        assert s(conf, None) == "draft-out"
        got = vc.live_buf()[:bs].clone() if bs in eg._eligible_sizes() else None
        # the eager reference on the same tensor
        vc.live_buf().fill_(99)
        eg._vcap["orig"](s.confidence_out[:bs], bs)
        ref = vc.live_buf()[:bs].clone()
        if got is not None:
            assert torch.equal(got, ref), (bs, got, ref)
            assert int(ref.min()) >= 2 and int(ref.max()) <= 6
    # not capturing on CPU: nothing registered, so nothing is ever skipped
    assert eg._vcap["captured"] == set()
    eg._vcap["captured"].update({(None, 1), (None, 3)})
    p = dmod.DraftBlockProposer()
    conf = torch.rand(3, 5)

    def verify_step(bs, folded):
        p.propose(bs=bs, folded=folded)
        vc.live_buf().fill_(99)
        before = eg._stats["vcap_skip"]
        vc.set_live_from_confidence(conf[:bs], bs)
        return eg._stats["vcap_skip"] > before, int(vc.live_buf()[0])

    assert verify_step(1, True) == (True, 99)            # skipped: the graph wrote it
    assert verify_step(3, True) == (True, 99)
    assert verify_step(2, True)[0] is False               # size not captured with vcap
    assert verify_step(1, False)[0] is False              # eager draft: eager live length
    # consumed once: a second verify without a new proposal runs eagerly
    p.propose(bs=1, folded=True)
    vc.set_live_from_confidence(conf[:1], 1)
    before = eg._stats["vcap_skip"]
    vc.set_live_from_confidence(conf[:1], 1)
    assert eg._stats["vcap_skip"] == before
    # the variant being replayed decides
    fake = types.SimpleNamespace(ACTIVE=True, _MAX_BS=None, cur=1)
    fake.current = lambda: fake.cur
    fake.parts = lambda name, all_parts: frozenset({"vcap"})
    eg._ab = fake
    assert verify_step(1, True)[0] is False               # (1, 1) was never captured
    eg._vcap["captured"].add((1, 1))
    assert verify_step(1, True)[0] is True
    fake.parts = lambda name, all_parts: frozenset()
    assert verify_step(1, True)[0] is False               # part off for the replayed variant
    eg._ab = None


def test_vcap_needs_conf_policy():
    eg, vc = load("vcap", vcap_spec="3")
    smod, _ = make_vcap_modules(eg, vc, [1, 2])
    eg.install_sampler(smod)
    assert not getattr(smod.DsparkDraftSampler, "_dsv41_eager_glue_vcap", False)


def test_glue_kind():
    eg, _ = load("tvglue,dglue")

    class DeepseekV4AttnBackend:
        def __init__(self, **kw):
            self.online_c128_mtp = types.SimpleNamespace(enabled=lambda: kw.get("online", False))
            self.needs_cpu_seq_lens = kw.get("cpu", False)
            self.is_dspark_draft = kw.get("draft", False)

        def _move_to_device(self, x):
            return x

    class Other(DeepseekV4AttnBackend):
        pass

    def runner(be, draft=False, dspark=True, ragged=False):
        algo = types.SimpleNamespace(is_dspark=lambda: dspark)
        mr = types.SimpleNamespace(spec_algorithm=algo, attn_backend=be, is_draft_worker=draft)
        return types.SimpleNamespace(model_runner=mr, ragged_verify_mode=ragged, enable_two_batch_overlap=False)

    assert eg._glue_kind(runner(DeepseekV4AttnBackend())) == "tvglue"
    assert eg._glue_kind(runner(DeepseekV4AttnBackend(draft=True), draft=True)) == "dglue"
    assert eg._glue_kind(runner(DeepseekV4AttnBackend(cpu=True))) is None
    assert eg._glue_kind(runner(DeepseekV4AttnBackend(online=True))) is None
    assert eg._glue_kind(runner(DeepseekV4AttnBackend(), ragged=True)) is None
    assert eg._glue_kind(runner(DeepseekV4AttnBackend(), dspark=False)) is None
    assert eg._glue_kind(runner(Other())) is None
    assert eg._glue_kind(runner(DeepseekV4AttnBackend(), draft=True)) is None   # draft backend flag unset


def test_glue_dropped():
    eg, _ = load("all")
    mod = types.SimpleNamespace(DecodeCudaGraphRunner=type("R", (), {}), MetadataGlueGraph=object)
    eg.install_runner(mod)
    assert not getattr(mod.DecodeCudaGraphRunner, "_dsv41_eager_glue", False)


def test_move_cache():
    eg, _ = load("dglue")
    made = []

    class B:
        def _move_to_device(self, x):
            t = torch.tensor(x, dtype=torch.int32)
            made.append(t)
            return t

    b = B()
    mc = eg._MoveCache(b)
    with mc:
        x1 = b._move_to_device([5, 5, 5])
        x2 = b._move_to_device([5, 5, 5])
        y = b._move_to_device([1, 2])                       # not uniform: not cached
        y2 = b._move_to_device([1, 2])
    assert x1 is x2 and y is not y2 and len(made) == 3
    assert torch.equal(x1, torch.tensor([5, 5, 5], dtype=torch.int32))
    assert b._move_to_device([5, 5, 5]) is not x1          # restored outside the glue


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print("PASS", t.__name__, flush=True)
    print(f"{len(tests)}/{len(tests)} PASS")
