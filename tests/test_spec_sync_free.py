"""CPU checks for adapter/spec_sync_free.py's sync routing (no GPU, no engine):

  docker run --rm --network none -v <ds41>:/ds41:ro -e PYTHONPATH=/ds41/adapter \
      --entrypoint python3 <image> /ds41/tests/test_spec_sync_free.py

1. token parsing: all / 1 / audit / merge, unknown tokens refused;
2. with draft,accept the draft and epilogue sites skip the broadcast, every other site keeps it;
3. merge: the epilogue's three syncs become ONE broadcast of the packed values, and each of the three
   tensors ends up with rank 0's values (a fake group plays a receiver rank);
4. audit: every broadcast is kept and a receiver whose value differed from rank 0's is counted per
   site; a rank that agreed counts checks only;
5. verify_cap routes its live-length broadcast through tp_broadcast (skip / keep / audit);
6. the source checks accept the engine's shapes and refuse drifted ones.
"""
import enum
import importlib
import os
import sys
import types

import torch


class Site(enum.IntEnum):
    DSPARK_MEM = 1
    DSPARK_DRAFT_GREEDY = 2
    DSPARK_DRAFT_SAMPLE = 3
    DSPARK_DRAFT_MULTINOMIAL = 4
    DSPARK_GRAPH_SAMPLE = 5
    DSPARK_GRAPH_GREEDY = 6
    DSPARK_PLAN = 7
    DSPARK_ACCEPT_GREEDY = 8
    DSPARK_ACCEPT_SAMPLE = 9
    DSPARK_ACCEPT_GRAPH = 10
    DSPARK_TARGET = 11


SpecTpSyncSite = Site


def use(_):
    pass


class FakeGroup:
    """A receiver rank: broadcast overwrites the tensor with 'rank 0's' value (value + delta)."""

    def __init__(self, delta=0):
        self.world_size, self.calls, self.delta = 4, [], delta

    def broadcast(self, t, src=0):
        assert src == 0
        self.calls.append(tuple(t.shape))
        if self.delta:
            t.add_(self.delta)
        return t


def make_sync_cls():
    class SpecTpSync:
        def __init__(self, group):
            self._tp_group = group
            self._sites = frozenset(Site)

        def sync(self, site, values):
            if site in self._sites:
                self._tp_group.broadcast(values, src=0)
            return values
    return SpecTpSync


def load(spec):
    os.environ["DSV41_SPEC_SYNC_FREE"] = spec
    sys.modules.pop("spec_sync_free", None)
    return importlib.import_module("spec_sync_free")


def test_parse():
    m = load("all")
    assert (m.SKIP_DRAFT, m.SKIP_ACCEPT, m.SKIP_VCAP, m.MERGE, m.AUDIT) == (True, True, True, False, False)
    m = load("1")
    assert (m.SKIP_DRAFT, m.SKIP_ACCEPT, m.SKIP_VCAP, m.MERGE) == (True, True, True, False)
    m = load("draft,accept")
    assert (m.SKIP_DRAFT, m.SKIP_ACCEPT, m.SKIP_VCAP, m.MERGE) == (True, True, False, False)
    m = load("audit,all")
    assert not (m.SKIP_DRAFT or m.SKIP_ACCEPT or m.SKIP_VCAP) and m.AUDIT and m.NOISE
    m = load("draft,merge")
    assert m.SKIP_DRAFT and m.MERGE and not m.SKIP_ACCEPT
    m = load("accept,merge")
    assert m.SKIP_ACCEPT and not m.MERGE
    try:
        load("drafts")
    except ValueError:
        pass
    else:
        raise AssertionError("unknown token accepted")
    m = load("")
    assert not m.ENABLED


def test_skip():
    m = load("draft,accept")
    cls = make_sync_cls()
    m.install_sync(cls, Site)
    g = FakeGroup()
    s = cls(g)
    for site in (Site.DSPARK_GRAPH_SAMPLE, Site.DSPARK_GRAPH_GREEDY, Site.DSPARK_ACCEPT_GRAPH):
        s.sync(site, torch.zeros(3, dtype=torch.int64))
    assert g.calls == [], g.calls
    kept = [Site.DSPARK_MEM, Site.DSPARK_DRAFT_SAMPLE, Site.DSPARK_ACCEPT_SAMPLE, Site.DSPARK_ACCEPT_GREEDY,
            Site.DSPARK_PLAN, Site.DSPARK_TARGET, Site.DSPARK_DRAFT_MULTINOMIAL, Site.DSPARK_DRAFT_GREEDY]
    for site in kept:
        s.sync(site, torch.zeros(2))
    assert len(g.calls) == len(kept)
    # a site switched off by SGLANG_SPEC_TP_SYNC stays off
    s._sites = frozenset()
    s.sync(Site.DSPARK_MEM, torch.zeros(1))
    assert len(g.calls) == len(kept)


def test_merge():
    m = load("merge")
    cls = make_sync_cls()
    m.install_sync(cls, Site)
    g = FakeGroup(delta=7)
    s = cls(g)
    correct_len = torch.tensor([1, 2, 3], dtype=torch.int64)
    bonus = torch.tensor([10, 20, 30], dtype=torch.int64)
    cap = torch.tensor([4, 5, 6], dtype=torch.int32)
    for _ in range(2):                                  # twice: the pending list is reset
        c0, b0, t0 = correct_len.clone(), bonus.clone(), cap.clone()
        s.sync(Site.DSPARK_ACCEPT_GRAPH, correct_len)
        s.sync(Site.DSPARK_ACCEPT_GRAPH, bonus)
        n = len(g.calls)
        assert torch.equal(correct_len, c0) and n == (0 if torch.equal(cap, torch.tensor([4, 5, 6], dtype=torch.int32)) else 1)
        s.sync(Site.DSPARK_ACCEPT_GRAPH, cap)
        assert len(g.calls) == n + 1 and g.calls[-1] == (3, 3), g.calls
        assert torch.equal(correct_len, c0 + 7) and torch.equal(bonus, b0 + 7)
        assert torch.equal(cap, t0 + 7) and cap.dtype == torch.int32
    assert len(g.calls) == 2
    # other sites are untouched by merge
    s.sync(Site.DSPARK_GRAPH_SAMPLE, torch.zeros(3, dtype=torch.int64))
    assert len(g.calls) == 3


def test_audit():
    m = load("audit")
    cls = make_sync_cls()
    m.install_sync(cls, Site)
    agree, differ = cls(FakeGroup(0)), cls(FakeGroup(1))
    for _ in range(5):
        agree.sync(Site.DSPARK_GRAPH_SAMPLE, torch.zeros(2, dtype=torch.int64))
    for _ in range(3):
        differ.sync(Site.DSPARK_ACCEPT_GRAPH, torch.zeros(2, dtype=torch.int64))
    m.tp_broadcast(FakeGroup(1), torch.zeros(4, dtype=torch.int64), "vcap")
    vals = m.audit_tick(force=True)
    assert vals == [[0, 5], [3, 3], [1, 1]], vals
    assert len(agree._tp_group.calls) == 5 and len(differ._tp_group.calls) == 3   # all kept


def test_vcap():
    m = load("vcap")
    g = FakeGroup(1)
    t = torch.zeros(2, dtype=torch.int64)
    m.tp_broadcast(g, t, "vcap")
    assert g.calls == [] and t.sum() == 0
    m = load("draft")
    m.tp_broadcast(g, t, "vcap")
    assert len(g.calls) == 1 and t.tolist() == [1, 1]
    # verify_cap calls it only when the gate is set
    os.environ["DSV41_VERIFY_CAP"] = "conf:0.1"
    os.environ["DSV41_SPEC_SYNC_FREE"] = "vcap"
    sys.modules.pop("spec_sync_free", None)
    sys.modules.pop("verify_cap", None)
    vc = importlib.import_module("verify_cap")
    g = FakeGroup(1)
    vc._state["tp_group"] = g
    vc._state["live"] = torch.full((vc.MAX_BS,), vc.STRIDE, dtype=torch.int64)
    vc.set_live_from_confidence(torch.full((2, 5), 0.9), 2)
    assert g.calls == [], "vcap: verify_cap still broadcast"
    os.environ["DSV41_SPEC_SYNC_FREE"] = ""
    vc.set_live_from_confidence(torch.full((2, 5), 0.9), 2)
    assert g.calls == [(2,)], g.calls


def test_ab_dispatch():
    """In-boot A/B: the env holds the union; each variant's parts decide at capture / replay."""
    fake = types.ModuleType("ab_variant")
    fake.ACTIVE, fake.current = True, {"parts": frozenset()}
    fake.parts = lambda name, all_parts: fake.current["parts"]
    sys.modules["ab_variant"] = fake
    try:
        m = load("1")
        cls = make_sync_cls()
        m.install_sync(cls, Site)
        g = FakeGroup()
        s = cls(g)
        fake.current["parts"] = frozenset()                    # v0: stock broadcasts
        s.sync(Site.DSPARK_GRAPH_SAMPLE, torch.zeros(1, dtype=torch.int64))
        for t in (torch.zeros(1, dtype=torch.int64),) * 3:
            s.sync(Site.DSPARK_ACCEPT_GRAPH, t)
        m.tp_broadcast(g, torch.zeros(1, dtype=torch.int64), "vcap")
        assert len(g.calls) == 5, g.calls
        fake.current["parts"] = frozenset({"draft", "accept", "vcap"})   # v1: all dropped
        s.sync(Site.DSPARK_GRAPH_SAMPLE, torch.zeros(1, dtype=torch.int64))
        for t in (torch.zeros(1, dtype=torch.int64),) * 3:
            s.sync(Site.DSPARK_ACCEPT_GRAPH, t)
        m.tp_broadcast(g, torch.zeros(1, dtype=torch.int64), "vcap")
        assert len(g.calls) == 5, g.calls
        fake.current["parts"] = frozenset({"merge"})           # v2: epilogue merged
        for t in (torch.zeros(1, dtype=torch.int64),) * 3:
            s.sync(Site.DSPARK_ACCEPT_GRAPH, t.clone())
        assert len(g.calls) == 6 and g.calls[-1] == (3, 1), g.calls
    finally:
        del sys.modules["ab_variant"]


def test_source_checks():
    m = load("merge,draft")

    class Ep:
        def _accept(self, *, candidates, logits, draft_tokens, seq_lens, cutoff_verify_lens=None):
            correct_len, bonus, cap_trim_lens = None, None, None
            self._tp_sync.sync(SpecTpSyncSite.DSPARK_ACCEPT_GRAPH, correct_len)
            self._tp_sync.sync(SpecTpSyncSite.DSPARK_ACCEPT_GRAPH, bonus)
            self._tp_sync.sync(SpecTpSyncSite.DSPARK_ACCEPT_GRAPH, cap_trim_lens)
            return correct_len

    class EpDrift:
        def _accept(self, *, candidates):
            correct_len = bonus = cap_trim_lens = None
            self._tp_sync.sync(SpecTpSyncSite.DSPARK_ACCEPT_GRAPH, correct_len)
            use(correct_len)
            self._tp_sync.sync(SpecTpSyncSite.DSPARK_ACCEPT_GRAPH, bonus)
            self._tp_sync.sync(SpecTpSyncSite.DSPARK_ACCEPT_GRAPH, cap_trim_lens)

    assert m._accept_syncs_back_to_back(Ep)
    assert not m._accept_syncs_back_to_back(EpDrift)

    class Sampler:
        def __call__(self, hidden_states, input_ids):
            bs = 1
            noise = self.exp_noise[:bs].exponential_()
            return noise

    class SamplerDrift:
        def __call__(self, hidden_states, input_ids):
            return torch.empty(1).exponential_()

    assert m._call_uses_exp_noise(Sampler) and not m._call_uses_exp_noise(SamplerDrift)
    mod = types.SimpleNamespace(DsparkDraftSampler=SamplerDrift, SpecTpSync=make_sync_cls(), SpecTpSyncSite=Site)
    try:
        m.install_sampler(mod)
    except RuntimeError:
        pass
    else:
        raise AssertionError("drifted sampler accepted")


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name, flush=True)
    print("ALL OK")
