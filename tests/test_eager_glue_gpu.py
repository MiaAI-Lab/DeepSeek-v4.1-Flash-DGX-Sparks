"""GPU checks for adapter/eager_glue.py (one GPU, < 1 GB, in the engine image):

  docker run --rm --gpus all --network none -v <ds41>:/ds41:ro -e PYTHONPATH=/ds41/adapter \
      --entrypoint python3 <image> /ds41/tests/test_eager_glue_gpu.py

1. fence: fused_clone on CUDA gives the same bytes in ONE kernel (profiler count) where the six
   clones are six memcpys;
2. vcapk: torch.cumprod's association on [bs, 5] fp32 is identified (sequential through cub for one
   row, Sklansky for more) and the fused kernel matches the torch ops bit for bit on rows built to
   sit within ulps of the threshold (both threshold forms, NaN and negative inputs, bs 1..16, 33, 64);
   vcap: verify_cap's live-length update captured in a CUDA graph behind a draft-sampler-shaped
   call, replayed on fresh confidences (random, and products placed exactly on the threshold),
   leaves the live buffer bit-identical to the eager call on the same tensor, for every
   padding-free size 1..8; the captured key is registered and the eager call is skipped for it;
3. glue: the image's own MetadataGlueGraph under eager_glue's checked subclass reproduces an eager
   prep bit for bit over steps with changing inputs (plain and dglue move-cache variants); a prep
   that reads a host value, and one that swaps a tensor by reference, are caught by the check and
   fall back to eager;
4. the engine sources this adapter hooks still have the expected shape.
"""
import os
import sys
import types

# TP4-line adapters: armed only under start-tp4.sh, which sets DSV41_LAUNCHER=tp4 in every rank.
os.environ["DSV41_LAUNCHER"] = "tp4"

os.environ.setdefault("DSV41_EAGER_GLUE", "all")
os.environ.setdefault("DSV41_VERIFY_CAP", "conf:0.1")
import torch  # noqa: E402

torch.cuda.set_per_process_memory_fraction(min(1.0, 1e9 / torch.cuda.get_device_properties(0).total_memory))
import eager_glue as eg  # noqa: E402
import verify_cap as vc  # noqa: E402

DEV = "cuda"


def kernels_of(fn):
    from torch.profiler import ProfilerActivity, profile
    fn()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        fn()
        torch.cuda.synchronize()
    return [e.name for e in prof.events() if e.device_type.name == "CUDA"]


def test_fence():
    bs = 3
    bufs = [torch.randint(0, 6, (64,), device=DEV), torch.randint(0, 129280, (64,), device=DEV),
            torch.randint(0, 6, (64,), device=DEV, dtype=torch.int32),
            torch.randint(1, 7, (64,), device=DEV, dtype=torch.int32),
            torch.randint(0, 1 << 40, (64,), device=DEV), torch.randint(0, 129280, (64, 6), device=DEV)]
    src = [b[:bs] for b in bufs]
    out = eg.fused_clone(src)
    assert all(torch.equal(o, s) and o.dtype == s.dtype and o.shape == s.shape for o, s in zip(out, src))
    fused = kernels_of(lambda: eg.fused_clone(src))
    stock = kernels_of(lambda: [s.clone() for s in src])
    assert len(fused) == 1 and "Cat" in fused[0], fused
    assert len(stock) == 6, stock
    print(f"  fence: stock {len(stock)} ops ({stock[0][:40]}...), fused {len(fused)} ({fused[0][:50]})")


class FakeGroup:
    world_size = 1

    def broadcast(self, t, src=0):
        return t


def test_vcap_graph():
    import sglang.srt.runtime_context as rc                     # the image's; only get_exec is faked
    sizes = [1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16]
    real_get_exec = rc.get_exec
    rc.get_exec = lambda: types.SimpleNamespace(graph=types.SimpleNamespace(
        cuda_graph_config=types.SimpleNamespace(decode=types.SimpleNamespace(bs=sizes))))
    eg._vcap["eligible"] = None
    try:
        _vcap_graph_body()
    finally:
        rc.get_exec = real_get_exec


def _vcap_graph_body():
    vc._state["tp_group"] = FakeGroup()

    class DsparkDraftSampler:
        query_token_num = 6
        folded_sampling = False
        temperatures = None

        def __init__(self):
            self.confidence_out = torch.zeros(16, 5, device=DEV)

        def __call__(self, hidden_states, input_ids):
            bs = hidden_states.shape[0] // self.query_token_num
            self.confidence_out[:bs].copy_(hidden_states.view(bs, 6)[:, 1:])
            return None

        def stage_sampling_params(self, *, bs, sampling_info):
            pass

    mod = types.SimpleNamespace(DsparkDraftSampler=DsparkDraftSampler)
    eg.install_sampler(mod)
    orig = eg._vcap["orig"]
    s = DsparkDraftSampler()
    g = torch.Generator(device=DEV).manual_seed(7)
    thr = 0.1
    for bs in range(1, 9):
        static_in = torch.rand(bs * 6, device=DEV, generator=g)
        vc.live_buf(DEV)
        for _ in range(2):                                    # warm-ups, as capture_one does
            s(static_in, None)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            s(static_in, None)
        assert (None, bs) in eg._vcap["captured"], eg._vcap["captured"]
        for trial in range(200):
            x = torch.rand(bs * 6, device=DEV, generator=g)
            if trial % 4 == 1:                                # products straddling / on the threshold
                x.view(bs, 6)[:, 1] = thr
                x.view(bs, 6)[:, 2:] = 1.0
            elif trial % 4 == 2:
                x.view(bs, 6)[:, 1:] = torch.tensor([0.5, 0.2, 1.0, 1.0, 1.0], device=DEV)
            elif trial % 4 == 3:
                x.view(bs, 6)[:, 1] = torch.nextafter(torch.tensor(thr, device=DEV), torch.tensor(1.0, device=DEV))
                x.view(bs, 6)[:, 2:] = 0.999999
            static_in.copy_(x)
            vc.live_buf().fill_(99)
            graph.replay()
            got = vc.live_buf()[:bs].clone()
            vc.live_buf().fill_(98)
            orig(s.confidence_out[:bs], bs)                    # the eager call on the same tensor
            ref = vc.live_buf()[:bs].clone()
            assert torch.equal(got, ref), (bs, trial, got.tolist(), ref.tolist())
    # bs 10 (9 pads into it) is never captured with vcap
    s(torch.rand(60, device=DEV), None)
    assert (None, 10) not in eg._vcap["captured"]
    print(f"  vcap: captured keys {sorted(k[1] for k in eg._vcap['captured'])}, 8 x 200 replays bit-identical")


def _orders(x):
    """cumprod of [n, 5] fp32 rows in three associations (each product rounded to fp32)."""
    x0, x1, x2, x3, x4 = x.unbind(1)
    c1 = x0 * x1
    seq = torch.stack([x0, c1, c1 * x2, c1 * x2 * x3, c1 * x2 * x3 * x4], 1)
    t3 = (x2 * x3) * c1
    skl = torch.stack([x0, c1, x2 * c1, t3, x4 * t3], 1)
    k12, k23, k34 = x1 * x2, x2 * x3, x3 * x4
    ks3 = c1 * k23
    kog = torch.stack([x0, c1, x0 * k12, ks3, x0 * (k12 * k34)], 1)
    return {"sequential": seq, "sklansky": skl, "kogge-stone": kog}


def test_scan_order():
    """Which association torch.cumprod(dim=1) uses on [bs, 5] fp32 (the fused kernel copies it)."""
    g = torch.Generator(device=DEV).manual_seed(11)
    found = {}
    for bs in (1, 2, 3, 5, 8, 16):
        match = {k: 0 for k in ("sequential", "sklansky", "kogge-stone")}
        diff_seq_skl = 0
        for _ in range(3000):
            x = torch.rand(bs, 5, device=DEV, generator=g) * 0.9 + 0.1
            ref = torch.cumprod(x.clamp(0, 1), dim=1)
            cands = _orders(x)
            for k, c in cands.items():
                match[k] += int(torch.equal(c.view(torch.int32), ref.view(torch.int32)))
            diff_seq_skl += int(not torch.equal(cands["sequential"], cands["sklansky"]))
        found[bs] = [k for k, v in match.items() if v == 3000]
        assert diff_seq_skl > 100, "inputs too easy to tell the orders apart"
    assert found[1] == ["sequential"], found
    assert all(found[b] == ["sklansky"] for b in (2, 3, 5, 8, 16)), found
    print(f"  cumprod association: bs=1 {found[1]}, bs>1 {found[2]} (3000 draws each, exact)")


def test_vcapk():
    """The fused kernel vs verify_cap's torch ops, on rows built to land within ulps of the threshold."""
    assert eg._vcap_kernel is not None
    vc._state["tp_group"] = FakeGroup()
    if eg._vcap["orig"] is None:                               # verify_cap's own torch ops
        eg._vcap["orig"] = vc.set_live_from_confidence
    g = torch.Generator(device=DEV).manual_seed(5)
    total = flips = 0
    for thr_spec in (0.1, [0.0, 0.25, 0.25, 0.03, 0.06]):
        vc._state["thr"] = thr_spec
        vc._state["thr_t"] = None
        for bs in list(range(1, 17)) + [33, 64]:
            for trial in range(150):
                x = torch.rand(bs, 5, device=DEV, generator=g) * 0.7 + 0.3
                t = 0.1 if not isinstance(thr_spec, list) else 0.03
                j = trial % 5
                prod = torch.cumprod(x, 1)[:, j]
                x[:, 0] *= t / prod                           # position j lands on the threshold
                x[:, 0] += (torch.randint(-3, 4, (bs,), device=DEV, generator=g) * 1e-8)
                if trial % 7 == 0:
                    x[0, 2] = float("nan")
                if trial % 11 == 0:
                    x[-1, 1] = -0.5
                conf = x.contiguous()
                c = _orders(conf.clamp(0, 1))
                flips += int(not torch.equal(c["sequential"] >= t, c["sklansky"] >= t))
                vc.live_buf(DEV).fill_(77)
                eg._vcap["orig"](conf, bs)
                ref = vc.live_buf()[:bs].clone()
                out = torch.empty_like(ref)
                eg._launch_vcapk(vc, conf, bs, out)
                assert torch.equal(out, ref), (thr_spec, bs, trial, out.tolist(), ref.tolist())
                total += 1
    vc._state["thr"] = 0.1
    vc._state["thr_t"] = None
    assert flips > 20, flips
    print(f"  vcapk: {total} cases bit-identical to the torch ops ({flips} where the two associations disagree)")


class Meta:
    def __init__(self, **kw):
        self.__dict__.update(kw)

    def copy_(self, other):
        for k, v in self.__dict__.items():
            v.copy_(other.__dict__[k])


class FakeBackend:
    """DSV4-shaped replay prep: temps built from the static fb buffers, copied into a persistent
    per-bs metadata object that becomes forward_metadata."""
    attn_backend_list = None

    def __init__(self, mode="plain"):
        self.mode = mode
        self.persist = {}
        self.forward_metadata = None

    def _move_to_device(self, x):
        return torch.tensor(x, dtype=torch.int32, pin_memory=True).to(DEV, non_blocking=True)

    def init_forward_metadata_out_graph(self, fb):
        bs = fb.bs
        seq = fb.seq_lens[:bs]
        ext = self._move_to_device([5] * bs) if self.mode == "dglue" else torch.full((bs,), 5, device=DEV,
                                                                                     dtype=torch.int32)
        extra = fb.host_val if self.mode == "hostfed" else 0
        tmp = Meta(a=(seq + ext + extra).to(torch.int32), b=fb.req[:bs] * 7, c=torch.cumsum(seq, 0))
        chosen = self.persist.get(bs)
        if chosen is None:
            self.persist[bs] = tmp
            self.forward_metadata = tmp
            return
        if self.mode == "byref":
            chosen.c = tmp.c                                   # a field swapped by reference
            chosen.a.copy_(tmp.a)
            chosen.b.copy_(tmp.b)
        else:
            chosen.copy_(tmp)
        self.forward_metadata = chosen


def run_glue(mode, kind, steps=40):
    from sglang.srt.model_executor.runner.metadata_glue_graph import MetadataGlueGraph
    be, ref_be = FakeBackend(mode), FakeBackend(mode if mode != "dglue" else "plain")
    glue = eg._make_glue(MetadataGlueGraph, torch.device(DEV), kind)
    seq = torch.zeros(16, dtype=torch.int64, device=DEV)
    req = torch.zeros(16, dtype=torch.int64, device=DEV)
    g = torch.Generator(device=DEV).manual_seed(3)
    for step in range(steps):
        bs = 1 + step % 3
        seq.copy_(torch.randint(0, 1 << 20, (16,), device=DEV, generator=g))
        req.copy_(torch.randint(0, 64, (16,), device=DEV, generator=g))
        fb = types.SimpleNamespace(bs=bs, seq_lens=seq, req=req, host_val=step)
        if glue.disabled:
            be.init_forward_metadata_out_graph(fb)
        else:
            glue.run(be, fb, (bs,))
        ref_be.init_forward_metadata_out_graph(fb)
        m, r = be.forward_metadata, ref_be.forward_metadata
        for k in ("a", "b", "c"):
            assert torch.equal(getattr(m, k), getattr(r, k)), (mode, step, k)
    return glue


def test_glue():
    g = run_glue("plain", "tvglue")
    assert not g.disabled and all(st["graph"] is not None for st in g._states.values())
    g = run_glue("dglue", "dglue")
    assert not g.disabled and all(st["graph"] is not None for st in g._states.values())
    g = run_glue("hostfed", "tvglue")
    assert g.disabled, "a host-fed prep must be caught"
    g = run_glue("byref", "tvglue")
    assert g.disabled, "a by-reference field swap must be caught"
    print("  glue: plain + dglue exact over 40 steps; host-fed and by-reference preps caught")


def test_engine_sources():
    import inspect
    from sglang.srt.model_executor.runner import decode_cuda_graph_runner as dcgr
    from sglang.srt.speculative.dspark_components import dspark_draft, dspark_draft_sampler
    src = inspect.getsource(dcgr.DecodeCudaGraphRunner.load_batch)
    assert "self._metadata_glue.run(" in src and "raw_bs == bs" in src
    assert "self.NUM_WARMUP" in inspect.getsource(dcgr.MetadataGlueGraph.run)
    assert "def stage_sampling_params(self, *, bs: int, sampling_info)" in inspect.getsource(
        dspark_draft_sampler.DsparkDraftSampler)
    assert "self.confidence_out[:bs].copy_(confidence)" in inspect.getsource(
        dspark_draft_sampler.DsparkDraftSampler.__call__)
    assert "folded=folded" in inspect.getsource(dspark_draft.DraftBlockProposer.propose)
    from sglang.srt.layers.attention import deepseek_v4_backend as dsv4
    body = inspect.getsource(dsv4.DeepseekV4AttnBackend.init_forward_metadata_dspark_draft_block)
    assert "self._move_to_device(lengths.extend_seq_lens_cpu)" in body
    print("  engine sources: hooks present")


if __name__ == "__main__":
    tests = [test_fence, test_scan_order, test_vcapk, test_vcap_graph, test_glue, test_engine_sources]
    for t in tests:
        t()
        print("PASS", t.__name__, flush=True)
    print(f"{len(tests)}/{len(tests)} PASS")
