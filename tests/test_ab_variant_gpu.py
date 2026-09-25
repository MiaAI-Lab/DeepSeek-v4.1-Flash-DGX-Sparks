"""Engine-image checks for adapter/ab_variant.py with the real adapters and the real graph backend.

Every L2 feature (the main gate included) is OFF in variant 0 and ON in variant 1.
A (torch): l2_prefetch reads its sub-gates per variant: the wo_a record, the Engram window, the
   draft bracket gate and the plan signature (variant in the cache key) follow the variant being
   captured / replayed.
A2 (torch): the REAL install_* functions install every hook although variant 0 has the feature
   off (they read the union in os.environ), and the installed hooks then act only in variant 1.
D (torch, CPU, 2 processes, gloo): the rank-0 marker rides a real broadcast, both ranks switch,
   the [v, -v, seq, -seq] MAX all-reduce reports agreement in the ack, and a forced seq mismatch
   makes both ranks raise.
B (sglang): fuse_quant's PARTS is the union and _off() selects per variant.
C (GPU, ~1 MB): SGLang's own FullCudaGraphBackend, hooked through ab_variant's import finder,
   captures one real graph per variant around hc_combine_norm_mxfp8, whose fused variant (hcpad)
   is ON only in variant 1. The fused kernel is traced only while variant 1 is current; replaying
   either set gives bit-identical outputs equal to eager stock, and both sets read the same static
   input (rewritten in place between replays). The warm-up reset runs before the second variant.

  docker run --rm --gpus all -v $PWD:/ds41 -w /ds41 -e PYTHONPATH=/ds41/adapter \
      -e DSV41_LAUNCHER=tp4 -e DSV41_AB_VARIANTS=2 \
      -e "DSV41_AB_V1=DSV41_L2_PREFETCH=1;DSV41_L2_PREFETCH_WOA=1;DSV41_L2_PREFETCH_ENGRAM=1;DSV41_L2_PREFETCH_DRAFT=1;DSV41_L2_PREFETCH_LMHEAD=1;DSV41_FUSE_QUANT=hcpad" \
      --entrypoint python3 <image> tests/test_ab_variant_gpu.py
(sitecustomize.py arms the harness at interpreter start, as in the engine; part C is skipped
without a GPU.)
"""
import json
import os
import types

os.environ.setdefault("DSV41_LAUNCHER", "tp4")   # TP4 line: set by start-tp4.sh
os.environ.setdefault("DSV41_AB_VARIANTS", "2")
os.environ.setdefault("DSV41_AB_V1", "DSV41_L2_PREFETCH=1;DSV41_L2_PREFETCH_WOA=1;DSV41_L2_PREFETCH_ENGRAM=1;"
                                     "DSV41_L2_PREFETCH_DRAFT=1;DSV41_L2_PREFETCH_LMHEAD=1;DSV41_FUSE_QUANT=hcpad")
import ab_variant as AB  # noqa: E402

if not AB.ACTIVE:           # sitecustomize not on the path: arm the way it would
    AB.configure()
assert AB.ACTIVE and AB.N == 2, "harness not armed"

import torch  # noqa: E402

CUDA = torch.cuda.is_available()
if CUDA and __name__ == "__main__":        # spawned gloo children (test D) never touch the GPU
    torch.cuda.set_per_process_memory_fraction(min(1.0, 1e9 / torch.cuda.get_device_properties(0).total_memory))
elif not CUDA:
    torch.cuda.is_current_stream_capturing = lambda: False

import l2_prefetch as LP  # noqa: E402


class _Mode:
    def __init__(self, verify):
        self.verify = verify

    def is_decode(self):
        return False

    def is_target_verify(self):
        return self.verify

    def is_draft_extend(self):
        return False

    def __str__(self):
        return "TARGET_VERIFY" if self.verify else "EXTEND"


def _fake_weight(ptr, nbytes):
    return types.SimpleNamespace(is_cuda=True, data_ptr=lambda: ptr, numel=lambda: nbytes, element_size=lambda: 1,
                                 stride=lambda: (1,), shape=(1,))


def test_l2_prefetch_per_variant():
    AB.set_runtime(0)
    assert not LP.enabled() and not LP._sub("WOA") and not LP._sub("ENGRAM") and not LP._sub("DRAFT")
    with AB.capturing(1):
        assert LP.enabled() and LP._sub("WOA") and LP._sub("ENGRAM") and LP._sub("DRAFT") and LP._sub("LMHEAD")
    for k in ("", "_WOA", "_ENGRAM", "_DRAFT", "_LMHEAD"):           # the union, for install-time gates
        assert os.environ["DSV41_L2_PREFETCH" + k] == "1"
    assert LP._installed() and LP._installed("ENGRAM") and not LP._installed("AHEAD")

    sig = ("t", 6, "m")
    for variant, want_w, want_e in ((0, 0, 0), (1, 1, 1)):
        LP._state["sig"], LP._state["learning"], LP._state["eng"] = sig, {sig: []}, 0
        with AB.capturing(variant):
            LP._record_wo_a(types.SimpleNamespace(shape=(4, 2, 4096)), _fake_weight(0x1000, 4096))
            LP._before_engram()
        ev = LP._state["learning"][sig]
        assert sum(e[0] == "w" for e in ev) == want_w and sum(e[0] == "e" for e in ev) == want_e, (variant, ev)
    LP._state["sig"], LP._state["learning"], LP._state["eng"] = None, {}, 0

    seen = []

    class Target:
        def forward(self, input_ids, positions, forward_batch):
            seen.append(("target", AB.current(), LP._state["sig"]))
            return "out"

    class Draft:
        def forward(self, input_ids, positions, forward_batch):
            seen.append(("draft", AB.current(), LP._state["sig"]))
            return "out"

    LP._bracket(Target)
    LP._bracket(Draft, gate="DRAFT")
    t, d = Target(), Draft()
    ids, fb = torch.zeros(6, dtype=torch.long), types.SimpleNamespace(forward_mode=_Mode(True))
    for variant in (0, 1):
        with AB.capturing(variant):
            assert t.forward(ids, None, fb) == "out" and d.forward(ids, None, fb) == "out"
    tsig = [s for k, v, s in seen if k == "target"]
    assert tsig[0] is None and tsig[1] is not None and tsig[1][-1] == 1             # L2 only in v1, variant in the key
    dsig = [s for k, v, s in seen if k == "draft"]
    assert dsig[0] is None and dsig[1] is not None and dsig[1][-1] == 1              # DRAFT only in v1
    # a prefill forward is never bracketed
    seen.clear()
    t.forward(ids, None, types.SimpleNamespace(forward_mode=_Mode(False)))
    assert seen[0][2] is None
    assert LP._state["sig"] is None


def test_real_installs_feature_only_in_v1():
    """Regression for install gates read through the variant: variant 0 has L2 off entirely."""
    AB.set_runtime(0)
    events = []

    def mxfp8_linear(input, weight, weight_scale, *a, **k):
        return "y"

    fp8 = types.SimpleNamespace(flashinfer_mxfp8_blockscaled_linear=mxfp8_linear)
    router = types.SimpleNamespace(tiny_gemm_bf16=lambda h, w, *a, **k: "r")

    class Roce:
        def should_allreduce(self, inp):
            return True

        def all_reduce(self, inp, *a, **k):
            return "ar"

        def all_gather(self, inp, *a, **k):
            return "ag"

    class Model:
        def forward(self, input_ids, positions, forward_batch):
            events.append(("model", AB.current(), LP._state["sig"]))
            return "h"

    class Causal:
        def __init__(self):
            self.model = Model()
            self.lm_head = types.SimpleNamespace(weight=_fake_weight(0x7000, 1 << 20))

        def forward(self, input_ids, positions, forward_batch):
            return self.model.forward(input_ids, positions, forward_batch)

    class Draft:
        def forward(self, input_ids, positions, forward_batch):
            events.append(("draft", AB.current(), LP._state["sig"]))
            return "d"

    class Engram:
        def _owned_rows(self, indices):
            return "rows"

    v4 = types.SimpleNamespace(DeepseekV4Model=Model, DeepseekV4ForCausalLM=Causal,
                               _apply_wo_a_bf16_matmul=lambda o, w, *a, **k: "wo_a")
    saved_lib = LP._lib
    LP._lib = lambda: None                                   # install_roce pre-compiles the kernel
    try:
        LP.install_fp8_utils(fp8)
        LP.install_router(router)
        LP.install_roce(types.SimpleNamespace(RoceOneshotAllReduce=Roce))
        LP.install_model(v4)
        LP.install_draft(types.SimpleNamespace(DeepseekV4ForCausalLMDSpark=Draft))
        LP.install_engram(types.SimpleNamespace(EngramEmbedding=Engram))
    finally:
        LP._lib = saved_lib
    assert fp8.flashinfer_mxfp8_blockscaled_linear is not mxfp8_linear, "fp8 linear not installed"
    assert router.tiny_gemm_bf16.__module__ == "l2_prefetch", "router not installed"
    assert getattr(Roce, "_dsv41_l2_prefetch", False), "RoCE hooks not installed"
    assert getattr(Model, "_dsv41_l2_prefetch", False), "model bracket not installed"
    assert getattr(v4._apply_wo_a_bf16_matmul, "_dsv41_l2_prefetch", False), "wo_a recorder not installed"
    assert getattr(Draft, "_dsv41_l2_prefetch", False), "draft bracket not installed"
    assert getattr(Engram, "_dsv41_l2_prefetch", False), "engram window not installed"
    assert Causal.forward.__name__ == "causal_forward", "LM head recorder not installed"

    ids, fb = torch.zeros(6, dtype=torch.long), types.SimpleNamespace(forward_mode=_Mode(True))
    causal, draft = Causal(), Draft()
    for variant in (0, 1):
        with AB.capturing(variant):
            causal.forward(ids, None, fb)
            draft.forward(ids, None, fb)
    sigs = {(k, v): s for k, v, s in events}
    assert sigs[("model", 0)] is None and sigs[("draft", 0)] is None           # v0: L2 off, nothing bracketed
    assert sigs[("model", 1)][-1] == 1 and sigs[("draft", 1)][-1] == 1
    # inside a v1 forward the installed hooks record; inside v0 they do not
    for variant, want in ((0, 0), (1, 4)):
        sig = ("rec", variant)
        LP._state["sig"], LP._state["learning"] = sig, {sig: []}
        LP._state["idx"] = LP._state["lin"] = LP._state["eng"] = 0
        with AB.capturing(variant):
            fp8.flashinfer_mxfp8_blockscaled_linear(None, _fake_weight(0x1000, 4096), _fake_weight(0x2000, 64))
            Engram()._owned_rows(None)
        kinds = [e[0] for e in LP._state["learning"][sig]]
        # the fp8 wrapper records weight + scale ("r", "r") and the WOA linear mark ("L"); Engram an "e"
        assert (len(kinds) if variant else kinds.count("L") + kinds.count("e")) == want, (variant, kinds)
    LP._state["sig"], LP._state["learning"] = None, {}
    LP._state["idx"] = LP._state["lin"] = LP._state["eng"] = 0


def _rank_main(rank, tmp):
    import torch.distributed as dist
    dist.init_process_group("gloo", init_method=f"file://{tmp}/pg", rank=rank, world_size=2)
    flag = os.path.join(tmp, "ab_variant")
    AB._FILE, AB._POLL_S = flag, 0.0

    class Receiver:
        tp_cpu_group = dist.group.WORLD

        def __init__(self):
            self.ps = types.SimpleNamespace(pp_size=1)

        def _pull_raw_reqs(self):
            return ["req"] if rank == 0 else None

        def recv_requests(self, local_reqs=None):
            obj = [self._pull_raw_reqs()]
            dist.broadcast_object_list(obj, src=0)
            return obj[0]

    AB.install_receiver(types.SimpleNamespace(SchedulerRequestReceiver=Receiver))
    rx = Receiver()
    out = {}
    assert rx.recv_requests() == ["req"] and AB._runtime == 0
    if rank == 0:
        with open(flag + ".tmp", "w") as f:
            f.write("1 t1\n")
        os.replace(flag + ".tmp", flag)
    dist.barrier()
    out["reqs"] = rx.recv_requests()
    out["runtime"], out["seq"], out["ranks"] = AB._runtime, AB._state["seq"], AB._state["ranks"]
    if rank == 0:
        out["ack"] = json.load(open(flag + ".ack"))
    # forced divergence: rank 1's switch count is off; the next switch must raise on both
    if rank == 1:
        AB._state["seq"] += 5
    if rank == 0:
        with open(flag + ".tmp", "w") as f:
            f.write("0 t2\n")
        os.replace(flag + ".tmp", flag)
    dist.barrier()
    try:
        rx.recv_requests()
        out["raised"] = False
    except RuntimeError as exc:
        out["raised"] = "disagree" in str(exc)
    if rank == 0:
        out["ack2"] = json.load(open(flag + ".ack"))
    with open(os.path.join(tmp, f"rank{rank}.json"), "w") as f:
        json.dump(out, f)
    dist.destroy_process_group()


def test_ranks_agree_over_gloo():
    import tempfile
    import torch.multiprocessing as mp
    tmp = tempfile.mkdtemp()
    mp.spawn(_rank_main, args=(tmp,), nprocs=2, join=True)
    r = [json.load(open(os.path.join(tmp, f"rank{i}.json"))) for i in range(2)]
    for x in r:
        assert x["reqs"] == ["req"] and x["runtime"] == 1 and x["seq"] == 1, x   # marker stripped, switched
        assert x["ranks"] == {"world": 2, "variant": [1, 1], "seq": [1, 1], "agree": True}, x["ranks"]
        assert x["raised"] is True, x                                               # both refuse the mismatch
    assert r[0]["ack"]["ranks"]["agree"] is True and r[0]["ack"]["token"] == "t1"
    assert r[0]["ack2"]["ranks"]["agree"] is False and r[0]["ack2"]["ranks"]["seq"] == [2, 7]


def test_fuse_quant_parts_per_variant():
    try:
        import fuse_quant as FQ
    except Exception as exc:  # needs the engine's sglang + triton
        print(f"SKIP fuse_quant ({exc})")
        return
    assert "hcpad" in FQ.PARTS
    AB.set_runtime(0)
    assert FQ._off("hcpad")
    with AB.capturing(1):
        assert not FQ._off("hcpad") and FQ._off("qnorm")
    AB.set_runtime(1)
    assert not FQ._off("hcpad")
    FQ._state["disabled"].add("hcpad")
    assert FQ._off("hcpad")                                     # the live check still wins
    FQ._state["disabled"].discard("hcpad")
    AB.set_runtime(0)


def test_real_backend_two_graph_sets():
    if not CUDA:
        print("SKIP real backend (no GPU)")
        return
    import fuse_quant as FQ
    import sglang.srt.model_executor.runner_backend.full_cuda_graph_backend as FB
    from sglang.kernels.ops.layernorm import hc_combine_norm as hcn
    from sglang.srt.model_executor.runner.shape_key import ShapeKey
    assert getattr(FB.FullCudaGraphBackend, "_dsv41_ab", False), "finder did not hook the backend"

    stock = hcn.hc_combine_norm_mxfp8
    FQ._install_hcpad()
    wrapper = hcn.hc_combine_norm_mxfp8
    pad_calls = []
    real_pad = FQ.hc_combine_norm_mxfp8_pad

    def counted_pad(*a, **k):
        pad_calls.append((AB.current(), torch.cuda.is_current_stream_capturing()))
        return real_pad(*a, **k)

    FQ.hc_combine_norm_mxfp8_pad = counted_pad
    try:
        dev, m, eps = "cuda", 4, 1e-6
        g = torch.Generator(device=dev).manual_seed(7)
        nw = (torch.rand(5120, device=dev, generator=g) * 2).to(torch.bfloat16)
        x = (torch.randn(m, 20480, device=dev, generator=g) * 10).to(torch.bfloat16)
        pre = torch.rand(m, 4, device=dev, generator=g)

        def fwd():
            return wrapper(x, pre, nw, eps)

        runner = types.SimpleNamespace(
            device_module=torch.cuda,
            model_runner=types.SimpleNamespace(tp_group=types.SimpleNamespace(barrier=lambda: None, world_size=1),
                                               is_draft_worker=False))
        be = FB.FullCudaGraphBackend(runner)
        resets = []
        key = ShapeKey(size=m)
        AB.set_runtime(0)
        with be.capture_session(torch.cuda.Stream()):
            be.capture_one(key, fwd, None, lambda: resets.append(AB.current()))
        torch.cuda.synchronize()
        assert resets == [0, 0, 0, 1, 1], resets   # v0's 2 warm-ups, the reset before v1, v1's 2
        # fused kernel traced only for variant 1: two eager warm-ups (the first is the live check) + capture
        assert [c for c, _ in pad_calls] == [1, 1, 1] and [cap for _, cap in pad_calls] == [False, False, True]
        assert set(be._graphs) == {key} and key in be._dsv41_ab_sets[1][0]

        def eq(a, b):
            return all(torch.equal(p.view(torch.uint8) if p.dtype == torch.float8_e4m3fn else p,
                                   q.view(torch.uint8) if q.dtype == torch.float8_e4m3fn else q)
                       for p, q in zip(a, b))

        for rnd in range(2):
            ref = stock(x, pre, nw, eps)
            outs = []
            for v in (0, 1, 0, 1):
                AB.set_runtime(v)
                out = be.replay(key, None)
                torch.cuda.synchronize()
                outs.append(tuple(t.clone() for t in out))
            assert all(eq(o, ref) for o in outs), f"round {rnd}: a graph set differs from eager stock"
            # the two sets are distinct graphs with distinct outputs over the same static input
            assert be._dsv41_ab_sets[0][1][key][0].data_ptr() != be._dsv41_ab_sets[1][1][key][0].data_ptr()
            x.copy_((torch.randn(m, 20480, device=dev, generator=g) * 3).to(torch.bfloat16))
        assert AB._state["replays"]["target"][:2] == [4, 4]
        be.cleanup()
        assert not be._graphs
        AB.set_runtime(0)
    finally:
        FQ.hc_combine_norm_mxfp8_pad = real_pad
        hcn.hc_combine_norm_mxfp8 = stock


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("PASS", name, flush=True)
