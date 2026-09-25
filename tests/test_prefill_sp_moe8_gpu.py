"""GPU checks for prefill SP stage 2b (DSV41_PREFILL_SP_FP8_MOE=1) on one GPU (<= 2 GB), in the image:

  docker run --rm --gpus all --network none -v <ds41>:/ds41:ro -v <model>:/model:ro \
      -v <cache>:/cache -e B12X_NEXT_COMPILE_CACHE_DIR=/cache \
      -e PYTHONPATH=/ds41/runtime/fetched/b12x_next:/ds41/adapter:/ds41/tests:/opt/b12x \
      --entrypoint python3 <image> /ds41/tests/test_prefill_sp_moe8_gpu.py [quant] [moe] [router] [emulate]

The routed MoE is the real engine method + adapter/moe_b12x_next on real checkpoint experts
(tests/test_moe_b12x_next.py's shell, CKPT_EXPERTS of layer 5, deterministic reduction, capacities
{5, 6, 2048, 4096}), on the PATCHED b12x_next (runtime/fetched/b12x_next = scripts/fetch_runtime.sh output).

quant    b12x's in-kernel A8 quantization (read back from the plan's packed_input /
         packed_input_scale after a normal bf16 launch) == FlashInfer mxfp8_quantize (SGLang's
         flashinfer_mxfp8_quantize, linear scales) of the same rows, full-M and per 4-way shard,
         == a torch model of b12x's quantize_block_fp8_mx:
           - realistic rows at M = 4096 / 3152 / 2052 / 2048 / 1500 (shards 1024 / 788 / 513 / 512 / 375)
           - exhaustive: every finite positive bf16 value as a block maximum (scale byte), and for a
             ladder of block maxima covering every scale byte (mantissas 1.0 / 1.0078 / 1.75 / 1.99 and
             every subnormal), every finite bf16 value within 2^-20 of the maximum as an element
moe      pre-quantized launch (FlashInfer rows written into prequant_views, front-end variant that
         does not quantize) == the normal bf16 launch, bit for bit, at M = 4096 / 3152 / 2052 / 2048;
         the launch leaves the input storage untouched; a corrupted scale byte changes the output
         (the kernel reads the caller's bytes)
router   router GEMM (MoEGate.forward: cuBLAS linear_bf16_fp32 at these M) and vision_topk on a
         1/4 shard == rows of the full-M call, and on a zero-padded full-M input
emulate  4 rank-threads: the real DeepseekV2MoE.forward / forward_normal / _forward_shared_experts,
         FusedMoE.forward / forward_impl / _dispatch_with_pre_quant, MoEGate, vision_topk on real
         router weights, b12x experts per TP rank, an MXFP8 shared expert (b12x dense), inside
         prefill_sp's layer loop (real hc kernels): stage 2b == stage 1 bit for bit on every rank,
         EXACT and fast, M = 2052 / 2048 / 3152 (EMU_CASES)
"""
import os
import sys
import threading
import time
import types
from types import SimpleNamespace as NS

os.environ.setdefault("DSV41_MOE_B12X_NEXT", "1")
os.environ.setdefault("DSV41_MOE_B12X_NEXT_DETERMINISTIC", "1")
os.environ.setdefault("DSV41_MOE_B12X_NEXT_GRAPH_BS", "1")
os.environ.setdefault("DSV41_MOE_B12X_NEXT_LADDER", "2048,4096")
os.environ.setdefault("DSV41_MOE_B12X_NEXT_TUNE", "0")
os.environ.setdefault("DSV41_MOE_B12X_NEXT_MEMSTATS", "0")
os.environ.setdefault("CHUNKED_PREFILL_SIZE", "4096")
os.environ.setdefault("DSV41_PREFILL_SP", "1")
os.environ.setdefault("DSV41_PREFILL_SP_FP8_MOE", "1")
os.environ.setdefault("CKPT_EXPERTS", "24")
os.environ.setdefault("TEST_MAX_GB", "2.0")

import torch  # noqa: E402

import test_moe_b12x_next as tb  # noqa: E402  (engine stubs, adapter installed, checkpoint access)
import moe_b12x_next as ad  # noqa: E402
import prefill_sp as sp  # noqa: E402

DEV, BF, H, W = tb.DEV, torch.bfloat16, 5120, 4
NE = tb.NE
KB = H // 32
LOG = []


def say(msg):
    print(msg, flush=True)
    LOG.append(msg)


_LAYERS = {}


def layer(rank=0):
    """FusedMoE shell of layer 5, TP rank `rank` (real checkpoint experts), converted by the adapter."""
    if rank not in _LAYERS:
        t0 = time.time()
        ly, me, src, _ = tb.make_layer("layers.5", n_global=NE, topk=6, ep_size=1, ep_rank=0,
                                       moe_tp_size=4, moe_tp_rank=rank)
        _LAYERS[rank] = (ly, me, src)
        g = ly._dsv41_b12x_next.geom
        say(f"  layer TP rank {rank}: {time.time() - t0:.1f} s, caps {g.caps}, chunk {g.chunk}, "
            f"pre-quantized warm {ad.PREQUANT_STATS['warm_caps'].get(g.key)}, "
            f"tiles {[(c, 'M64' if 'dynamic_tile_m=64' in g.plans[c].config else 'M16') for c in g.caps]}")
    return _LAYERS[rank]


def rows(m, seed, kind="act"):
    g = torch.Generator(device=DEV).manual_seed(seed)
    x = torch.randn(m, H, device=DEV, generator=g)
    if kind == "act":               # RMSNorm-output-like: unit rows, a few hot channels, rare spikes
        x[:, :64] *= 12
        x[:, 1000:1008] *= 60
        spikes = torch.rand(m, H, device=DEV, generator=g) < 1e-4
        x = torch.where(spikes, x * 300, x)
    return x.to(BF)


def routing(m, seed):
    g = torch.Generator(device=DEV).manual_seed(seed)
    ids = torch.rand(m, NE, device=DEV, generator=g).argsort(dim=1)[:, :6].to(torch.int32).contiguous()
    w = torch.softmax(torch.randn(m, 6, device=DEV, generator=g), dim=1).float()
    return ids, w


# ------------------------------------------------------------------------------------ drift
def test_drift():
    """The stage-2b engine sources hash as pinned in this image; a drifted one is refused."""
    import test_prefill_sp_gpu as tg
    tg.setup()
    old = sp.MODE, sp.FP8, sp.FP8_MOE
    sp.MODE, sp.FP8, sp.FP8_MOE = "shard", True, True
    try:
        sp.check_engine()
        saved = dict(sp._EXPECTED_FP8_MOE)
        sp._EXPECTED_FP8_MOE["fmoe.FusedMoE._dispatch_with_pre_quant"] = "0000000000000000"
        try:
            sp.check_engine()
            raise AssertionError("drift accepted")
        except RuntimeError as exc:
            assert "drifted" in str(exc)
        finally:
            sp._EXPECTED_FP8_MOE.clear()
            sp._EXPECTED_FP8_MOE.update(saved)
    finally:
        sp.MODE, sp.FP8, sp.FP8_MOE = old
    say(f"drift: {len(sp._EXPECTED) + len(sp._EXPECTED_FP8) + len(sp._EXPECTED_FP8_MOE)} pinned engine sources "
        f"match (incl. {len(sp._EXPECTED_FP8_MOE)} for stage 2b); a drifted one is refused: OK")


# ------------------------------------------------------------------------------------ quantizers
def fi_quant(x):
    """SGLang's flashinfer_mxfp8_quantize, linear scales: (E4M3 bytes [M, K], ue8m0 [M, K/32])."""
    q, sf, _ = sp.mxfp8_shard(x)
    return q, sf


def fi_quant_shards(x):
    m = x.shape[0]
    qs, ss = [], []
    for r in range(W):
        lo, hi = r * (m // W), (r + 1) * (m // W)
        q, sf = fi_quant(x[lo:hi].contiguous())
        qs.append(q)
        ss.append(sf)
    return torch.cat(qs), torch.cat(ss)


def b12x_torch(x):
    """quantize_block_fp8_mx modelled in torch with b12x's own replicas (pow2_ceil_ue8m0_torch,
    _ue8m0_output_scale_torch): scale byte = pow2ceil(amax * fl(1/448)), payload = RN-sat e4m3 of
    v * 2^(127 - byte)."""
    from b12x_next._lib import intrinsics as it
    m = x.shape[0]
    blk = x.float().view(m, KB, 32)
    amax = blk.abs().amax(-1, keepdim=True)
    _, byte = it.pow2_ceil_ue8m0_torch(amax * it._INV_FLOAT8_E4M3_MAX)
    inv = it._ue8m0_output_scale_torch(byte)
    q = (blk * inv).clamp(-448.0, 448.0).to(torch.float8_e4m3fn).view(torch.uint8).view(m, H)
    return q, byte.view(m, KB)


def b12x_kernel(x, rank=0):
    """b12x's own in-kernel quantization of x: a normal launch, then the plan's input storage."""
    ly, me, _ = layer(rank)
    geom = ly._dsv41_b12x_next.geom
    m = x.shape[0]
    ids, w = routing(m, 7 + m)
    q, sf = geom.prequant_views(m)
    q.fill_(0x55)
    sf.fill_(0x55)
    tb.apply(ly, me, x, ids, w)
    torch.cuda.synchronize()
    return q.clone(), sf.clone()


def _cmp(tag, a, b):
    if torch.equal(a, b):
        return True
    d = (a != b)
    idx = d.nonzero()[:3].tolist()
    say(f"    {tag}: {int(d.sum())} of {d.numel()} bytes differ, first at {idx}")
    return False


def quant_case(name, x, shards=True):
    """b12x in-kernel == torch model everywhere; == FlashInfer on every block that is not a 'danger'
    block (FlashInfer scale byte 0 on a block with a nonzero value), the danger blocks all differ,
    and prefill_sp.danger_rows (FlashInfer scale byte 0 anywhere in the row) flags every row where
    they differ."""
    m = x.shape[0]
    q_b, s_b = b12x_kernel(x)
    q_f, s_f = fi_quant(x)
    q_t, s_t = b12x_torch(x)
    blk = x.view(m, KB, 32)
    mn, mx = torch.aminmax(blk, dim=-1)
    danger = (s_f == 0) & ((mx > 0) | (mn < 0))                       # [m, KB]
    diff = (s_b != s_f) | (q_b.view(m, KB, 32) != q_f.view(m, KB, 32)).any(-1)
    res = {"b12x==torch": _cmp(f"{name} data b12x vs torch", q_b, q_t) & _cmp(f"{name} scales b12x vs torch", s_b, s_t),
           "b12x==FI off danger": not bool((diff & ~danger).any()),
           "danger blocks all differ": bool(diff[danger].all()) if bool(danger.any()) else True,
           "danger_rows covers": not bool((diff.any(-1) & ~sp.danger_rows(x, s_f).squeeze(1).bool()).any())}
    if shards and m % W == 0:
        q_s, s_s = fi_quant_shards(x)
        res["FI shards==FI full"] = torch.equal(q_s, q_f) and torch.equal(s_s, s_f)
    say(f"  quant {name}: {res}; danger blocks {int(danger.sum())} (rows {int(danger.any(-1).sum())}), "
        f"rows flagged by danger_rows {int(sp.danger_rows(x, s_f).sum())}"
        + (f", max|x| there {float(torch.maximum(mx.float().abs(), mn.float().abs())[danger].max()):.3e}"
           if bool(danger.any()) else ""))
    return all(res.values())


def _all_bf16():
    v = torch.arange(-32768, 32768, dtype=torch.int32, device=DEV).to(torch.int16).view(BF)
    return v[torch.isfinite(v.float())]


def exhaustive_blocks():
    """Rows (160 blocks of 32 each) covering (1) every finite positive bf16 value as a block max,
    (2) for maxima at every scale byte, every finite bf16 element within 2^-20 of the max."""
    vals = _all_bf16()
    f = vals.float()
    pos = vals[f > 0]
    blocks = []
    # (1) every positive finite value as a block maximum (sign alternating), the rest scaled randomly below it
    g = torch.Generator(device=DEV).manual_seed(3)
    n = pos.numel()
    u = torch.rand(n, 31, device=DEV, generator=g) * 2 - 1
    rest = (pos.float()[:, None] * u).to(BF)
    rest = torch.where(rest.float().abs() > pos.float()[:, None], torch.zeros_like(rest), rest)
    sign = torch.where(torch.arange(n, device=DEV) % 2 == 0, 1.0, -1.0)
    blocks.append(torch.cat([(pos.float() * sign).to(BF)[:, None], rest], 1))
    # (2) element coverage per scale byte
    bits = pos.view(torch.int16).to(torch.int32)
    mant = bits & 0x7F
    maxima = pos[(mant == 0) | (mant == 1) | (mant == 0x60) | (mant == 0x7F) | ((bits >> 7) == 0)]
    fa = f.abs()
    for mx in maxima.float().tolist():
        sel = vals[fa <= mx]
        k = sel.numel()
        pad = (-k) % 31
        if pad:
            sel = torch.cat([sel, torch.zeros(pad, dtype=BF, device=DEV)])
        body = sel.view(-1, 31)
        head = torch.full((body.shape[0], 1), mx, dtype=BF, device=DEV)
        head[1::2] = -mx
        blocks.append(torch.cat([head, body], 1))
    allb = torch.cat(blocks)                        # [n_blocks, 32]
    nb = allb.shape[0]
    rows_ = (nb + KB - 1) // KB
    pad = rows_ * KB - nb
    if pad:
        allb = torch.cat([allb, torch.zeros(pad, 32, dtype=BF, device=DEV)])
    return allb.view(rows_, H), nb, maxima.numel()


def test_quant():
    ok = True
    for m in (4096, 3152, 2052, 2048, 1500):
        ok &= quant_case(f"act M={m} (shard {m // W})", rows(m, 11 + m))
    ok &= quant_case("randn M=2048", rows(2048, 5, kind="randn"))
    x, nb, nmax = exhaustive_blocks()
    for i, lo in enumerate(range(0, x.shape[0], 4096)):
        part = x[lo:lo + 4096]
        if part.shape[0] < 2048:            # keep the M64 prefill plans (and whole rows)
            part = torch.cat([part, rows(2048 - part.shape[0], 99)])
        ok &= quant_case(f"exhaustive part {i} ({part.shape[0]} rows)", part, shards=False)
        del part
    say(f"  quant exhaustive: {nb} blocks ({x.shape[0]} rows), {nmax} scale-ladder maxima x every finite "
        f"bf16 value <= each, {_all_bf16().numel()} finite bf16 values")
    del x
    torch.cuda.empty_cache()
    assert ok
    say("quant: b12x in-kernel == torch model everywhere, == FlashInfer (full and per shard) except exactly "
        "the danger blocks: OK")


# ------------------------------------------------------------------------------------ pre-quantized launch
def test_moe():
    ly, me, src = layer(0)
    st = ly._dsv41_b12x_next
    geom = st.geom
    ok = True
    for m in (4096, 3152, 2052, 2048):
        x = rows(m, 200 + m)
        ids, w = routing(m, 300 + m)
        ref = torch.empty(m, H, dtype=BF, device=DEV)
        geom.forward(st.impl, x, ids, w, ref)
        ref2 = torch.empty_like(ref)
        geom.forward(st.impl, x, ids, w, ref2)
        q, sf = geom.prequant_views(m)
        fq, fs = fi_quant_shards(x) if m % W == 0 else fi_quant(x)
        q.copy_(fq)
        sf.copy_(fs)
        calls = ad.PREQUANT_STATS["calls"]
        out = torch.empty_like(ref)
        geom.forward_prequant(st.impl, m, ids, w, out)
        torch.cuda.synchronize()
        untouched = torch.equal(q, fq) and torch.equal(sf, fs)
        same = torch.equal(out, ref)
        # through the fused function with a marker, as the engine dispatch delivers it
        marker = ad.prequant_target(ly, m)[0]
        pre = NS(_dsv41_b12x_rows=marker)
        xf = x.new_full((1, 1), float("nan")).expand(m, H)
        from sglang.srt.layers.moe.token_dispatcher.standard import StandardDispatchOutput
        from sglang.srt.layers.moe.topk import StandardTopKOutput
        d = StandardDispatchOutput(hidden_states=xf, hidden_states_scale=None,
                                   topk_output=StandardTopKOutput(topk_weights=w, topk_ids=ids, router_logits=None),
                                   hidden_states_pre_quant=pre)
        via = me.apply(ly, d).hidden_states
        same_via = torch.equal(via, ref)
        # negative control: one scale byte changed -> different output
        sf[m // 2, 7] += 1
        bad = torch.empty_like(ref)
        geom.forward_prequant(st.impl, m, ids, w, bad)
        reads = not torch.equal(bad, ref)
        rec = {"M": m, "cap": geom.cap_for(m), "prequant==bf16": same, "via fused fn": same_via,
               "rerun==": torch.equal(ref, ref2), "input untouched": untouched,
               "corrupted scale changes out": reads, "launches": ad.PREQUANT_STATS["calls"] - calls}
        say(f"  moe {rec}")
        ok &= same and same_via and untouched and reads and rec["launches"] == 3
        del x, ref, ref2, out, bad, via
    torch.cuda.empty_cache()
    assert ok
    say("moe: pre-quantized launch == bf16 launch bit for bit: OK")


# ------------------------------------------------------------------------------------ router
def _stub_engine_globals():
    from sglang.srt.models import deepseek_v2 as v2
    from sglang.srt.layers.moe import topk as topk_mod
    from sglang.srt.layers.moe import utils as moe_utils
    ex = NS(deterministic=NS(enable_deterministic_inference=False),
            moe=NS(enable_eplb=False, enable_waterfill=False, ep_num_redundant_experts=0))
    v2.get_exec = lambda: ex
    topk_mod.get_exec = lambda: ex
    be = tb._Backend("is_flashinfer_mxfp4")
    topk_mod.get_moe_runner_backend = lambda: be
    moe_utils.get_moe_runner_backend = lambda: be
    topk_mod.get_tp_group = lambda: NS(world_size=1)
    topk_mod.is_allocation_symmetric = lambda: False
    return v2


def make_gate_topk(n_exp=None):
    """MoEGate + TopK of layer 5 with the checkpoint router (first n_exp experts), built the way
    DeepseekV2MoE.__init__ builds them for DSV4.1 (ungrouped sqrtsoftplus, fp4 experts)."""
    v2 = _stub_engine_globals()
    from sglang.srt.layers.moe.topk import TopK
    n_exp = n_exp or NE
    gate = v2.MoEGate.__new__(v2.MoEGate)
    torch.nn.Module.__init__(gate)
    gate.is_deepseek_v4 = True
    gate.weight = torch.nn.Parameter(tb.ckpt("layers.5.ffn.gate.weight")[:n_exp].to(DEV, BF).contiguous(),
                                     requires_grad=False)
    gate.e_score_correction_bias = torch.nn.Parameter(
        tb.ckpt("layers.5.ffn.gate.bias")[:n_exp].to(DEV, torch.float32).contiguous(), requires_grad=False)
    gate.e_score_correction_bias_vl = torch.nn.Parameter(
        tb.ckpt("layers.5.ffn.gate.bias_vl")[:n_exp].to(DEV, torch.float32).contiguous(), requires_grad=False)
    gate.tiny_router_gemm_max_tokens = v2.tiny_router_gemm_max_tokens(
        num_experts=n_exp, hidden_size=H, weight_dtype=BF)
    topk = TopK(top_k=6, layer_id=5, renormalize=True, use_grouped_topk=False, num_fused_shared_experts=0,
                scoring_func="sqrtsoftplus", correction_bias=gate.e_score_correction_bias,
                routed_scaling_factor=1.5, apply_routed_scaling_factor_on_output=False, is_fp4_experts=True)
    return v2, gate, topk


def test_router():
    v2, gate, topk = make_gate_topk(384)
    moe = NS(gate=gate, topk=topk, config=NS(image_token_id=129264), is_hash=False)
    ok = True
    for m in (4096, 3152, 2052, 2048):
        x = rows(m, 500 + m)
        ids = torch.randint(0, 129280, (m,), device=DEV)
        ids[::97] = 129264                       # some image tokens take the VL bias
        full_l = gate(x, None)
        full_t = v2.vision_topk(moe, full_l, ids)
        s = m // W
        res = {}
        for mode in ("shard", "padded"):
            good = True
            for r in range(W):
                lo, hi = r * s, (r + 1) * s
                xs = x[lo:hi].contiguous()
                if mode == "shard":
                    l = gate(xs, None)
                else:
                    xp = torch.zeros_like(x)
                    xp[lo:hi] = xs
                    l = gate(xp, None)[lo:hi]
                t = v2.vision_topk(moe, l, ids[lo:hi])
                good &= torch.equal(l, full_l[lo:hi]) and torch.equal(t.topk_ids, full_t.topk_ids[lo:hi]) \
                    and torch.equal(t.topk_weights, full_t.topk_weights[lo:hi]) \
                    and (not hasattr(t, "packed_topk_ids")
                         or torch.equal(t.packed_topk_ids, full_t.packed_topk_ids[lo:hi]))
            res[mode] = good
        say(f"  router M={m} (shard {s}): logits+top-k rows equal {res}; top-k output {type(full_t).__name__}; "
            f"tiny-gemm max {gate.tiny_router_gemm_max_tokens}")
        ok &= res["shard"] or res["padded"]
        del x, full_l, full_t
    assert ok
    say("router: shard (or padded) routing == rows of the full-M routing: OK")


# ------------------------------------------------------------------------------------ 4-rank emulation
def test_emulate():
    import test_prefill_sp_gpu as tg
    v2, gate, topk = make_gate_topk(NE)
    from sglang.srt.layers.moe import mega_moe
    from sglang.srt.layers.moe.fused_moe_triton import layer as fmoe_mod
    from sglang.srt.layers.moe.token_dispatcher.standard import StandardDispatchOutput
    lin = tg.setup()["fp8_utils"].flashinfer_mxfp8_blockscaled_linear     # routed to b12x as production
    mega_moe.should_use_mega_moe = lambda *a, **k: False
    fmoe_mod.is_in_tc_piecewise_cuda_graph = lambda: False
    fmoe_mod.is_allocation_symmetric = lambda: False
    fmoe_mod.get_tp_group = lambda: NS(world_size=1)
    shells = [layer(r)[0] for r in range(W)]
    for sh in shells:
        sh._use_ascend_fuseep = False
        sh._dwdp_bound = False
        sh.should_fuse_routed_scaling_factor_in_topk = False      # as the TopK above (scale applied after)
        sh.dispatcher = NS(
            dispatch=lambda hidden_states, topk_output: StandardDispatchOutput(
                hidden_states=hidden_states, hidden_states_scale=None, topk_output=topk_output),
            combine=lambda combine_input: combine_input[0])
    cur = {}
    G = dict(v2.__dict__)
    G.update(
        get_forward=lambda: tg.FORWARD, get_is_capture_mode=lambda: False, is_in_breakable_cuda_graph=lambda: False,
        get_exec=v2.get_exec, should_skip_post_experts_all_reduce=lambda **kw: tg.FORWARD.mlp_reduce_scatter,
        tensor_model_parallel_all_reduce=lambda t: cur["group"].all_reduce(t))

    def rebind(fn):
        return types.FunctionType(fn.__code__, G, fn.__name__, fn.__defaults__, fn.__closure__)

    class Experts:
        """FusedMoE of the current rank (the real FusedMoE.forward on that rank's shell)."""

        def __getattr__(self, name):
            return getattr(shells[cur["group"].rank_in_group], name)

        def __call__(self, hidden_states, topk_output, pre_quant_input=None):
            sh = shells[cur["group"].rank_in_group]
            return fmoe_mod.FusedMoE.forward(sh, hidden_states, topk_output, pre_quant_input=pre_quant_input)

    class Shared(torch.nn.Module):
        """TP-split shared expert: MXFP8 gate_up (b12x dense, pre-quantized tuple or bf16), SwiGLU, down."""

        def __init__(self, seed):
            super().__init__()
            self.gu = [tg._Fp8Linear(2 * 576, H, seed + r) for r in range(W)]
            g = torch.Generator(device=DEV).manual_seed(seed + 50)
            self.down = (torch.randn(W, 576, H, device=DEV, generator=g) / 24).to(BF)
            self.gate_up_proj = NS(use_intel_amx_backend=False)
            self.tuple_calls = 0

        def forward(self, x, forward_batch=None, gemm_output_zero_allocator=None, gateup_pre_quant=None):
            r = cur["group"].rank_in_group
            p = self.gu[r]
            if gateup_pre_quant is not None:
                self.tuple_calls += 1
                gu = lin(gateup_pre_quant[0], p.w, p.sf, input_scale=gateup_pre_quant[1], backend="cutlass")
            else:
                gu = lin(x, p.w, p.sf, backend="cutlass")
            a, b = gu.float().chunk(2, dim=-1)
            h = (torch.nn.functional.silu(a.clamp(max=10.0)) * b.clamp(-10.0, 10.0)).to(BF)
            return h @ self.down[r]

    class Moe8(torch.nn.Module):
        def __init__(self, seed, layer_id):
            super().__init__()
            self.gate, self.topk = gate, topk
            self.shared_experts = Shared(seed)
            self.experts = Experts()
            self.config = NS(image_token_id=129264)
            self.tp_size, self.layer_id, self.is_nextn, self.is_hash = W, layer_id, False, False
            self._enable_a2a_moe = self._shared_expert_tp1 = self._fuse_shared_experts_inside_sbo = False
            self.num_fused_shared_experts = 0
            self.alt_stream = None
            self.routed_scaling_factor = 1.5

        def _can_dual_stream_graph(self, hidden_states):
            return False

        def _maybe_quant_moe_input_once(self, hidden_states):
            return None

    Moe8.forward = sp._make_moe_forward(rebind(v2.DeepseekV2MoE.forward))
    Moe8.forward_normal = rebind(v2.DeepseekV2MoE.forward_normal)
    Moe8._forward_shared_experts = rebind(v2.DeepseekV2MoE._forward_shared_experts)

    class Model8m(tg.Model):
        """Layer 0: full-row layer (stand-in attention, real-engine MoE). Layer 1: bounded-replay tail."""

        def __init__(self, group, seed):
            self.pp_group = types.SimpleNamespace(world_size=1)
            self.hidden_size, self.hc_mult, self.hc_pre_from_prev_sublayer = H, tg.HC, True
            self.start_layer, self.end_layer, self.late_layer_start = 0, 2, 1
            self.config = tg.Layer.config
            self.layers = [tg.Layer(seed + 10 * i, i, tg.Attn(group, seed + 10 * i + 3), Moe8(seed + 10 * i + 4, 5),
                                    None) for i in range(2)]
            self.dspark_layers_to_capture = [0]

        def engram_hasher(self, input_ids, fb):
            return None

    Model8m.engram_hasher = None
    FB8 = NS(forward_mode=tg.FB.forward_mode, num_token_non_padded=None)
    sp._agree_min = lambda p, local: bool(min(int(v) for v in p.group._exchange(
        torch.tensor([1 if local else 0], device=DEV))))

    same = lambda a, b: tg._eq(a[0], b[0]) and tg._eq(a[1], b[1]) and tg._eq(tuple(a[2]), tuple(b[2]))  # noqa: E731

    def arm(m, lens, fp8_moe, exact, seed=9, debug=False, force_danger=False):
        """Two chunks of the same size through the loop: the first decides (self-check), the
        second runs the steady-state path. Returns the second chunk's outputs."""
        tg._bind_layer_methods()
        s0 = tg.setup()
        if debug:
            tg.Layer._hc_mix_stats = sp._make_hc_mix_stats(s0["stats"])
        tail = tg._tail(lens)
        g = torch.Generator(device=DEV).manual_seed(seed)
        R0 = torch.randn(m, tg.HC, H, device=DEV, generator=g, dtype=BF) * 0.5
        ids = torch.randint(0, 129280, (m,), device=DEV, generator=g)
        ids[::53] = 129264
        pos = torch.arange(m, device=DEV)
        group = tg.ThreadGroup(W)
        cur["group"] = group
        model = Model8m(group, seed)
        sp._M["v4"] = tg._engine_ns(group, tg._Backend(tail))
        sp._M["v2"] = types.SimpleNamespace(DeepseekV2MoE=Moe8, vision_topk=v2.vision_topk)
        sp._M["linear"] = types.SimpleNamespace(RowParallelLinear=tg.RowParallel)
        sp._STATIC["checked"] = False
        sp._MOE8.clear()
        old = sp.MODE, sp.EXACT, sp.FP8, sp.FP8_MOE

        def body(r):
            res = []
            for chunk in range(2):
                forced[threading.get_ident()] = force_danger and chunk == 1
                p = sp._plan(model, R0, FB8)
                aux = []
                d = sp._Dbg(chunk + 1, m, p, r) if debug and chunk == 1 else None
                sp._ctx.dbg = d
                try:
                    hs, pre, _ = sp._sp_forward_layers(model, p, pos, R0, FB8, ids, ids, True, aux)
                finally:
                    sp._ctx.dbg = None
                if d is not None:
                    sp._debug_end(d, (hs, pre))
                res.append((hs, pre, aux, (d.checked, list(d.bad)) if d is not None else None))
            assert same(res[0], res[1]), "second chunk differs from the first"
            return res[1]

        sp.MODE, sp.EXACT, sp.FP8, sp.FP8_MOE = "shard", exact, False, fp8_moe
        dbg_old = set(sp.DEBUG)
        sp.DEBUG.clear()
        if debug:
            sp.DEBUG.update({"compare", "fp"})
        danger_fn = sp.danger_rows
        forced = {}
        if force_danger:            # the steady-state chunk sees a tiny-block row: bf16 for that layer
            sp.danger_rows = lambda x, sf: (torch.ones((x.shape[0], 1), dtype=torch.int32, device=x.device)
                                            if forced.get(threading.get_ident()) else danger_fn(x, sf))
        calls, dz = ad.PREQUANT_STATS["calls"], sp._MOE8_DANGER[0]
        try:
            out = group.run(body)
        finally:
            sp.MODE, sp.EXACT, sp.FP8, sp.FP8_MOE = old
            sp.danger_rows = danger_fn
            sp.DEBUG.clear()
            sp.DEBUG.update(dbg_old)
            tg._bind_layer_methods()
        info = {"prequant launches": ad.PREQUANT_STATS["calls"] - calls, "check": {
            k: (v["ok"], v["router"]) for k, v in sp._MOE8.items()},
            "shared tuple calls": [l.mlp.shared_experts.tuple_calls for l in model.layers],
            "danger fallbacks": sp._MOE8_DANGER[0] - dz}
        if debug:
            info["debug checked/bad"] = [o[3] for o in out]
        del model
        torch.cuda.empty_cache()
        return out, info

    ok = True
    parts = os.environ.get("EMU_PARTS", "main,danger,debug").split(",")
    # 4 ranks in one process hold 4x the per-rank activations (and the first-chunk check keeps the
    # bf16 reference, the fp8 output and the byte copies): 4096 rows do not fit 2 GB here; the
    # 4096-row plan itself is covered by `moe` (same kernels, one rank)
    cases = [(int(v.split(":")[0]), [int(t) for t in v.split(":")[1].split("/")])
             for v in os.environ.get("EMU_CASES", "2052:513/1/1025/513,2048:2048,3152:3152").split(",")]
    for m, lens in cases if "main" in parts else ():
        for exact in (True, False):
            s1, i1 = arm(m, lens, False, exact)
            s2, i2 = arm(m, lens, True, exact)
            per = [same(s1[r], s2[r]) for r in range(W)]
            say(f"  emulate M={m} {'exact' if exact else 'fast '}: stage2b == stage1 per rank {per}; {i2}")
            # 2 chunks x 1 full-row layer x 4 ranks: the check chunk and the steady-state chunk
            ok &= all(per) and all(v[0] for v in i2["check"].values()) and i2["prequant launches"] == 2 * W
    m, lens = cases[0]
    s1, _ = arm(m, lens, False, True)
    if "danger" in parts:
        s3, i3 = arm(m, lens, True, True, force_danger=True)
        per = [same(s1[r], s3[r]) for r in range(W)]
        say(f"  emulate M={m} exact, tiny-block row forced in the steady-state chunk: == stage1 per rank {per}; {i3}")
        ok &= all(per) and i3["danger fallbacks"] == W and i3["prequant launches"] == W
        del s3
    if "debug" not in parts:
        assert ok
        say("emulate: stage 2b == stage 1 on every rank: OK")
        return
    torch.cuda.empty_cache()
    s4, i4 = arm(m, lens, True, True, debug=True)
    per = [same(s1[r], s4[r]) for r in range(W)]
    bad = [o for o in i4["debug checked/bad"] if o[1]]
    say(f"  emulate M={m} exact, DSV41_PREFILL_SP_DEBUG=compare on the steady-state chunk: == stage1 per rank "
        f"{per}; checked/bad per rank {[(c, len(b)) for c, b in i4['debug checked/bad']]}")
    ok &= all(per) and not bad
    assert ok
    say("emulate: stage 2b == stage 1 on every rank: OK")


if __name__ == "__main__":
    which = sys.argv[1:] or ["drift", "quant", "moe", "router", "emulate"]
    say(f"torch {torch.__version__}, {torch.cuda.get_device_name(0)}, b12x_next patch "
        f"{open(os.path.join(os.path.dirname(ad._b12x()['pkg'].__file__), 'SOURCE_PATCH')).read().strip()}, "
        f"prequant runtime {ad._B.get('impl') is not None}, experts {NE}")
    t0 = time.time()
    for name in which:
        globals()[f"test_{name}"]()
    say(f"test_prefill_sp_moe8_gpu: ok ({time.time() - t0:.0f} s, peak "
        f"{torch.cuda.max_memory_allocated() / 2**30:.2f} GiB)")
