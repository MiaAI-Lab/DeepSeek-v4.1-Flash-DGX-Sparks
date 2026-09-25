"""Per-step verify length for DSpark without changing the verify layout. Default OFF.

The target verify runs the fixed [bs, 6] rectangle (anchor + 5 drafts). With DSV41_VERIFY_CAP set,
each request gets a live length L = 1 + k (k = drafts actually verified):

  * acceptance is capped at k drafts through the engine's own cutoff (CapCorrectLen + bonus from
    row k), so the committed tokens are exactly those of a verify with k drafts;
  * the router ids of the dead rows (k+1..5) are replaced by the anchor row's ids, so the dead rows
    add no experts to the per-layer set the MoE streams (the saving; MoE is bandwidth-bound);
  * dead rows are never committed (commit <= k + 1), and live rows only attend to earlier rows, so
    their wrong MoE output cannot reach a committed token.

k comes from DSV41_VERIFY_CAP:
  "1".."5"     fixed k for every step (5 = all live, measures the machinery's own cost)
  "conf:T"     k = number of leading drafts whose running product of the draft confidence head's
               per-position survival stays >= T (at least DSV41_VERIFY_CAP_MIN, default 1).
               The head's value for position i is computed from drafts before i only, so the
               cut is a stopping rule: it never looks at the token it drops (exact for sampling).
Sampled rows go through block_verify, which treats dead positions as q = 0, p(X) = 0.
"""
import os

import torch

SPEC = os.environ.get("DSV41_VERIFY_CAP", "").strip()
ENABLED = SPEC not in ("", "0", "off")
# verify rows per request = DSPARK block size + 1 (anchor); 6 for the stock block of 5
STRIDE = int(os.environ.get("DSV41_VERIFY_CAP_STRIDE") or int(os.environ.get("DSPARK_BLOCK_SIZE", "5")) + 1)
MAX_BS = 64
E_TARGET, K_TARGET = 384, 6

_state = {"live": None, "in_verify": False, "conf": None, "fixed": None, "thr": None,
          "kmin": int(os.environ.get("DSV41_VERIFY_CAP_MIN", "1")), "logged": False}
if ENABLED:
    if SPEC.startswith("conf:"):
        # one threshold for every draft, or one per draft 2..5 ("conf:0.25,0.25,0.03,0.06");
        # draft 1 is always verified (DSV41_VERIFY_CAP_MIN, default 1)
        vals = [float(v) for v in SPEC.split(":", 1)[1].split(",")]
        _state["thr"] = vals[0] if len(vals) == 1 else [0.0] + vals
    else:
        _state["fixed"] = int(SPEC)


def _capturing():
    return torch.cuda.is_available() and torch.cuda.is_current_stream_capturing()


def live_buf(device=None):
    buf = _state["live"]
    if buf is None:
        assert not _capturing(), "verify_cap: live buffer must exist before capture"
        buf = torch.full((MAX_BS,), STRIDE, dtype=torch.int64,
                         device=device if device is not None else torch.cuda.current_device())
        if _state["fixed"] is not None:
            buf.fill_(1 + _state["fixed"])
        _state["live"] = buf
    return buf


def cutoff(bs):
    return live_buf()[:bs] if ENABLED else None


def set_live_from_confidence(confidence, bs):
    """confidence [bs, 5] per-position survival (position i given drafts < i)."""
    if _state["thr"] is None or confidence is None:
        return
    buf = live_buf(confidence.device)
    _state["last_conf"] = confidence
    cum = torch.cumprod(confidence.float().clamp(0, 1), dim=1)
    thr = _state["thr"]
    if isinstance(thr, list):
        if _state.get("thr_t") is None or _state["thr_t"].device != cum.device:
            t = (thr + [thr[-1]] * cum.shape[1])[:cum.shape[1]]
            _state["thr_t"] = torch.tensor(t, dtype=torch.float32, device=cum.device)
        thr = _state["thr_t"]
    k = (cum >= thr).to(torch.int64).cumprod(dim=1).sum(dim=1).clamp(min=_state["kmin"], max=STRIDE - 1)
    buf[:bs].copy_(k + 1)
    share_live(buf, bs)


def share_live(buf, bs):
    """Rank 0's live lengths to every rank (also called by adapter/eager_glue.py's fused update)."""
    # Rank-invariant by construction: the head's input comes out of an all-reduce that may differ
    # in the last bits between ranks, and a live length that differs across ranks would give one
    # verify row two expert sets. Rank 0 decides, as for the engine's own accept (SpecTpSync).
    group = _state.get("tp_group")
    if group is None:
        from sglang.srt.distributed import get_tp_group
        group = _state["tp_group"] = get_tp_group()
    if group.world_size > 1:
        if (os.environ.get("DSV41_LAUNCHER") == "tp4"
                and os.environ.get("DSV41_SPEC_SYNC_FREE", "").strip() not in ("", "0", "off", "false")):
            # adapter/spec_sync_free.py: skipped with its vcap token, counted in audit mode
            from spec_sync_free import tp_broadcast
            tp_broadcast(group, buf[:bs], "vcap")
        else:
            group.broadcast(buf[:bs], src=0)


def _src_rows(m, device):
    """Row each verify row takes its router output from (itself if live, else its anchor).
    Computed once per forward and reused by all 43 layers (inside the graph: once per replay)."""
    src = _state.get("src")
    if src is not None and src.shape[0] == m:
        return src
    live = live_buf(device)
    rows = torch.arange(m, device=device)
    req, pos = rows // STRIDE, rows % STRIDE
    dead = pos >= live[req.clamp(max=MAX_BS - 1)]
    src = torch.where(dead, req * STRIDE, rows)
    _state["src"] = src
    if _capturing() and m not in _state.setdefault("captured", set()):
        _state["captured"].add(m)
        print(f"[verify_cap] remap captured into the verify graph for M={m}", flush=True)
    return src


try:
    import triton
    import triton.language as tl

    @triton.jit
    def _remap_kernel(W, I, LIVE, M, SW: tl.constexpr, SI: tl.constexpr, STRIDE_R: tl.constexpr,
                      K: tl.constexpr, BLOCK: tl.constexpr):
        r = tl.arange(0, BLOCK)
        rmask = r < M
        req = r // STRIDE_R
        pos = r % STRIDE_R
        live = tl.load(LIVE + req, rmask, STRIDE_R)
        dead = rmask & (pos >= live)
        anchor = req * STRIDE_R
        for k in tl.static_range(K):
            w = tl.load(W + anchor * SW + k, dead, 0.0)
            i = tl.load(I + anchor * SI + k, dead, 0)
            tl.store(W + r * SW + k, w, dead)
            tl.store(I + r * SI + k, i, dead)
except Exception:                                   # no triton: fall back to index_select
    _remap_kernel = None


def remap_dead_rows(weights, indices):
    """weights/indices [M, 6] for M = bs * 6 verify rows (request-major). In place when possible:
    one kernel per layer copies the anchor row's ids and weights over the dead rows."""
    m = indices.shape[0]
    if (_remap_kernel is not None and weights.is_contiguous() and indices.is_contiguous()
            and weights.shape[1] == K_TARGET and m <= 1024):
        if _capturing() and m not in _state.setdefault("captured", set()):
            _state["captured"].add(m)
            print(f"[verify_cap] in-place remap captured into the verify graph for M={m}", flush=True)
        _remap_kernel[(1,)](weights, indices, live_buf(indices.device), m, weights.stride(0),
                            indices.stride(0), STRIDE, K_TARGET, triton.next_power_of_2(m))
        return weights, indices
    src = _src_rows(m, indices.device)
    return weights.index_select(0, src), indices.index_select(0, src)


def install_gate(module):
    """sglang.kernels.ops.moe.moe_fused_gate"""
    if not ENABLED or getattr(module, "_dsv41_verify_cap", False):
        return
    module._dsv41_verify_cap = True
    original = module.moe_fused_gate
    fused = None
    if (os.environ.get("DSV41_LAUNCHER") == "tp4"
            and os.environ.get("DSV41_ROUTER_LIVE", "0").strip() not in ("0", "", "off", "false")):
        # DSV41_ROUTER_LIVE: the router loads the anchor's scores for dead rows (no remap launch)
        from router_live import build
        fused = build(module)
        print("[verify_cap] dead-row remap folded into the router kernel", flush=True)

    def wrapped(scores, bias, topk, *a, **kw):
        m = scores.shape[0] if scores.dim() == 2 else 0
        verify = (_state["in_verify"] and scores.shape[-1] == E_TARGET and topk == K_TARGET
                  and m and m % STRIDE == 0 and m // STRIDE <= MAX_BS)
        if verify and fused is not None:
            return fused(scores, bias, topk, *a, live=live_buf(scores.device), live_stride=STRIDE, **kw)
        weights, indices = original(scores, bias, topk, *a, **kw)
        if verify:
            return remap_dead_rows(weights, indices)
        return weights, indices

    module.moe_fused_gate = wrapped
    print(f"[verify_cap] armed ({SPEC}): dead verify rows route to the anchor's experts", flush=True)


def install_model(dsv4_module):
    """sglang.srt.models.deepseek_v4: mark target-verify forwards (eager and graph capture)."""
    if not ENABLED:
        return
    cls = dsv4_module.DeepseekV4ForCausalLM
    if getattr(cls, "_dsv41_verify_cap", False):
        return
    cls._dsv41_verify_cap = True
    orig = cls.forward

    def forward(self, input_ids, positions, forward_batch, *a, **kw):
        if _state["live"] is None and not _capturing():
            live_buf(input_ids.device)
        prev = _state["in_verify"]
        _state["src"] = None
        _state["in_verify"] = (type(self).__name__ == "DeepseekV4ForCausalLM"
                               and forward_batch.forward_mode.is_target_verify())
        try:
            return orig(self, input_ids, positions, forward_batch, *a, **kw)
        finally:
            _state["in_verify"] = prev

    cls.forward = forward


class _Cut:
    def __init__(self, verify_lens):
        self.verify_lens = verify_lens


def install_verify(verify_module):
    """sglang.srt.speculative.dspark_components.dspark_verify"""
    if not ENABLED:
        return
    ep = verify_module.DsparkVerifyEpilogue
    if getattr(ep, "_dsv41_verify_cap", False):
        return
    ep._dsv41_verify_cap = True
    orig_accept = ep._accept

    def _accept(self, *, candidates, logits, draft_tokens, seq_lens, cutoff_verify_lens=None):
        if cutoff_verify_lens is None:
            cutoff_verify_lens = cutoff(candidates.shape[0])
        return orig_accept(self, candidates=candidates, logits=logits, draft_tokens=draft_tokens,
                           seq_lens=seq_lens, cutoff_verify_lens=cutoff_verify_lens)

    ep._accept = _accept

    orig_adt = verify_module.accept_draft_tokens

    def accept_draft_tokens(*, candidates, cutoff_layout=None, **kw):
        if cutoff_layout is None:
            cutoff_layout = _Cut(cutoff(candidates.shape[0]))
        return orig_adt(candidates=candidates, cutoff_layout=cutoff_layout, **kw)

    verify_module.accept_draft_tokens = accept_draft_tokens

    ex = verify_module.TargetVerifyExecutor
    orig_run = ex.run_non_compact

    def run_non_compact(self, *, batch, draft_input, verify_ids_2d, **kw):
        bs = verify_ids_2d.shape[0]
        if _state["thr"] is not None:
            set_live_from_confidence(_state["conf"], bs)
            if not _state["logged"]:
                _state["logged"] = True
                print(f"[verify_cap] confidence {'present' if _state['conf'] is not None else 'MISSING'}",
                      flush=True)
        _state["conf"] = None
        return orig_run(self, batch=batch, draft_input=draft_input, verify_ids_2d=verify_ids_2d, **kw)

    ex.run_non_compact = run_non_compact

    # DSV41_VERIFY_CAP_LOG=/state/vcap-log.bin (rank 0): per verify step and request, the confidence
    # vector, the live length used and the accepted drafts. Costs a device sync per step: measurement only.
    log_path = os.environ.get("DSV41_VERIFY_CAP_LOG", "")
    if log_path and _state["thr"] is not None:
        orig_aaf = ex.accept_and_finalize
        _state["log"] = None

        def accept_and_finalize(self, *a, **kw):
            outs = orig_aaf(self, *a, **kw)
            conf = _state.get("last_conf")
            if conf is not None:
                import numpy as np
                if _state["log"] is None:
                    from sglang.srt.distributed import get_tp_group
                    _state["log"] = open(log_path, "ab") if get_tp_group().rank_in_group == 0 else False
                if _state["log"]:
                    bs = conf.shape[0]
                    rec = torch.cat([conf.float(), live_buf()[:bs].float().unsqueeze(1),
                                     outs.correct_len[:bs].float().unsqueeze(1)], dim=1)
                    _state["log"].write(rec.cpu().numpy().astype(np.float32).tobytes())
                    _state["log"].flush()
            return outs

        ex.accept_and_finalize = accept_and_finalize


def install_draft(draft_module):
    """sglang.srt.speculative.dspark_components.dspark_draft: keep a folded proposal's confidence."""
    if not ENABLED or _state["thr"] is None:
        return
    cls = draft_module.DraftBlockProposer
    orig = cls.propose

    def propose(self, *a, **kw):
        out = orig(self, *a, **kw)
        if getattr(out, "confidence", None) is not None:
            _state["conf"] = out.confidence
        return out

    cls.propose = propose


def install_planner(planner_module):
    """sglang.srt.speculative.dspark_components.dspark_planner: keep the step's confidence."""
    if not ENABLED or _state["thr"] is None:
        return
    cls = planner_module.DSparkVerifyPlanner
    orig = cls.compute_confidence_tensor

    def compute_confidence_tensor(self, **kw):
        out = orig(self, **kw)
        _state["conf"] = out
        return out

    cls.compute_confidence_tensor = compute_confidence_tensor


def install_dspark(dspark_module):
    """sglang.srt.models.deepseek_v4_dspark: the engine builds the draft's confidence head only in
    the ragged-verify modes; build it in static mode too when the conf policy needs it."""
    if not ENABLED or _state["thr"] is None:
        return
    orig = dspark_module.build_dspark_v4_confidence_head
    mode_fn = dspark_module.read_ragged_verify_mode
    from sglang.srt.speculative.ragged_verify import RaggedVerifyMode

    def build(*a, **kw):
        dspark_module.read_ragged_verify_mode = lambda: RaggedVerifyMode.CAP_ACCEPT
        try:
            head = orig(*a, **kw)
        finally:
            dspark_module.read_ragged_verify_mode = mode_fn
        print(f"[verify_cap] draft confidence head built: {head is not None}", flush=True)
        return head

    dspark_module.build_dspark_v4_confidence_head = build
