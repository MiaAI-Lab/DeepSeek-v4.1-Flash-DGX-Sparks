"""Per-step rank-0 broadcasts of the DSpark decode step: drop the provably redundant ones, merge or
audit the rest. Gated on DSV41_SPEC_SYNC_FREE (unset / 0 = off, the stock broadcasts).

A c1 decode step runs nine NCCL broadcasts (ring LL, 3 hops to the last rank; RoCEnante covers
all-reduce and all-gather only):
  * 5 in the draft graph, one per Markov step: SpecTpSyncSite.DSPARK_GRAPH_SAMPLE on the sampled
    token (folded sampling, SGLANG_DSPARK_FOLDED_SAMPLING=2), on the serial token chain;
  * 3 in the verify graph epilogue: DSPARK_ACCEPT_GRAPH on correct_len, bonus, cap_trim_lens;
  * 1 eager, verify_cap's live length (adapter/verify_cap.py).
Profile 2026-09-25 (4 ranks aligned on the RoCE all-reduce ends, counterfactual at the next
collective): ~89 us/step for the draft chain's 4 inner broadcasts, ~26 us for the last one +
verify_cap's, ~28 us for the epilogue's three: ~0.14 ms/step on the critical path.

Why each broadcast is redundant (nothing is taken on trust: "audit" measures it live):
  * draft: the step logits come out of the vocab all-gather (a byte copy, identical on every rank);
    temperatures and the greedy mask come from the request, identical. The only rank-local input is
    the Gumbel/exponential noise, drawn from each rank's torch generator. This adapter replaces that
    draw with a counter-based Philox stream keyed by (seed broadcast once at init, draft-sampler
    call counter, Markov step): every rank computes the same noise, hence the same token, so the
    broadcast carries nothing. Greedy rows ignore the noise: their tokens are bit-identical to stock.
    Sampled rows draw other (equally exact) random numbers: rejection sampling uses the corrected
    logits, not the noise, so the output distribution is unchanged.
  * accept: accept_greedy_triton(candidates, target logits, cutoff) - candidates are the draft
    tokens (identical, above), the target logits come out of the LM-head all-gather (identical), the
    cutoff is verify_cap's live length (broadcast, or identical when vcap is dropped too).
  * vcap: the confidence head reads the draft's final hidden state, which is replicated compute on
    the output of a fixed-rank-order RoCE all-reduce (bit-identical on every rank). It is the one
    link that relies on deterministic replicated kernels, so drop it only after an audit run.

DSV41_SPEC_SYNC_FREE is a comma list:
  draft   rank-invariant draft noise + skip DSPARK_GRAPH_SAMPLE / DSPARK_GRAPH_GREEDY
  accept  skip DSPARK_ACCEPT_GRAPH (the graph epilogue's three broadcasts)
  vcap    skip verify_cap's live-length broadcast
  merge   the epilogue's three broadcasts as one packed broadcast (when accept is not dropped)
  audit   keep every broadcast, count on device how often a rank's own value differed from rank
          0's (per site), log it every DSV41_SPEC_SYNC_FREE_AUDIT_EVERY steps; installs the
          rank-invariant noise so the draft site is audited as it would run without its broadcast
  all     draft,accept,vcap (1 / on: the same; merge is moot once accept is dropped)
In-boot A/B (adapter/ab_variant.py, test only): DSV41_SPEC_SYNC_FREE is a union-parts gate there
(parts draft, accept, vcap, merge): the noise is installed for the union, and each variant's graphs
keep or drop their broadcasts at capture (verify_cap's eager one per replayed variant). audit is
not switchable in-boot.
Failure to install (engine drift) raises: a rank that silently kept a broadcast its peers dropped
would deadlock the fleet.
"""
import inspect
import logging
import os

import torch

logger = logging.getLogger(__name__)

_SPEC = os.environ.get("DSV41_SPEC_SYNC_FREE", "").strip().lower()
_TOKENS = {t.strip() for t in _SPEC.replace(";", ",").split(",") if t.strip()} - {"0", "off", "false"}
if "all" in _TOKENS:
    _TOKENS |= {"draft", "accept", "vcap", "merge"}
_UNKNOWN = _TOKENS - {"draft", "accept", "vcap", "merge", "audit", "all", "1", "on"}
if _UNKNOWN:
    raise ValueError(f"DSV41_SPEC_SYNC_FREE: unknown token(s) {sorted(_UNKNOWN)}")
if _TOKENS & {"1", "on"}:
    _TOKENS |= {"draft", "accept", "vcap", "merge"}
ENABLED = bool(_TOKENS)
AUDIT = "audit" in _TOKENS
# audit keeps every broadcast; it only adds the comparison
SKIP_DRAFT = "draft" in _TOKENS and not AUDIT
SKIP_ACCEPT = "accept" in _TOKENS and not AUDIT
SKIP_VCAP = "vcap" in _TOKENS and not AUDIT
MERGE = "merge" in _TOKENS and not SKIP_ACCEPT
NOISE = "draft" in _TOKENS or AUDIT
AUDIT_EVERY = int(os.environ.get("DSV41_SPEC_SYNC_FREE_AUDIT_EVERY", "1024") or 1024)

SITES = ("draft", "accept", "vcap")
AB_PARTS = ("draft", "accept", "vcap", "merge")

try:  # DSV41_AB_VARIANTS in-boot A/B (test only): the env holds the union, each variant its own parts
    import ab_variant as _ab
except ImportError:
    _ab = None


def _live():
    """(skip draft, skip accept, skip vcap, merge) for the variant being captured or replayed."""
    if _ab is not None and _ab.ACTIVE and not AUDIT:
        p = _ab.parts("DSV41_SPEC_SYNC_FREE", AB_PARTS)
        return "draft" in p, "accept" in p, "vcap" in p, "merge" in p and "accept" not in p
    return SKIP_DRAFT, SKIP_ACCEPT, SKIP_VCAP, MERGE
_state = {"counters": None, "calls": 0, "last": None, "installed": set(), "pending": [],
          "merge_logged": False}


# ---------------------------------------------------------------- rank-invariant draft noise ----
try:
    import triton
    import triton.language as tl

    @triton.jit
    def _exp_noise_kernel(OUT, SEED, CTR, step, V, stride, BLOCK: tl.constexpr):
        """OUT[row, :V] = -log(U), U = Philox uniform keyed by (seed + 64 * counter + step, row * V + col)."""
        row = tl.program_id(0)
        cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
        mask = cols < V
        key = tl.load(SEED) + tl.load(CTR) * 64 + step
        u = tl.rand(key, row * V + cols)
        # u in [0, 1): u == 0 gives +inf, whose key 0 never wins the race (probability 2^-31)
        tl.store(OUT + row * stride + cols, -tl.log(u), mask=mask)
except Exception:                                    # pragma: no cover - no triton on the host
    triton = None
    _exp_noise_kernel = None


def fill_exp_noise(out, seed, ctr, step):
    """Exponential(1) noise in out[bs, V] from the (seed, counter, step) Philox stream; the same on
    every rank that holds the same seed and counter. out: fp32, rows contiguous."""
    assert out.dtype == torch.float32 and out.stride(-1) == 1 and 0 <= step < 64
    bs, v = out.shape
    if bs == 0:
        return out
    assert bs * v < 2 ** 31
    block = 4096
    _exp_noise_kernel[(bs, triton.cdiv(v, block))](out, seed, ctr, step, v, out.stride(0), BLOCK=block,
                                                   num_warps=4)
    return out


class _NoiseSlice:
    """What `exp_noise[:bs]` returns: `.exponential_()` fills it from the rank-invariant stream."""

    def __init__(self, owner, view):
        self._owner, self._view = owner, view

    def exponential_(self, *a, **kw):
        if a or kw:
            raise RuntimeError("DSV41_SPEC_SYNC_FREE: exp_noise.exponential_ called with arguments; engine drifted")
        o = self._owner
        step = o.step
        o.step += 1
        return fill_exp_noise(self._view, o.seed, o.ctr, step)

    def __getattr__(self, name):
        return getattr(self._view, name)


class RankInvariantNoise:
    """Stands in for DsparkDraftSampler.exp_noise. The sampler's __call__ reads `exp_noise[:bs]`
    and calls `.exponential_()` once per Markov step; the step index resets at every __call__ and
    the counter advances once per __call__ (captured into the draft graph, so once per replay)."""

    def __init__(self, buf, seed, ctr):
        self.buf, self.seed, self.ctr, self.step = buf, seed, ctr, 0

    def __getitem__(self, idx):
        return _NoiseSlice(self, self.buf[idx])

    def __getattr__(self, name):
        return getattr(self.buf, name)


def _call_uses_exp_noise(cls):
    src = inspect.getsource(cls.__call__)
    return src.count("self.exp_noise[:bs].exponential_()") == 1


def _seed_tensor(device, tp_sync):
    base = os.environ.get("DSV41_SPEC_SYNC_FREE_SEED", "").strip()
    value = int(base, 0) if base else (torch.cuda.initial_seed() & ((1 << 47) - 1))
    seed = torch.tensor([value], dtype=torch.int64, device=device)
    group = getattr(tp_sync, "_tp_group", None)
    if group is not None and group.world_size > 1:
        # once, at sampler construction (eager, every rank, before capture): rank 0's seed
        group.broadcast(seed, src=0)
    return seed


def install_sampler(module):
    """sglang.srt.speculative.dspark_components.dspark_draft_sampler"""
    if not ENABLED:
        return
    install_sync(module.SpecTpSync, module.SpecTpSyncSite)
    if not NOISE or "sampler" in _state["installed"]:
        return
    cls = module.DsparkDraftSampler
    if not _call_uses_exp_noise(cls):
        raise RuntimeError("DSV41_SPEC_SYNC_FREE: DsparkDraftSampler.__call__ no longer draws "
                           "self.exp_noise[:bs].exponential_(); engine drifted")
    if _exp_noise_kernel is None:
        raise RuntimeError("DSV41_SPEC_SYNC_FREE: triton is required for the draft noise")
    _state["installed"].add("sampler")
    orig_init, orig_call, orig_stage = cls.__init__, cls.__call__, cls.stage_sampling_params

    def __init__(self, *a, **kw):
        orig_init(self, *a, **kw)
        if self.exp_noise is not None:
            dev = self.exp_noise.device
            self.exp_noise = RankInvariantNoise(self.exp_noise, _seed_tensor(dev, self._tp_sync),
                                                torch.zeros(1, dtype=torch.int64, device=dev))
            if AUDIT:
                counters(dev)
            logger.info("DSV41_SPEC_SYNC_FREE: draft noise from the rank-invariant Philox stream")

    def __call__(self, *a, **kw):
        noise = self.exp_noise if isinstance(self.exp_noise, RankInvariantNoise) else None
        if noise is not None:
            noise.step = 0
        out = orig_call(self, *a, **kw)
        if noise is not None:
            noise.ctr.add_(1)
        return out

    def stage_sampling_params(self, *a, **kw):
        orig_stage(self, *a, **kw)
        if AUDIT:
            audit_tick()

    cls.__init__, cls.__call__, cls.stage_sampling_params = __init__, __call__, stage_sampling_params
    print(f"[spec_sync_free] armed ({_SPEC}): skip draft={SKIP_DRAFT} accept={SKIP_ACCEPT} "
          f"vcap={SKIP_VCAP} merge={MERGE} audit={AUDIT}", flush=True)


# ------------------------------------------------------------------------- the sync sites -----
def _capturing():
    return torch.cuda.is_available() and torch.cuda.is_current_stream_capturing()


def counters(device):
    c = _state["counters"]
    if c is None:
        assert not _capturing(), "spec_sync_free: counters must exist before graph capture"
        c = _state["counters"] = torch.zeros((len(SITES), 2), dtype=torch.int64, device=device)
    return c


def _audited_broadcast(group, values, site):
    """Broadcast from rank 0 and count, on device, whether this rank's own value differed."""
    c = _state["counters"]
    if c is None and not _capturing():
        c = counters(values.device)
    if c is None:
        group.broadcast(values, src=0)
        return values
    before = values.clone()
    group.broadcast(values, src=0)
    i = SITES.index(site)
    c[i, 0].add_((before != values).any().to(torch.int64))
    c[i, 1].add_(1)
    return values


def audit_tick(force=False):
    """Host readout every AUDIT_EVERY steps (one device sync; audit runs only)."""
    _state["calls"] += 1
    c = _state["counters"]
    if c is None or not (force or _state["calls"] % AUDIT_EVERY == 0):
        return None
    vals = c.tolist()
    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    diverged = any(v[0] for v in vals)
    if diverged or vals != _state["last"]:
        msg = " ".join(f"{s}={v[0]}/{v[1]}" for s, v in zip(SITES, vals))
        print(f"[spec_sync_free] audit rank={rank} mismatches/checks {msg}"
              + ("  DIVERGED: keep that site's broadcast" if diverged else ""), flush=True)
    _state["last"] = vals
    return vals


def tp_broadcast(group, values, site):
    """verify_cap's live-length broadcast, routed here when DSV41_SPEC_SYNC_FREE is set."""
    if AUDIT:
        return _audited_broadcast(group, values, site)
    if site == "vcap" and _live()[2]:
        return values
    group.broadcast(values, src=0)
    return values


def install_sync(sync_cls, site_enum):
    """Wrap SpecTpSync.sync (shared by the draft sampler and the verify epilogue)."""
    if not ENABLED or getattr(sync_cls, "_dsv41_sync_free", False):
        return
    sync_cls._dsv41_sync_free = True
    draft_sites = {site_enum.DSPARK_GRAPH_SAMPLE, site_enum.DSPARK_GRAPH_GREEDY}
    accept_site = site_enum.DSPARK_ACCEPT_GRAPH
    orig = sync_cls.sync

    def sync(self, site, values):
        if site not in self._sites:
            return values
        name = "draft" if site in draft_sites else "accept" if site == accept_site else None
        if name is None:
            return orig(self, site, values)
        if AUDIT:
            return _audited_broadcast(self._tp_group, values, name)
        skip_draft, skip_accept, _, merge = _live()
        if name == "draft" and skip_draft:
            return values
        if name == "accept":
            if skip_accept:
                return values
            if merge:
                return _merged_accept(self._tp_group, values)
        return orig(self, site, values)

    sync_cls.sync = sync


def _merged_accept(group, values):
    """The epilogue syncs correct_len, bonus, cap_trim_lens back to back and reads none of them
    before the third call (checked at install): hold the first two, broadcast all three packed."""
    pend = _state["pending"]
    pend.append(values)
    if len(pend) < 3:
        return values
    a, b, c = pend
    pend.clear()
    if not (a.shape == b.shape == c.shape and a.dim() == 1):
        raise RuntimeError(f"spec_sync_free merge: unexpected accept shapes {a.shape} {b.shape} {c.shape}")
    packed = torch.stack([a.to(torch.int64), b.to(torch.int64), c.to(torch.int64)])
    group.broadcast(packed, src=0)
    a.copy_(packed[0])
    b.copy_(packed[1])
    c.copy_(packed[2])
    if not _state["merge_logged"]:
        _state["merge_logged"] = True
        print("[spec_sync_free] epilogue accept broadcasts merged (3 -> 1)", flush=True)
    return values


def _accept_syncs_back_to_back(ep_cls):
    src = inspect.getsource(ep_cls._accept)
    lines = [ln.strip() for ln in src.splitlines()]
    want = [f"self._tp_sync.sync(SpecTpSyncSite.DSPARK_ACCEPT_GRAPH, {n})"
            for n in ("correct_len", "bonus", "cap_trim_lens")]
    for i in range(len(lines) - 2):
        if lines[i:i + 3] == want:
            return src.count("DSPARK_ACCEPT_GRAPH") == 3
    return False


def install_verify(module):
    """sglang.srt.speculative.dspark_components.dspark_verify: before verify_cap's install_verify,
    so the source check reads the engine's own _accept."""
    if not ENABLED:
        return
    install_sync(module.SpecTpSync, module.SpecTpSyncSite)
    if "verify" in _state["installed"]:
        return
    _state["installed"].add("verify")
    ep = module.DsparkVerifyEpilogue
    if "merge" in _TOKENS and not _accept_syncs_back_to_back(ep):
        raise RuntimeError("DSV41_SPEC_SYNC_FREE=merge: DsparkVerifyEpilogue._accept no longer syncs "
                           "correct_len, bonus, cap_trim_lens back to back; engine drifted")
    if AUDIT:
        orig_init = ep.__init__

        def __init__(self, *a, device=None, **kw):
            orig_init(self, *a, device=device, **kw)
            if device is not None:
                counters(torch.device(device) if not isinstance(device, torch.device) else device)

        ep.__init__ = __init__
