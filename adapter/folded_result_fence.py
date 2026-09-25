"""Detach folded DSpark verify results from the verify graph's persistent buffers. Default OFF.

DSV41_FOLDED_FENCE=1. On the folded path (all-greedy batches replayed from the verify CUDA graph),
accept_and_finalize returns views of DsparkVerifyEpilogue's persistent buffers (out_tokens_buf,
commit_lens_buf, ...). With overlap scheduling the scheduler copies them to host on its copy stream
while the next step already runs, and the next verify replay can overwrite those buffers before the
copy finishes (sglang#40919). Cloning them on the forward stream right after the verify, before
anything else is queued, gives the copy its own memory; later replays cannot touch it. Six tensors
of bs x 6 ints: negligible. Sampled batches take the eager path, which already allocates fresh
tensors, and are left alone.
"""
import os

ENABLED = os.environ.get("DSV41_FOLDED_FENCE", "0").strip() not in ("0", "", "off", "false")
_GLUE = "fence" in os.environ.get("DSV41_EAGER_GLUE", "") or os.environ.get("DSV41_EAGER_GLUE", "").strip().lower() in (
    "1", "on", "true", "all")
FIELDS = ("correct_len", "bonus", "cap_trim_lens", "commit_lens", "new_seq_lens", "out_tokens")


def install(module):
    """sglang.srt.speculative.dspark_components.dspark_verify"""
    if not ENABLED:
        return
    cls = getattr(module, "TargetVerifyExecutor", None)
    outs_cls = getattr(module, "AcceptOuts", None)
    if cls is None or outs_cls is None or not hasattr(cls, "accept_and_finalize"):
        raise RuntimeError("DSV41_FOLDED_FENCE: TargetVerifyExecutor.accept_and_finalize / AcceptOuts gone; engine drifted")
    if getattr(cls, "_dsv41_folded_fence", False):
        return
    cls._dsv41_folded_fence = True
    orig = cls.accept_and_finalize

    def accept_and_finalize(self, *a, **kw):
        outs = orig(self, *a, **kw)
        if not kw.get("folded_accept", False):
            return outs
        if _GLUE:
            # DSV41_EAGER_GLUE=fence (adapter/eager_glue.py): the six copies in one kernel
            from eager_glue import fused_clone
            fresh = fused_clone([getattr(outs, f) for f in FIELDS])
            if fresh is not None:
                return outs_cls(**dict(zip(FIELDS, fresh)))
        return outs_cls(**{f: getattr(outs, f).clone() for f in FIELDS})

    cls.accept_and_finalize = accept_and_finalize
    print("[folded_fence] folded DSpark results cloned off the persistent verify buffers", flush=True)
