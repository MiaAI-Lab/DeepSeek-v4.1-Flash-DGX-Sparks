"""Keep the FlashInfer autotune cache across boots under expert parallelism. Default OFF.

DSV41_AUTOTUNE_KEEP=1. The engine's entry gate (_drop_diverged_autotune_cache) digests each rank's
whole cache file and drops every cache when the digests differ. Under EP each rank tunes its own
MoE shapes, so the files can never agree: the cache is deleted on every boot and the fused-MoE
tactics are re-drawn from timing noise (sglang#40320, fix proposed in sglang#40420).

Here the digest covers only the load decision, as in #40420, plus one guard of our own: a sidecar
fingerprint of the launch (argv and the SGLANG_/DSV41_/SPARK_/B12X_/NCCL_ environment) written next
to the cache after tuning. A cache is kept only when its sidecar matches the current launch, so a
config change cannot leave one rank with a partial hit (a hit skips a profile, and a skipped
profile on one rank desynchronises the timing reduction).
"""
import hashlib
import json
import os
import sys
from pathlib import Path

ENABLED = os.environ.get("DSV41_AUTOTUNE_KEEP", "0").strip() not in ("0", "", "off", "false")
_PREFIXES = ("SGLANG_", "DSV41_", "SPARK_", "B12X_", "NCCL_")
# Switches that change neither the GEMM shapes nor the kernels tuned: A/B arms over them share tactics.
# ./start.sh keeps this set. The extra names are volatile only on start-tp4.sh: they are
# not set on the TP3 profile, and treating them as volatile there would change nothing,
# but a TP3 boot must keep the fingerprint this file had before the TP4 line.
_VOLATILE = ("DSV41_DRAFT_CAPTURE", "DSV41_DRAFT_CAPTURE_OUT", "DSV41_DRAFT_CAPTURE_TRIGGER",
             "DSV41_DRAFT_CAPTURE_MAX_GIB", "DSV41_VERIFY_CAP", "DSV41_VERIFY_CAP_MIN", "DSV41_DRAFT_TAU",
             "DSV41_BLOCK_VERIFY", "DSV41_FOLDED_FENCE", "DSV41_AUTOTUNE_KEEP",
             # TP3 additions: neither touches a FlashInfer-tuned op (the draft LM head is a Triton
             # kernel beside a cuBLAS bf16 GEMM; prefetch only forks a stream around the Engram lookup)
             "DSV41_DRAFT_HEAD_FP8", "DSV41_ENGRAM_PREFETCH", "DSV41_ENGRAM_PREFETCH_CHECK",
             # set by the launcher to a fresh timestamp on every boot: fingerprinting it made every
             # sidecar stale, so nothing was ever reused on dev-dsv41 (seen on TP3 2026-09-24)
             "SGLANG_RUN_ID", "DSV41_WO_A_W8_DRAFT")
# Switches that change neither the tuned MoE shapes nor their kernels. On the TP4 launcher,
# toggling one must reuse the tactics, or every A/B boot re-draws them (which moves the numerics).
_VOLATILE_TP4 = ("DSV41_REPLICATED_SPLIT", "DSV41_REPLICATED_SPLIT_MAX_M", "DSV41_DRAFT_MAIN_PROJ_SPLIT",
                 "DSV41_DRAFT_MAIN_PROJ_MAX_M", "DSV41_ROCE_GATHER", "DSV41_ROUTER_LIVE", "DSV41_DYN_SHARED",
                 "DSV41_DYN_SHARED_RATIO", "DSV41_DYN_SHARED_FIXED", "DSV41_DYN_SHARED_MAX_M",
                 "DSV41_DYN_SHARED_SERIAL", "DSV41_VERIFY_CAP_LOG", "DSV41_VERIFY_CAP_LOG_FEATURES",
                 "DSV41_ROUTE_STATS", "DSV41_ROUTE_RING", "DSV41_DRAFT_TAU_POS", "DSV41_ENGRAM_DRM_NODE",
                 "DSV41_ENGRAM_DRM_MIB", "DSV41_ENGRAM_DRM_LAYER", "DSV41_DRAFT_HEAD_FP8_IMPL")


def _volatile():
    if os.environ.get("DSV41_LAUNCHER") == "tp4":
        return _VOLATILE + _VOLATILE_TP4
    return _VOLATILE


def launch_fingerprint() -> str:
    volatile = _volatile()
    env = {k: v for k, v in os.environ.items() if k.startswith(_PREFIXES) and k not in volatile}
    if os.environ.get("DSV41_LAUNCHER") == "tp4":
        # In-boot A/B (adapter/ab_variant.py, test only, TP4 launcher) unions some gates into
        # os.environ; fingerprint the base config it recorded instead, so an A/B boot keys like the
        # boot of its base env.
        env = {k: v for k, v in env.items() if not k.startswith("DSV41_AB_")}
        for k, v in json.loads(os.environ.get("DSV41_AB_BASE") or "{}").items():
            if k in volatile:
                continue
            if v is None:
                env.pop(k, None)
            else:
                env[k] = v
    payload = {"argv": sys.argv, "env": env}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def sidecar(cache_path: Path) -> Path:
    return cache_path.with_suffix(".launch")


def install(mod):
    """sglang.srt.model_executor.runner.flashinfer_autotune"""
    if not ENABLED:
        return
    for name in ("_autotune_cache_digest", "flashinfer_autotune_context"):
        if not hasattr(mod, name):
            raise RuntimeError(f"DSV41_AUTOTUNE_KEEP: {name} is gone; engine drifted")
    if getattr(mod, "_dsv41_autotune_keep", False):
        return
    mod._dsv41_autotune_keep = True
    orig_ctx = mod.flashinfer_autotune_context
    fp = launch_fingerprint()
    # start-tp4.sh deletes a mismatched cache inside the digest. ./start.sh keeps the
    # previous behaviour: the digest only reports "", and the context drops the file.
    tp4 = os.environ.get("DSV41_LAUNCHER") == "tp4"

    def _autotune_cache_digest(cache_path, env):
        cache_path = Path(cache_path)
        if not cache_path.is_file():
            return ""
        try:
            configs = json.loads(cache_path.read_text())
            same_launch = sidecar(cache_path).read_text().strip() == fp
        except (OSError, ValueError):
            if not tp4:
                return ""
            configs, same_launch = None, False
        if not isinstance(configs, dict) or not same_launch:
            # a different launch re-tunes: an empty digest alone would still let the stock code
            # load the old file when every rank reports "" (they agree)
            if tp4:
                cache_path.unlink(missing_ok=True)
            return ""
        stamp = configs.get("_metadata")
        if not isinstance(stamp, dict):
            stamp = None
        # fp itself differs between ranks (node rank, per-node env); only "same launch as when
        # this rank saved" enters the cross-rank digest.
        payload = {"loadable": True, "stamp": stamp, "env": env}
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()

    import contextlib

    @contextlib.contextmanager
    def flashinfer_autotune_context(model_runner, *args, **kwargs):
        cache_path = Path(mod.flashinfer_autotune_cache_path(model_runner))
        before = _autotune_cache_digest(cache_path, {}) != ""
        if not tp4 and not before and cache_path.is_file():
            # No sidecar for this launch. When no rank has one, every digest is "" and the stock gate
            # sees agreement, so FlashInfer would still load each rank's stale file (the cache key
            # omits block size, graph sizes and adapter switches). Tune from scratch instead.
            cache_path.unlink(missing_ok=True)
        with orig_ctx(model_runner, *args, **kwargs):
            yield
        try:
            if cache_path.is_file():
                sidecar(cache_path).write_text(fp + "\n")
        except OSError:
            pass
        print(f"[autotune_keep] {'reused' if before else 'tuned and saved'} {cache_path.parent.name}/{cache_path.name}",
              flush=True)

    mod._autotune_cache_digest = _autotune_cache_digest
    mod.flashinfer_autotune_context = flashinfer_autotune_context
    print("[autotune_keep] armed", flush=True)
