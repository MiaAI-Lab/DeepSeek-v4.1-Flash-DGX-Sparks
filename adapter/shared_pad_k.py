"""Pad the shared expert's down_proj K from 576 to 640 so the shape goes back to b12x.

DIAGNOSIS. At TP4 the shared expert's `down_proj` is N=5120, K=moe_intermediate_size/4
= 576. FlashInfer's b12x MXFP8 kernel rejects any K that is not a multiple of 128
(576 = 4.5 x 128), so this ONE shape in the whole model falls back to CUTLASS -- a
128x32x128 tile that pads M to 128 at decode (M=6), takes 96256 B of shared memory,
and therefore fits one block per SM at 25% occupancy. The boot log names it:

    DSV41 MXFP8 backend 'b12x' rejected N=5120 K=576 (b12x mm_mxfp8 requires the
    contraction dim K to be a multiple of 128 (one full BK128 tile). Got K=576.)

It is the only rejection in the model; `gate_up` has K=5120 and takes b12x normally.

MEASUREMENT (43 weight copies so L2 is cold, CUDA graphs captured once, interleaved
rounds, M=6):

    cutlass K=576  (production)  58.00 us/call   52.4 GB/s   2.494 ms/step
    b12x    K=640  (this patch)  17.24 us/call  196.0 GB/s   0.742 ms/step   -70.3%
    cutlass K=640  (control)     58.41 us/call   57.9 GB/s   2.512 ms/step    +0.7%

The third row is the important one: padding by itself buys nothing. The entire gain
comes from the padded shape being eligible for b12x.

LOSSLESS. Measured, not argued: b12x@640 against cutlass@576 is 0 of 30720 elements
different bitwise, max |difference| 0.000e+00. The scale code used for the padding
does not matter (checked 0, 127, 200) -- an e4m3 zero weight times any scale is zero.

ON THE HEADLINE NUMBER. 92% of this kernel overlaps the routed experts' grouped GEMM,
because the shared expert runs on alt_stream (deepseek_v2.py:1105). So 1.75 ms/step
saved on the kernel is an UPPER BOUND, not a forecast. What settles it is the step.
Measured on the live server at concurrency 1: DeviceGemmMxf disappears entirely
(43 calls x 2.391 ms -> 0), dense_blockscaled grows 1.49 ms, net -0.90 ms of a 61.3 ms
step = +1.47% predicted. Measured steady state (first run after each boot discarded,
because FlashInfer autotunes during it and that run reads several percent off):

    prose c=1   49.81 / 51.69  ->  50.59 / 52.06 tok/s   +1.14%
    code  c=1   93.80          ->  95.22        tok/s    +1.51%

and greedy output is identical on the live server, 4 of 4 prompts, across two boots each
way -- the same test reports 1 of 4 for a change that does alter the output, so it
discriminates.

WHAT THE PATCH DOES
  1. Before `_prepare_block_fp8_as_mxfp8`, pads `weight` [5120,576]->[5120,640] with
     zeros and `weight_scale_inv` [160,18]->[160,20] with ones. SGLang then runs its
     own swizzle on the padded shape, so this uses EXACTLY the production path
     (block_fp8_scale_to_mxfp8_e8m0 + block_scale_interleave), not a copy of it.
  2. At call time, pads the input [M,576]->[M,640] through a PERSISTENT, once-zeroed
     buffer (a CUDA graph needs a stable address; the tail is never written).

Measurement showed that garbage or NaN in the tail of h does not corrupt the result,
because it multiplies against zero weights -- but that is a property of the MXFP8
quantizer rather than of the algebra, so the buffer is zeroed once at allocation
anyway.

DSV41_SHARED_PAD_K=1 enables it (OFF by default).
DSV41_SHARED_PAD_BUF_ROWS sets the input buffer capacity (default 256 rows; anything
larger takes a fallback that allocates on the fly, which is safe because prefill runs
eager).

Every shape or scale-layout mismatch RAISES instead of serving on: silent degradation
under 32x32 scales is exactly what produced garbage in the wo_a fp8 path.
"""
import logging
import os

import torch

logger = logging.getLogger(__name__)

HID = 5120          # hidden_size -- the N of the padded GEMM
SRC_K = 576         # moe_intermediate_size / TP4
TGT_K = 640         # 5 * 128, the smallest multiple of 128 above 576
BLOCK = 32          # weight_block_size from config.json

_bufs = {}          # weight data_ptr -> persistent, zeroed input buffer
_state = {"padded": 0, "logged_first": False, "logged_accept": False, "warned_big": False}


def _enabled() -> bool:
    return os.environ.get("DSV41_SHARED_PAD_K", "0").strip() not in ("0", "off", "false", "")


def _looks_like_shared_down(layer) -> bool:
    w = getattr(layer, "weight", None)
    return (w is not None and w.ndim == 2
            and tuple(w.shape) == (HID, SRC_K)
            and w.dtype is torch.float8_e4m3fn)


def _pad_weight(layer, prefix: str) -> None:
    """weight [5120,576]->[5120,640] with zeros, weight_scale_inv [160,18]->[160,20] with ones."""
    w = layer.weight
    s = getattr(layer, "weight_scale_inv", None)
    if s is None:
        raise RuntimeError(
            f"{prefix}: shared-expert down_proj without weight_scale_inv -- "
            "this is not the block-fp8 path this patch expects"
        )
    n, k = w.shape
    want = (n // BLOCK, k // BLOCK)
    if tuple(s.shape) != want:
        raise RuntimeError(
            f"{prefix}: weight {(n, k)} requires scales {want}, found {tuple(s.shape)} -- "
            f"the block layout is not {BLOCK}x{BLOCK}, refusing to pad"
        )
    if s.dtype is not torch.float32:
        raise RuntimeError(
            f"{prefix}: weight_scale_inv has dtype {s.dtype}, expected float32 "
            "(block-fp8 keeps scales in fp32 and only then encodes them to e8m0)"
        )
    sf = s.data.detach().float().contiguous()
    bits = sf.view(torch.int32)
    if not bool((((bits & 0x7FFFFF) == 0) & (sf > 0)).all()):
        raise RuntimeError(
            f"{prefix}: weight_scale_inv is not made of positive powers of two, so it "
            "is not the ue8m0 layout this patch requires"
        )

    nw = torch.zeros((n, TGT_K), dtype=w.dtype, device=w.device)
    nw[:, :SRC_K] = w.data
    ns = torch.ones((want[0], TGT_K // BLOCK), dtype=s.dtype, device=s.device)
    ns[:, :want[1]] = s.data

    keep = getattr(s, "format_ue8m0", None)
    w.data = nw
    s.data = ns
    if keep is not None:
        s.format_ue8m0 = keep
    layer._dsv41_shared_pad_k = True
    _state["padded"] += 1
    if not _state["logged_first"]:
        _state["logged_first"] = True
        logger.warning(
            "DSV41 shared-expert K padding: %s padded %s -> %s, scales %s -> %s "
            "(SGLang runs the swizzle on the padded shape)",
            prefix, (n, SRC_K), (n, TGT_K), want, tuple(ns.shape),
        )


def _pad_input(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """[M,576] -> [M,640] through a persistent zeroed buffer (stable address for CUDA graphs)."""
    x2 = x.view(-1, x.shape[-1])
    rows = x2.shape[0]
    cap = int(os.environ.get("DSV41_SHARED_PAD_BUF_ROWS", "256"))
    key = (weight.data_ptr(), x2.dtype, x2.device.index)
    buf = _bufs.get(key)
    if buf is None or buf.shape[0] < rows:
        if rows > cap:
            # Prefill is eager (prefill CUDA graph: disabled), so allocating here is
            # safe; the decode graph never reaches this branch.
            if not _state["warned_big"]:
                _state["warned_big"] = True
                logger.warning(
                    "DSV41 shared-expert K padding: %d rows exceed the %d-row buffer, "
                    "allocating on the fly. Expected for prefill, which runs eager; raise "
                    "DSV41_SHARED_PAD_BUF_ROWS only if this appears for a decode batch.",
                    rows, cap,
                )
            out = torch.zeros((rows, TGT_K), dtype=x2.dtype, device=x2.device)
            out[:, :SRC_K] = x2
            return out
        buf = torch.zeros((cap, TGT_K), dtype=x2.dtype, device=x2.device)
        _bufs[key] = buf
    buf[:rows, :SRC_K] = x2          # the tail is never written and stays zero
    return buf[:rows]


def install_fp8(module):
    """Hook Fp8LinearMethod._prepare_block_fp8_as_mxfp8 -- pad BEFORE the scales are
    prepared, so the swizzle already runs on the 640 shape."""
    if not _enabled():
        return
    cls = getattr(module, "Fp8LinearMethod", None)
    if cls is None or not hasattr(cls, "_prepare_block_fp8_as_mxfp8"):
        raise RuntimeError(
            "DSV41 shared-expert K padding: Fp8LinearMethod._prepare_block_fp8_as_mxfp8 "
            "does not exist -- the MXFP8 path differs from what this assumes, not arming"
        )
    original = cls._prepare_block_fp8_as_mxfp8

    def prepare(self, layer):
        prefix = str(getattr(layer, "prefix", layer.__class__.__name__))
        if "shared_experts.down_proj" in prefix and not _looks_like_shared_down(layer):
            w = getattr(layer, "weight", None)
            raise RuntimeError(
                f"{prefix}: looks like the shared-expert down_proj, but the weight is "
                f"{None if w is None else (tuple(w.shape), w.dtype)} where "
                f"{((HID, SRC_K), torch.float8_e4m3fn)} was expected -- the TP sharding "
                "or the weight layout changed, refusing to skip silently"
            )
        if _looks_like_shared_down(layer) and not getattr(layer, "_dsv41_shared_pad_k", False):
            _pad_weight(layer, prefix)
        return original(self, layer)

    cls._prepare_block_fp8_as_mxfp8 = prepare
    logger.warning(
        "DSV41 shared-expert K padding ARMED: down_proj %s -> %s, to take this shape "
        "off the CUTLASS fallback and back onto b12x", (HID, SRC_K), (HID, TGT_K)
    )


def install(module):
    """Hook fp8_utils.flashinfer_mxfp8_blockscaled_linear -- pad the input.
    Install AFTER mxfp8_b12x.install so this wraps its wrapper, not the original."""
    if not _enabled():
        return
    inner = module.flashinfer_mxfp8_blockscaled_linear

    def linear(input, weight, weight_scale, input_scale=None, bias=None,
               output_dtype=None, backend="cutlass", pin_tactic=False):
        if (weight.ndim == 2 and weight.shape[0] == HID and weight.shape[1] == TGT_K
                and input.shape[-1] == SRC_K):
            if input_scale is not None:
                raise RuntimeError(
                    "DSV41 shared-expert K padding: the input is already quantized "
                    "(input_scale != None) and the padding has to happen before "
                    "quantization -- refusing, because the result would be wrong"
                )
            input = _pad_input(input, weight)
            if not _state["logged_accept"]:
                _state["logged_accept"] = True
                out = inner(input, weight, weight_scale, input_scale, bias,
                            output_dtype, backend=backend, pin_tactic=pin_tactic)
                try:
                    from mxfp8_b12x import _REJECTED
                    accepted = (HID, TGT_K) not in _REJECTED
                except Exception:
                    accepted = None
                logger.warning(
                    "DSV41 shared-expert K padding: first call N=%d K=%d went through; "
                    "b12x accepted the shape: %s (%d layers padded)",
                    HID, TGT_K, accepted, _state["padded"],
                )
                return out
        return inner(input, weight, weight_scale, input_scale, bias,
                     output_dtype, backend=backend, pin_tactic=pin_tactic)

    module.flashinfer_mxfp8_blockscaled_linear = linear
