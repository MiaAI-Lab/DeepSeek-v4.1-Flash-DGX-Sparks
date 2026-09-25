"""GPU check for adapter/replicated_split.py's window GEMMs (one GPU, <= 2 GB, in the image):

  docker run --rm --gpus all --network none -v <ds41>:/ds41:ro \
      -e PYTHONPATH=/ds41/adapter:/ds41/tests --entrypoint python3 <image> /ds41/tests/test_replicated_split_gpu.py

wqkv_a's shape (N=1792 = 14 tiles of 128, K=5120) on the production MXFP8 path (FlashInfer b12x
dense via mxfp8_b12x, weight scales 128x4-swizzled), four TP ranks emulated on one GPU:
  - every rank's decide() candidates, as the boot gate builds them: rank 2/3 windows (rows
    1024..1535 / 1280..1791) are bit-exact against the stock layer's columns;
  - window mode needs no pad: every rank's GEMM output is already [M, 512] (the old exact slices
    were [M, 384] on ranks 2/3, then F.pad = fill + copy before the gather);
  - the gathered parts assembled == the stock full-width output, bit for bit, window mode and the
    exact-slice fallback, at M = 1..96, for bf16 input and for a pre-quantized
    Mxfp8SwizzledInput (the decode path's fused-norm quantized input);
  - the proxies' weights are views of the stock weight (no extra memory).
"""
import os
import time
import types

os.environ.setdefault("DSV41_REPLICATED_SPLIT", "wqkv_a")
import torch  # noqa: E402

import replicated_split as rs  # noqa: E402

torch.cuda.set_per_process_memory_fraction(min(1.0, 2e9 / torch.cuda.get_device_properties(0).total_memory))
DEV = "cuda"
N, K, W = 1792, 5120, 4


def setup():
    import mxfp8_b12x
    from sglang.srt.layers.quantization import fp8_utils
    mxfp8_b12x.install(fp8_utils)
    return fp8_utils.flashinfer_mxfp8_blockscaled_linear


class Layer(torch.nn.Module):
    """A ReplicatedLinear's parameters as the block-FP8-as-MXFP8 method leaves them."""

    def __init__(self, seed):
        super().__init__()
        from flashinfer import block_scale_interleave
        g = torch.Generator(device=DEV).manual_seed(seed)
        self.weight = torch.nn.Parameter((torch.randn(N, K, device=DEV, generator=g) * 2).to(torch.float8_e4m3fn),
                                         requires_grad=False)
        su8 = (127 + torch.randint(-6, 1, (N, K // 32), device=DEV, generator=g)).to(torch.uint8)
        self.weight_scale_inv_swizzled = torch.nn.Parameter(block_scale_interleave(su8.contiguous()).contiguous(),
                                                            requires_grad=False)
        self.output_size = N
        self._dsv41_prefix = "model.layers.5.self_attn.wqkv_a"


LIN = []


def orig_forward(layer, x):
    f = LIN[0]
    if isinstance(x, tuple):
        return f(x[0], layer.weight, layer.weight_scale_inv_swizzled, input_scale=x[1], backend="cutlass"), None
    return f(x, layer.weight, layer.weight_scale_inv_swizzled, backend="cutlass"), None


def quantized(x):
    from sglang.srt.layers.quantization.fp8_utils import flashinfer_mxfp8_quantize
    from sglang.srt.layers.quantization.mxfp8_input import Mxfp8SwizzledInput
    q, s = flashinfer_mxfp8_quantize(x, is_sf_swizzled_layout=True, alignment=32)
    return Mxfp8SwizzledInput(q, s)


def main():
    LIN.append(setup())
    layer = Layer(5)
    assert [n for n, _ in rs._scale_params(layer)] == ["weight_scale_inv_swizzled"]
    g = torch.Generator(device=DEV).manual_seed(1)
    x_real = torch.randn(6, K, device=DEV, generator=g).to(torch.bfloat16)
    cands = [rs.decide(layer, orig_forward, r, W, x_real) for r in range(W)]
    for r, c in enumerate(cands):
        assert c["window"] is not None and c["slice"] is not None, (r, c)
        pw = c["window"][0]
        assert pw.weight.shape == (512, K) and pw.weight.untyped_storage().data_ptr() == \
            layer.weight.untyped_storage().data_ptr(), r
    offs = cands[0]["window"][3]
    ranges, width = rs._ranges(N, W)
    print(f"  decide: every rank's window and exact slice are bit-exact on the boot inputs; ranges {ranges}, "
          f"windows {[rs.window(N, a, width) for a, _ in ranges]}, offsets {offs}", flush=True)
    ok = True
    for m in (1, 2, 5, 6, 7, 12, 16, 30, 36, 48, 72, 96):
        x = (torch.randn(m, K, device=DEV, generator=g) * 0.5).to(torch.bfloat16)
        for kind, xin in (("bf16", x), ("mxfp8", quantized(x))):
            full = orig_forward(layer, xin)[0]
            res = {}
            for mode in ("window", "slice"):
                parts, pads = [], []
                for r in range(W):
                    proxy, rg, wd, of = cands[r][mode]
                    local = orig_forward(proxy, xin)[0]
                    pads.append(local.shape[1] < wd)
                    parts.append(rs.padded(local, wd))
                    assert of == (offs if mode == "window" else [0] * W)
                out = rs.assemble(torch.cat(parts, dim=1), rg, wd, of)
                res[mode] = (bool(torch.equal(out, full)), pads)
            ok &= res["window"][0] and res["slice"][0] and not any(res["window"][1]) \
                and res["slice"][1] == [False, False, True, True]
            if m in (1, 6, 96) or not (res["window"][0] and res["slice"][0]):
                print(f"  M={m:2d} {kind:5s}: assembled == stock: window {res['window'][0]} (pads "
                      f"{res['window'][1]}), exact slices {res['slice'][0]} (pads {res['slice'][1]})", flush=True)
    assert ok
    # the gathered width per rank is the same in both modes; only ranks 2/3 lose their pad kernels
    t = {}
    x = (torch.randn(6, K, device=DEV, generator=g) * 0.5).to(torch.bfloat16)
    for mode in ("window", "slice"):
        proxy, _, wd, _ = cands[3][mode]
        for _ in range(20):
            rs.padded(orig_forward(proxy, x)[0], wd)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(2000):
            rs.padded(orig_forward(proxy, x)[0], wd)
        torch.cuda.synchronize()
        t[mode] = (time.perf_counter() - t0) / 2000 * 1e6
    print(f"  rank 3, M=6, GEMM + pad per call (eager, host-bound upper bound): window {t['window']:.1f} us, "
          f"exact slice + pad {t['slice']:.1f} us", flush=True)
    print("test_replicated_split_gpu: ok", flush=True)


if __name__ == "__main__":
    main()
