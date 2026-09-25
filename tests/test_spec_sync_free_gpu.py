"""GPU checks for adapter/spec_sync_free.py's rank-invariant draft noise (one GPU, < 1 GB, in the image):

  docker run --rm --gpus all --network none -v <ds41>:/ds41:ro -e PYTHONPATH=/ds41/adapter \
      --entrypoint python3 <image> /ds41/tests/test_spec_sync_free_gpu.py

1. Exp(1) noise: mean 1, variance 1, P(E > 3) = e^-3, no NaN / negative values;
2. the stream is a pure function of (seed, counter, step): two independent "ranks" with their own
   buffers produce bit-identical noise; another counter or step gives other noise;
3. CUDA graph: a captured draft-sampler-shaped call (5 steps + counter bump) replayed 3 times gives
   the noise of counters 0, 1, 2 exactly as an eager recomputation, on two "ranks" alike;
4. the engine's SampleStepTokens with this noise samples softmax(logits / T) exactly (chi-square),
   greedy rows stay argmax, and two ranks pick identical tokens at the real vocab (129280);
5. the source checks accept the engine's real DsparkDraftSampler.__call__ and
   DsparkVerifyEpilogue._accept in this image.
"""
import math
import os

# TP4-line adapters: armed only under start-tp4.sh, which sets DSV41_LAUNCHER=tp4 in every rank.
os.environ["DSV41_LAUNCHER"] = "tp4"

os.environ.setdefault("DSV41_SPEC_SYNC_FREE", "draft,accept")
import torch  # noqa: E402

import spec_sync_free as ssf  # noqa: E402

torch.cuda.set_per_process_memory_fraction(min(1.0, 1e9 / torch.cuda.get_device_properties(0).total_memory))
DEV = "cuda"
V_REAL = 129280


def state(seed=12345):
    return (torch.tensor([seed], dtype=torch.int64, device=DEV), torch.zeros(1, dtype=torch.int64, device=DEV))


def test_distribution():
    seed, ctr = state()
    out = torch.empty(64, V_REAL, device=DEV)
    ssf.fill_exp_noise(out, seed, ctr, 0)
    x = out.double()
    assert torch.isfinite(x).all() and (x >= 0).all()
    n = x.numel()
    mean, var = x.mean().item(), x.var().item()
    tail = (x > 3).double().mean().item()
    assert abs(mean - 1) < 6 / math.sqrt(n), mean
    assert abs(var - 1) < 12 / math.sqrt(n), var
    assert abs(tail - math.exp(-3)) < 6 * math.sqrt(math.exp(-3) / n), tail
    print(f"  n={n} mean={mean:.5f} var={var:.5f} P(E>3)={tail:.5f} (e^-3={math.exp(-3):.5f})")


def test_pure_function():
    a = torch.empty(8, V_REAL, device=DEV)
    b = torch.empty(8, V_REAL, device=DEV)
    s0, c0 = state()
    s1, c1 = state()
    ssf.fill_exp_noise(a, s0, c0, 3)
    ssf.fill_exp_noise(b, s1, c1, 3)
    assert torch.equal(a, b)
    ssf.fill_exp_noise(b, s1, c1, 4)
    assert (a != b).float().mean() > 0.99
    c1.add_(1)
    ssf.fill_exp_noise(b, s1, c1, 3)
    assert (a != b).float().mean() > 0.99
    # a slice view (the engine passes exp_noise[:bs]) fills only its rows
    big = torch.full((16, V_REAL), -1.0, device=DEV)
    ssf.fill_exp_noise(big[:8], s0, c0, 3)
    assert torch.equal(big[:8], a) and (big[8:] == -1).all()


def fake_call(noise, steps=5):
    """What DsparkDraftSampler.__call__ does with the noise under the adapter's __call__ wrapper."""
    noise.step = 0
    outs = [noise[:4].exponential_().clone() for _ in range(steps)]
    noise.ctr.add_(1)
    return torch.stack(outs)


def test_graph_replay():
    ranks = []
    for _ in range(2):
        seed, ctr = state(777)
        noise = ssf.RankInvariantNoise(torch.empty(8, 1000, device=DEV), seed, ctr)
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            fake_call(noise)                       # warm-up (as the capture runner does), ctr -> 1
        torch.cuda.current_stream().wait_stream(s)
        ctr.zero_()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            out = fake_call(noise)
        ranks.append((g, out, noise))
    for rep in range(3):
        snaps = []
        for g, out, noise in ranks:
            g.replay()
            snaps.append(out.clone())
        torch.cuda.synchronize()
        assert torch.equal(snaps[0], snaps[1]), "ranks diverged"
        seed, ctr = state(777)
        ctr.fill_(rep)
        ref = torch.stack([ssf.fill_exp_noise(torch.empty(4, 1000, device=DEV), seed, ctr, k) for k in range(5)])
        assert torch.equal(snaps[0], ref), f"replay {rep} is not counter {rep}"
        assert ranks[0][2].ctr.item() == rep + 1


def test_sampling_exact():
    from sglang.kernels.ops.speculative.dspark.dspark_draft_model import SampleStepTokens
    torch.manual_seed(0)
    v, rows, draws = 32, 64, 2000
    logits = (torch.randn(v, device=DEV) * 2).expand(rows, v).contiguous()
    temps = torch.full((rows,), 0.7, device=DEV)
    greedy = torch.zeros(rows, dtype=torch.bool, device=DEV)
    greedy[:4] = True
    seed, ctr = state(99)
    noise = torch.empty(rows, v, device=DEV)
    counts = torch.zeros(v, dtype=torch.float64, device=DEV)
    for _ in range(draws):
        ssf.fill_exp_noise(noise, seed, ctr, 0)
        tok = SampleStepTokens.execute(step_logits=logits, temperatures=temps, greedy_mask=greedy,
                                       exp_noise=noise)
        assert (tok[:4] == logits[0].argmax()).all()
        counts += torch.bincount(tok[4:], minlength=v).double()
        ctr.add_(1)
    p = torch.softmax(logits[0].double() / 0.7, -1)
    n = counts.sum()
    exp_ = p * n
    keep = exp_ > 5
    chi2 = (((counts - exp_) ** 2) / exp_)[keep].sum().item()
    dof = int(keep.sum().item()) - 1
    z = (chi2 - dof) / math.sqrt(2 * dof)
    print(f"  chi2={chi2:.1f} dof={dof} z={z:+.2f} over {int(n)} samples")
    assert abs(z) < 4, z
    # two ranks, real vocab, mixed greedy/sampled rows: identical tokens
    lg = torch.randn(16, V_REAL, device=DEV) * 3
    t = torch.full((16,), 1.0, device=DEV)
    gm = torch.arange(16, device=DEV) % 3 == 0
    toks = []
    for _ in range(2):
        s, c = state(4242)
        nz = ssf.fill_exp_noise(torch.empty(16, V_REAL, device=DEV), s, c, 2)
        toks.append(SampleStepTokens.execute(step_logits=lg, temperatures=t, greedy_mask=gm, exp_noise=nz))
    assert torch.equal(toks[0], toks[1])
    assert torch.equal(toks[0][gm], lg[gm].argmax(-1))


def test_engine_sources():
    from sglang.srt.speculative.dspark_components import dspark_draft_sampler, dspark_verify
    assert ssf._call_uses_exp_noise(dspark_draft_sampler.DsparkDraftSampler)
    assert ssf._accept_syncs_back_to_back(dspark_verify.DsparkVerifyEpilogue)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name, flush=True)
    print("ALL OK")
