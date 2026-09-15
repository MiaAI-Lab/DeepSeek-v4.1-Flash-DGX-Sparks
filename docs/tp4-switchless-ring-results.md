# TP4 switchless-ring serving numbers (4× DGX Spark)

Independent reproduction of the shipped TP4 profile on a **switchless
4-node ring** (no RoCE switch). The runtime was already serving
`deepseek-v4.1-flash` at the advertised envelope:

- context `1,048,576`
- `MAX_RUNNING_REQUESTS=8` (CUDA-graph decode batch 1–8)
- pinned KV `MAX_TOTAL_TOKENS=8,000,000`
- `MEM_FRACTION_STATIC=0.80`, chunked prefill `1024`
- DSpark `k=5`, MXFP4 experts + MXFP8 dense, Engram packed on local NVMe
- `NCCL_SWITCHLESS_RING_ONLY=1` (sparkring overlay, not `LD_LIBRARY_PATH`)

Raw JSON: [`results/tp4-switchless-ring-20260915.json`](results/tp4-switchless-ring-20260915.json).
Harnesses: `benchmarks/openai_serving.py` (public `/v1/chat/completions`) and
`benchmarks/decode_window.py` (idle C1–C8, character-scaled estimate of tokens 129–641).

This change also includes the opt-in launcher integration used by the live fleet:
patched-NCCL overlay without duplicate runtimes, ring-only env on every rank,
per-node preflight checks, GB10 UMA-safe serial weight loading, local-weight
`NFS_SHARE=0` operation/status, and a physical cabling/IP guide. The launcher work was developed independently; the NCCL transport patch
itself remains credited to FujitsuPolycom/sparkring.

## Hardware / software

| | |
|---|---|
| Nodes | 4× NVIDIA GB10 (DGX Spark), aarch64 |
| Kernel / driver | `6.17.0-1031-nvidia` / `580.173.02` on all four |
| Fabric | switchless ring, 200 GbE CX7, MTU 9000 |
| Image | `dsv41-4x-spark:local` (`lmsysorg/sglang:dev-dsv41`) |
| Measuring checkout | `68b34cf` (TP4 1M-needle README) |
| Weights | local copies, `NFS_SHARE=0` |

## Correctness (OpenAI path)

| Check | Result |
|---|---|
| `GET /v1/models` | `deepseek-v4.1-flash`, `max_model_len=1048576` |
| `19+23`, thinking off | `42`, `finish_reason=stop` |
| thinking on | `OK99`, `reasoning_tokens=17` |
| tool call | `get_weather({"city":"Vigo"})` |
| 8 concurrent streams | 8/8 HTTP 200, queue 0 |

## Decode-window sweep (idle cluster)

Greedy code generation, streaming, thinking off. Window **129–641** is
**estimated** by scaling cumulative output characters to final completion-token
usage; SSE chunks are not individual tokens. Historical TTFT values below were
also character-scaled estimates. The corrected harness measures TTFT at the first
nonempty content event. Wall-clock aggregate uses final token usage and elapsed
time. Server idle (`running=0 waiting=0`) before each wave.

| C | Aggregate tok/s | Median estimated window tok/s / stream | Estimated TTFT | Success |
|---|---:|---:|---:|---|
| 1 | 74.03 | **67.41** | 0.535 s | 1/1 |
| 2 | 111.48 | 54.98 | 0.738 s | 2/2 |
| 4 | 160.05 | 40.84 | 0.978 s | 4/4 |
| 8 | **224.84** | 29.91 | 1.181 s | 8/8 |

An earlier short-prompt report recorded about C1 ~68 / C8 ~201. Prompt,
completion length and measurement method differ; these runs do not establish
a topology advantage or isolate a DSpark acceptance-rate effect.

## Saturation (8 long streams)

8 concurrent HTML/code generations, 6000 completion tokens each, thinking off:

- 48,000 completion tokens in 222.413 s → **215.81 tok/s** wall-clock aggregate
- SGLang decode-batch peak **258.63 tok/s**
- `#running-req: 8`, `#queue-req: 0`
- ~27 tok/s per stream

## Long context (needle)

| Prompt tokens | C | Correct | Wall |
|---|---:|---:|---:|
| ~26k | 1 | 1/1 | 12.8 s |
| ~26k | 4 | 4/4 | 36.8 s |
| ~107k | 1 | 1/1 | 414.7 s |

A full 1M needle was not re-run in this pass (the shipped TP4 profile already
reports it passed). `CONTEXT_LENGTH` was not changed.

## Caveats

- A short-prose `max_tokens=256` matrix taken **while other requests held
  ~557k KV** is not a speed number (C1 looked like 16 tok/s). Discard it.
- Character-scaled decode-window tok/s is approximate and differs from end-to-end tok/s. Short completions are dominated by
  TTFT/prefill.
- Live checkout was `68b34cf` + DSpark **k=5**. `main` after PR #18 defaults
  DSpark **k=3** and adds loop-abort / `DSV41_MAX_NEW_TOKENS`. These benches do
  not claim k=3 performance.
- No A/B against a switched 4-node fabric.

## Reproduce

```bash
# Idle cluster required; set API_KEY if the endpoint requires authentication.
# These commands use the current serving configuration (default k=3), whereas
# the recorded measurements used k=5.
BASE_URL=http://127.0.0.1:8888 python3 benchmarks/decode_window.py
BASE_URL=http://127.0.0.1:8888 python3 benchmarks/openai_serving.py --skip-needles
```

Measured on Carlos's 4× DGX Spark ring.
Twitter: [@mankalan_122](https://x.com/mankalan_122)
