## v41-r2 (2026-09-26T17:24:49Z)



Prompt set `v1` (identical across boots), temperature 0, thinking off. Tokens from the server's usage block; TTFT = first token delta.

### Throughput by concurrency (8 categories; the counting ceiling is excluded)

| C | aggregate tok/s | per-stream tok/s | mean TTFT (s) |
|---|---|---|---|
| C1 | 83.74 | 95.78 | 0.201 |
| C8 | 354.32 | 56.2 | 0.331 |
| C16 | 559.76 | 43.76 | 0.402 |
| C32 | 809.77 | 31.58 | 0.567 |

### Per-stream tok/s by category

| category | C1 | C8 | C16 | C32 |
|---|---|---|---|---|
| coding | 125.68 | 76.57 | 61.72 | 47.35 |
| json | 99.88 | 51.98 | 44.49 | 31.56 |
| narrative | 64.09 | 32.38 | 23.83 | 15.05 |
| prose | 59.3 | 34.34 | 26.02 | 17.36 |
| math | 128.04 | 69.75 | 58.95 | 41.93 |
| reasoning | 92.57 | 55.86 | 39.49 | 28.95 |
| summary | 64.73 | 35.52 | 25.78 | 18.97 |
| format | 131.96 | 93.18 | 69.79 | 51.47 |
| ceiling_count | 143.37 | 100.52 | 82.42 | 62.65 |

### Cold prefill (unique prefix)

| target | prompt tokens | TTFT (s) | prefill tok/s |
|---|---|---|---|
