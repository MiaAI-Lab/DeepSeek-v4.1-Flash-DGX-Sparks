## v41-r1 (2026-09-26T17:22:20Z)



Prompt set `v1` (identical across boots), temperature 0, thinking off. Tokens from the server's usage block; TTFT = first token delta.

### Throughput by concurrency (8 categories; the counting ceiling is excluded)

| C | aggregate tok/s | per-stream tok/s | mean TTFT (s) |
|---|---|---|---|
| C1 | 81.21 | 93.12 | 0.211 |
| C8 | 362.17 | 56.73 | 0.357 |
| C16 | 557.96 | 43.73 | 0.444 |
| C32 | 780.62 | 30.81 | 0.664 |

### Per-stream tok/s by category

| category | C1 | C8 | C16 | C32 |
|---|---|---|---|---|
| coding | 118.88 | 73.65 | 62.46 | 47.39 |
| json | 96.18 | 59.65 | 42.31 | 30.24 |
| narrative | 60.81 | 33.59 | 24.72 | 15.47 |
| prose | 58.45 | 33.97 | 28.63 | 17.48 |
| math | 122.88 | 73.28 | 56.88 | 39.9 |
| reasoning | 92.0 | 54.97 | 39.72 | 28.68 |
| summary | 63.43 | 36.28 | 27.87 | 16.98 |
| format | 132.34 | 88.41 | 67.29 | 50.34 |
| ceiling_count | 138.86 | 103.76 | 82.44 | 60.63 |

### Cold prefill (unique prefix)

| target | prompt tokens | TTFT (s) | prefill tok/s |
|---|---|---|---|
