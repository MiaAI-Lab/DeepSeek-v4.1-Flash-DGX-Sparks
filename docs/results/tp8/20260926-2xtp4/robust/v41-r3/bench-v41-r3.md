## v41-r3 (2026-09-26T17:27:15Z)



Prompt set `v1` (identical across boots), temperature 0, thinking off. Tokens from the server's usage block; TTFT = first token delta.

### Throughput by concurrency (8 categories; the counting ceiling is excluded)

| C | aggregate tok/s | per-stream tok/s | mean TTFT (s) |
|---|---|---|---|
| C1 | 84.65 | 96.86 | 0.197 |
| C8 | 367.67 | 56.45 | 0.323 |
| C16 | 564.29 | 44.47 | 0.419 |
| C32 | 813.31 | 31.71 | 0.599 |

### Per-stream tok/s by category

| category | C1 | C8 | C16 | C32 |
|---|---|---|---|---|
| coding | 125.21 | 84.89 | 67.62 | 48.26 |
| json | 101.06 | 56.41 | 41.49 | 31.16 |
| narrative | 63.8 | 32.01 | 24.07 | 14.9 |
| prose | 61.08 | 36.0 | 25.76 | 17.26 |
| math | 127.54 | 71.57 | 59.5 | 42.71 |
| reasoning | 94.98 | 52.16 | 40.2 | 28.97 |
| summary | 65.49 | 34.39 | 27.7 | 19.18 |
| format | 135.68 | 84.2 | 69.4 | 51.28 |
| ceiling_count | 144.17 | 103.64 | 82.93 | 61.23 |

### Cold prefill (unique prefix)

| target | prompt tokens | TTFT (s) | prefill tok/s |
|---|---|---|---|
