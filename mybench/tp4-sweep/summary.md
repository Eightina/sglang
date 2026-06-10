# TP4 Sweep Benchmark Report

- **Model**: `qwen/qwen3.5-27b-fp8`
- **Deployment**: single-server, TP=4 (GPU 0,1,2,3)
- **Config**: `pyscripts/q-tp4.yaml`
- **Dataset**: `random-ids`
- **Num prompts**: 100 per config
- **Max concurrency**: 32
- **Request rate**: inf (burst)
- **Date**: 2026-06-10

## Summary Table: Output Token Throughput (tok/s)

Higher is better. This is the primary metric for serving capacity.

| input_len \ output_len | 32 | 64 | 128 | 256 | 512 |
|---|---:|---:|---:|---:|---:|
| **128** | 189.08 | 315.23 | 481.94 | 637.62 | 794.69 |
| **256** | 206.53 | 309.20 | 480.88 | 631.48 | 768.63 |
| **512** | 164.72 | 261.08 | 424.42 | 594.72 | 754.39 |
| **1024** | 114.76 | 189.78 | 353.30 | 518.56 | 688.27 |
| **2048** | 64.63 | 110.13 | 218.63 | 366.03 | 537.65 |
| **4096** | 36.96 | 64.71 | 132.30 | 233.27 | 380.29 |

## Summary Table: Mean TTFT (ms)

Lower is better. TTFT = time to first token (dominated by prefill).

| input_len \ output_len | 32 | 64 | 128 | 256 | 512 |
|---|---:|---:|---:|---:|---:|
| **128** | 481.47 | 285.81 | 269.59 | 262.08 | 237.83 |
| **256** | 424.00 | 372.20 | 340.04 | 328.90 | 416.28 |
| **512** | 571.46 | 508.68 | 472.90 | 452.46 | 427.90 |
| **1024** | 937.25 | 829.14 | 750.20 | 720.31 | 703.03 |
| **2048** | 1709.54 | 1527.26 | 1410.33 | 1350.50 | 1298.35 |
| **4096** | 3420.26 | 2957.65 | 2789.71 | 2696.46 | 2587.12 |

## Summary Table: Mean ITL (ms)

Lower is better. ITL = inter-token latency (dominated by decode).

| input_len \ output_len | 32 | 64 | 128 | 256 | 512 |
|---|---:|---:|---:|---:|---:|
| **128** | 141.08 | 86.25 | 58.52 | 41.81 | 32.93 |
| **256** | 129.74 | 85.24 | 57.51 | 41.87 | 33.49 |
| **512** | 161.33 | 99.88 | 64.28 | 44.02 | 34.29 |
| **1024** | 226.44 | 135.70 | 75.06 | 49.80 | 37.29 |
| **2048** | 402.93 | 236.49 | 120.88 | 70.67 | 47.95 |
| **4096** | 677.82 | 394.71 | 194.92 | 109.76 | 67.54 |

## Summary Table: Mean TPOT (ms)

Lower is better. TPOT = time per output token excluding first token.

| input_len \ output_len | 32 | 64 | 128 | 256 | 512 |
|---|---:|---:|---:|---:|---:|
| **128** | 142.71 | 87.98 | 58.99 | 41.97 | 33.70 |
| **256** | 140.37 | 88.73 | 60.03 | 43.01 | 34.29 |
| **512** | 172.84 | 106.00 | 68.37 | 46.41 | 35.28 |
| **1024** | 246.66 | 150.45 | 83.71 | 55.10 | 38.69 |
| **2048** | 442.89 | 262.67 | 136.98 | 81.13 | 50.03 |
| **4096** | 781.03 | 463.35 | 236.52 | 134.84 | 71.03 |

## Summary Table: Request Throughput (req/s)

| input_len \ output_len | 32 | 64 | 128 | 256 | 512 |
|---|---:|---:|---:|---:|---:|
| **128** | 12.61 | 11.34 | 7.89 | 5.10 | 3.08 |
| **256** | 13.77 | 11.12 | 7.87 | 5.05 | 2.98 |
| **512** | 10.98 | 9.39 | 6.95 | 4.75 | 2.92 |
| **1024** | 7.65 | 6.83 | 5.78 | 4.15 | 2.67 |
| **2048** | 4.31 | 3.96 | 3.58 | 2.93 | 2.08 |
| **4096** | 2.46 | 2.33 | 2.17 | 1.86 | 1.47 |

## Summary Table: P99 ITL (ms)

Lower is better. Shows tail latency.

| input_len \ output_len | 32 | 64 | 128 | 256 | 512 |
|---|---:|---:|---:|---:|---:|
| **128** | 525.58 | 457.61 | 374.15 | 213.37 | 119.39 |
| **256** | 685.38 | 499.07 | 297.41 | 207.81 | 120.72 |
| **512** | 994.03 | 498.55 | 361.64 | 217.72 | 147.53 |
| **1024** | 1833.15 | 728.67 | 440.18 | 296.82 | 220.76 |
| **2048** | 1593.20 | 1635.77 | 940.31 | 631.58 | 460.29 |
| **4096** | 2549.81 | 2581.03 | 1608.26 | 1105.21 | 808.67 |

## Key Findings

### 1. Output throughput scales with output length
For a fixed input length, output tok/s roughly doubles when output length doubles:
- input=128: 189 → 315 → 482 → 638 → 795 (as output goes 32 → 64 → 128 → 256 → 512)

This is the classic "decode-amortization" pattern: the prefill cost is fixed per request, so longer outputs spread it over more tokens.

### 2. Peak throughput: ~795 tok/s at (input=128, output=512)
Best config for raw output token throughput.

### 3. TTFT scales linearly with input length
- input=128: ~250ms
- input=256: ~370ms
- input=512: ~480ms
- input=1024: ~780ms
- input=2048: ~1460ms
- input=4096: ~2890ms

Roughly: **TTFT (ms) ≈ 0.7 × input_len** at concurrency 32.

### 4. ITL/TPOT also scales with input length
Even though decode is per-token, longer input means larger KV cache, which slows down each decode step (attention is O(n) over KV length):
- input=128, output=512: ITL=33ms
- input=4096, output=512: ITL=68ms (2× slower)

### 5. Short-output, long-input is the worst case
- input=4096, output=32: only **36.96 tok/s** output throughput, TTFT=3.4s
- That's **21× worse** than the peak config (795 tok/s)

### 6. P99 ITL shows high variance for short-output workloads
- input=4096, output=32: P99 ITL = 2549ms (vs mean 677ms) — 3.7× spike
- input=2048, output=32: P99 ITL = 1593ms (vs mean 403ms) — 4× spike

Long-output workloads have much tighter P99/mean ratios (typically 2-4×).

## Deployment Config

```yaml
model-path: qwen/qwen3.5-27b-fp8
host: 0.0.0.0
port: 8000
tensor-parallel-size: 4
enable-metrics: true
log-requests: true
```

## Raw Data

See `results.csv` in this directory for the full 29-column dataset.
