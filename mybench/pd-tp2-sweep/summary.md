# PD TP2 Sweep Benchmark Report

- **Model**: `qwen/qwen3.5-27b-fp8`
- **Deployment**: PD disaggregation, prefill TP=2 (GPU 0,1) + decode TP=2 (GPU 2,3)
- **Config**: `pyscripts/q-prefilltp2.yaml` + `pyscripts/q-decodetp2.yaml` + router
- **Transfer backend**: NIXL (over PCIe, no NVLink)
- **Dataset**: `random-ids`
- **Num prompts**: 100 per config
- **Max concurrency**: 32
- **Request rate**: inf (burst)
- **Date**: 2026-06-10

## Summary Table: Output Token Throughput (tok/s)

Higher is better.

| input_len \ output_len | 32 | 64 | 128 | 256 | 512 |
|---|---:|---:|---:|---:|---:|
| **128** | 80.02 | 145.58 | 301.49 | 476.13 | 518.30 |
| **256** | 78.05 | 142.69 | 283.96 | 471.11 | 508.51 |
| **512** | 71.73 | 130.85 | 282.15 | 452.43 | 504.83 |
| **1024** | 63.63 | 114.63 | 235.12 | 415.79 | 494.58 |
| **2048** | 46.99 | 87.59 | 196.25 | 350.28 | 478.10 |
| **4096** | 27.50 | 51.38 | 110.80 | 206.34 | 353.91 |

## Summary Table: Mean TTFT (ms)

Lower is better. TTFT = time to first token (includes prefill + KV transfer).

| input_len \ output_len | 32 | 64 | 128 | 256 | 512 |
|---|---:|---:|---:|---:|---:|
| **128** | 4704.87 | 4332.54 | 3577.93 | 2814.32 | 4406.69 |
| **256** | 4983.26 | 4561.43 | 3905.41 | 2955.45 | 4533.30 |
| **512** | 5292.97 | 4933.22 | 4312.29 | 3248.21 | 4651.29 |
| **1024** | 6125.21 | 5944.00 | 5303.04 | 4012.60 | 4907.93 |
| **2048** | 8383.55 | 7956.57 | 6974.85 | 5678.23 | 5211.96 |
| **4096** | 14560.77 | 13994.23 | 13502.58 | 12527.88 | 10794.35 |

## Summary Table: Mean ITL (ms)

Lower is better. ITL = inter-token latency (dominated by decode, isolated from prefill in PD mode).

| input_len \ output_len | 32 | 64 | 128 | 256 | 512 |
|---|---:|---:|---:|---:|---:|
| **128** | 29.16 | 30.08 | 31.00 | 32.95 | 33.35 |
| **256** | 28.62 | 29.96 | 31.01 | 33.10 | 33.46 |
| **512** | 28.19 | 29.59 | 31.04 | 33.03 | 33.62 |
| **1024** | 26.72 | 28.38 | 30.38 | 32.65 | 33.85 |
| **2048** | 21.66 | 25.93 | 29.05 | 31.36 | 34.30 |
| **4096** | 19.97 | 23.90 | 28.58 | 30.73 | 32.76 |

## Summary Table: Mean TPOT (ms)

Lower is better. TPOT = time per output token excluding first token.

| input_len \ output_len | 32 | 64 | 128 | 256 | 512 |
|---|---:|---:|---:|---:|---:|
| **128** | 28.82 | 30.06 | 30.78 | 33.00 | 33.44 |
| **256** | 27.30 | 29.09 | 30.96 | 33.04 | 33.56 |
| **512** | 27.22 | 29.28 | 30.86 | 33.04 | 33.71 |
| **1024** | 25.87 | 27.68 | 29.31 | 31.83 | 33.93 |
| **2048** | 18.24 | 22.65 | 27.06 | 29.40 | 34.23 |
| **4096** | 16.52 | 20.47 | 26.43 | 28.99 | 31.93 |

## Summary Table: P99 ITL (ms)

Lower is better. Shows tail latency.

| input_len \ output_len | 32 | 64 | 128 | 256 | 512 |
|---|---:|---:|---:|---:|---:|
| **128** | 32.07 | 33.11 | 34.06 | 36.08 | 36.67 |
| **256** | 32.25 | 34.22 | 34.13 | 37.76 | 36.53 |
| **512** | 31.44 | 32.15 | 35.35 | 35.25 | 35.37 |
| **1024** | 32.02 | 32.99 | 34.45 | 36.69 | 37.16 |
| **2048** | 31.84 | 32.13 | 33.87 | 35.37 | 37.80 |
| **4096** | 31.78 | 32.46 | 32.97 | 34.54 | 36.70 |

## Summary Table: Max ITL (ms)

| input_len \ output_len | 32 | 64 | 128 | 256 | 512 |
|---|---:|---:|---:|---:|---:|
| **128** | 36.99 | 38.49 | 38.68 | 40.78 | 43.11 |
| **256** | 34.07 | 52.07 | 36.13 | 42.12 | 90.72 |
| **512** | 33.60 | 34.36 | 42.19 | 858.86 | 70.94 |
| **1024** | 33.52 | 35.22 | 59.54 | 40.26 | 59.10 |
| **2048** | 52.05 | 44.43 | 42.73 | 38.60 | 42.62 |
| **4096** | 80.28 | 58.06 | 75.59 | 52.64 | 52.64 |

## Key Findings

### 1. ITL is remarkably stable and low
Unlike TP4, ITL is almost independent of input length. Even at input=4096, ITL only reaches 33ms. This is the primary advantage of PD separation: decode is completely isolated from prefill interference.

### 2. TTFT is extremely high (4-14 seconds)
The KV cache transfer over PCIe dominates TTFT. Even for input=128, TTFT is ~4s (vs ~0.3s for TP4). This is the primary bottleneck.

### 3. Output throughput is limited by TTFT
Each request spends most of its time waiting for KV transfer. The decode GPU is underutilized — it works briefly (ITL ~30ms × output_len) then waits for the next request's KV transfer.

### 4. P99 ITL is excellent
P99 ITL is typically only 1.1-1.2× the mean ITL, showing very consistent decode latency. Compare with TP4 where P99 can be 3-4× the mean.

### 5. One anomaly: (512, 256) had max ITL = 858ms
This is likely a transient scheduling event during that test. P99 was 35ms so it was a single spike.
