---
name: run-benchmark-suite
description: >
  Run a series of benchmark tests on a specific model/deployment and
  record results to CSV. Use this skill whenever the user wants to
  benchmark a model across multiple scenarios (long prefill, high
  concurrency, etc.), compare different deployment configurations
  (tp4 vs pd-disaggregation), or systematically explore performance
  characteristics. Also use it when the user says "run benchmarks",
  "test different scenarios", "compare tp4 vs pd", "sweep input/output
  lengths", or wants to generate performance reports.
---

# Run Benchmark Suite

This skill guides you through designing, executing, and recording a
series of `bench_serving` tests for a specific model and deployment
configuration. The goal is to systematically explore performance across
different workload characteristics and produce a structured CSV report.

## Workflow Overview

1. **Clarify test scenarios** with the user (long prefill? high concurrency?)
2. **Design experiment matrix** (input-len × output-len × concurrency × ...)
3. **Start the server** using the `start-server` skill
4. **Run each test** with `bench_serving`, parse output, append to CSV
5. **Summarize results** and offer analysis

---

## Step 1: Clarify Test Scenarios

Ask the user what they want to test. Common scenarios:

| Scenario | Typical Parameters | Goal |
|---|---|---|
| **Long prefill** | input-len: 2048-8192, output-len: 64-256 | Measure TTFT, prefill throughput |
| **Long decode** | input-len: 256-1024, output-len: 512-2048 | Measure TPOT/ITL, decode throughput |
| **High concurrency** | max-concurrency: 64-256, moderate input/output | Measure peak throughput, queueing behavior |
| **Short interactive** | input-len: 128-512, output-len: 32-128 | Measure latency for chat-like workloads |
| **Subagent** | input-len: 1024-4096, output-len: 32-64 | Measure PD disaggregation overhead |
| **Balanced** | input-len: 512-1024, output-len: 256-512 | General-purpose serving performance |

### Example dialogue

**User**: "我想测试长prefill场景下的性能"

**You**: "好的，长prefill场景通常关注TTFT和prefill吞吐。我建议测试以下参数范围：
- `input-len`: 2048, 4096, 8192 (覆盖中等到超长context)
- `output-len`: 64, 128, 256 (短output，因为重点是prefill)
- `num-prompts`: 100 (每个配置跑100个请求)
- `max-concurrency`: 32 (中等并发，避免queueing干扰)

这样会有 3 × 3 = 9 个实验点。你觉得这个范围合适吗？需要调整或增加其他场景吗？"

---

## Step 2: Design Experiment Matrix

Based on the user's goals, create a list of test configurations. Each
configuration is a combination of:

| Parameter | Description | Typical Values |
|---|---|---|
| `input-len` | Input token length | 128, 256, 512, 1024, 2048, 4096, 8192 |
| `output-len` | Output token length | 32, 64, 128, 256, 512, 1024, 2048 |
| `num-prompts` | Number of requests per test | 50, 100, 200 |
| `max-concurrency` | Max concurrent in-flight requests | 16, 32, 64, 128, 256 |
| `request-rate` | Requests per second | `inf` (burst), or a finite rate like `10`, `50` |

### Example experiment list

For a "long prefill + high concurrency" scenario:

```python
experiments = [
    # (input_len, output_len, num_prompts, max_concurrency, request_rate)
    (2048, 128, 100, 32, "inf"),
    (2048, 128, 100, 64, "inf"),
    (4096, 128, 100, 32, "inf"),
    (4096, 128, 100, 64, "inf"),
    (8192, 128, 100, 32, "inf"),
    (8192, 128, 100, 64, "inf"),
]
```

Present this list to the user and confirm before proceeding.

---

## Step 3: Start the Server

Use the `start-server` skill to launch the model. The user should specify:

- **Model path** (e.g., `qwen/qwen3.5-27b-fp8`)
- **Deployment mode** (single-server tp4, or PD disaggregation)
- **Config file** (e.g., `pyscripts/q-tp4.yaml`)

After the server is up and verified with `req.py`, proceed to Step 4.

---

## Step 4: Run Each Test and Record to CSV

### 4a. Create the experiment directory and CSV

```bash
# Create a directory for this benchmark suite
mkdir -p ./mybench/<experiment-name>

# Create the CSV file with headers
cat > ./mybench/<experiment-name>/results.csv << 'EOF'
timestamp,experiment_tag,input_len,output_len,num_prompts,max_concurrency,request_rate,duration_s,successful_requests,req_throughput,input_tok_throughput,output_tok_throughput,total_tok_throughput,concurrency,mean_e2e_ms,median_e2e_ms,p90_e2e_ms,p99_e2e_ms,mean_ttft_ms,median_ttft_ms,p99_ttft_ms,mean_tpot_ms,median_tpot_ms,p99_tpot_ms,mean_itl_ms,median_itl_ms,p95_itl_ms,p99_itl_ms,max_itl_ms
EOF
```

### 4b. CSV column definitions

| Column | Source in bench_serving output | Unit |
|---|---|---|
| `timestamp` | Current time (ISO 8601) | — |
| `experiment_tag` | User-defined label (e.g., "long_prefill_high_conc") | — |
| `input_len` | `--random-input-len` | tokens |
| `output_len` | `--random-output-len` | tokens |
| `num_prompts` | `--num-prompts` | requests |
| `max_concurrency` | `--max-concurrency` | requests |
| `request_rate` | `--request-rate` | req/s or "inf" |
| `duration_s` | `Benchmark duration (s)` | seconds |
| `successful_requests` | `Successful requests` | count |
| `req_throughput` | `Request throughput (req/s)` | req/s |
| `input_tok_throughput` | `Input token throughput (tok/s)` | tok/s |
| `output_tok_throughput` | `Output token throughput (tok/s)` | tok/s |
| `total_tok_throughput` | `Total token throughput (tok/s)` | tok/s |
| `concurrency` | `Concurrency` | aggregate time / wall time |
| `mean_e2e_ms` | `Mean E2E Latency (ms)` | ms |
| `median_e2e_ms` | `Median E2E Latency (ms)` | ms |
| `p90_e2e_ms` | `P90 E2E Latency (ms)` | ms |
| `p99_e2e_ms` | `P99 E2E Latency (ms)` | ms |
| `mean_ttft_ms` | `Mean TTFT (ms)` | ms |
| `median_ttft_ms` | `Median TTFT (ms)` | ms |
| `p99_ttft_ms` | `P99 TTFT (ms)` | ms |
| `mean_tpot_ms` | `Mean TPOT (ms)` | ms |
| `median_tpot_ms` | `Median TPOT (ms)` | ms |
| `p99_tpot_ms` | `P99 TPOT (ms)` | ms |
| `mean_itl_ms` | `Mean ITL (ms)` | ms |
| `median_itl_ms` | `Median ITL (ms)` | ms |
| `p95_itl_ms` | `P95 ITL (ms)` | ms |
| `p99_itl_ms` | `P99 ITL (ms)` | ms |
| `max_itl_ms` | `Max ITL (ms)` | ms |

### 4c. Run a single test

```bash
python3 -m sglang.bench_serving \
  --backend sglang \
  --host 127.0.0.1 --port 8000 \
  --model /root/.cache/modelscope/hub/models/qwen/qwen3___5-27b-fp8 \
  --dataset-name random-ids \
  --random-input-len <input_len> \
  --random-output-len <output_len> \
  --num-prompts <num_prompts> \
  --request-rate <request_rate> \
  --max-concurrency <max_concurrency> \
  --output-file ./mybench/<experiment-name>/<tag>_i<input>_o<output>_c<conc>.jsonl
```

### 4d. Parse output and append to CSV

After the test completes, `bench_serving` prints a summary block like:

```
============ Serving Benchmark Result ============
Backend:                                 sglang
Traffic request rate:                    inf
Max request concurrency:                 64
Successful requests:                     200
Benchmark duration (s):                  38.17
...
```

**Extract each metric and append a row to the CSV.** Example Python snippet:

```python
import re
from datetime import datetime

def parse_bench_output(output_text):
    """Parse bench_serving console output into a dict."""
    metrics = {}
    
    # Pattern: "Metric Name:    value"
    patterns = {
        "successful_requests": r"Successful requests:\s+(\d+)",
        "duration_s": r"Benchmark duration \(s\):\s+([\d.]+)",
        "req_throughput": r"Request throughput \(req/s\):\s+([\d.]+)",
        "input_tok_throughput": r"Input token throughput \(tok/s\):\s+([\d.]+)",
        "output_tok_throughput": r"Output token throughput \(tok/s\):\s+([\d.]+)",
        "total_tok_throughput": r"Total token throughput \(tok/s\):\s+([\d.]+)",
        "concurrency": r"Concurrency:\s+([\d.]+)",
        "mean_e2e_ms": r"Mean E2E Latency \(ms\):\s+([\d.]+)",
        "median_e2e_ms": r"Median E2E Latency \(ms\):\s+([\d.]+)",
        "p90_e2e_ms": r"P90 E2E Latency \(ms\):\s+([\d.]+)",
        "p99_e2e_ms": r"P99 E2E Latency \(ms\):\s+([\d.]+)",
        "mean_ttft_ms": r"Mean TTFT \(ms\):\s+([\d.]+)",
        "median_ttft_ms": r"Median TTFT \(ms\):\s+([\d.]+)",
        "p99_ttft_ms": r"P99 TTFT \(ms\):\s+([\d.]+)",
        "mean_tpot_ms": r"Mean TPOT \(ms\):\s+([\d.]+)",
        "median_tpot_ms": r"Median TPOT \(ms\):\s+([\d.]+)",
        "p99_tpot_ms": r"P99 TPOT \(ms\):\s+([\d.]+)",
        "mean_itl_ms": r"Mean ITL \(ms\):\s+([\d.]+)",
        "median_itl_ms": r"Median ITL \(ms\):\s+([\d.]+)",
        "p95_itl_ms": r"P95 ITL \(ms\):\s+([\d.]+)",
        "p99_itl_ms": r"P99 ITL \(ms\):\s+([\d.]+)",
        "max_itl_ms": r"Max ITL \(ms\):\s+([\d.]+)",
    }
    
    for key, pattern in patterns.items():
        match = re.search(pattern, output_text)
        metrics[key] = match.group(1) if match else ""
    
    return metrics

# Example usage:
# output = subprocess.run([...], capture_output=True, text=True)
# metrics = parse_bench_output(output.stdout)
# 
# row = f"{datetime.now().isoformat()},{tag},{input_len},{output_len},{num_prompts},{max_concurrency},{request_rate},"
# row += f"{metrics['duration_s']},{metrics['successful_requests']},{metrics['req_throughput']},"
# row += f"{metrics['input_tok_throughput']},{metrics['output_tok_throughput']},{metrics['total_tok_throughput']},"
# row += f"{metrics['concurrency']},{metrics['mean_e2e_ms']},{metrics['median_e2e_ms']},"
# row += f"{metrics['p90_e2e_ms']},{metrics['p99_e2e_ms']},{metrics['mean_ttft_ms']},{metrics['median_ttft_ms']},"
# row += f"{metrics['p99_ttft_ms']},{metrics['mean_tpot_ms']},{metrics['median_tpot_ms']},{metrics['p99_tpot_ms']},"
# row += f"{metrics['mean_itl_ms']},{metrics['median_itl_ms']},{metrics['p95_itl_ms']},{metrics['p99_itl_ms']},{metrics['max_itl_ms']}"
# 
# with open(csv_path, "a") as f:
#     f.write(row + "\n")
```

### 4e. Print progress after each test

After each test, print a summary to the user:

```
✓ Test 3/9 completed: input=4096, output=128, concurrency=32
  Output throughput: 496.25 tok/s
  Mean TTFT: 4921 ms
  Mean ITL: 33.60 ms
  Results appended to ./mybench/long_prefill/results.csv
```

---

## Step 5: Summarize and Analyze

After all tests complete:

1. **Print the full CSV** to the user (or offer to open it in a spreadsheet).
2. **Highlight key findings**:
   - Which configuration has the highest output throughput?
   - Which has the lowest TTFT?
   - How does throughput scale with concurrency?
   - How does ITL change with input length?
3. **Offer visualization** (if the user wants):
   - Line chart: throughput vs. concurrency
   - Heatmap: TTFT vs. (input_len, output_len)
   - Bar chart: compare different deployment modes (tp4 vs pd)

---

## Example: Full Workflow

**User**: "我想对比 tp4 和 pd 分离在长prefill场景下的性能"

**You**:
1. Confirm test scenarios: "长prefill场景，input-len: 2048-8192, output-len: 128, concurrency: 32"
2. Design experiment matrix: 6 configurations (3 input lengths × 2 deployment modes)
3. Start tp4 server using `start-server` skill
4. Run 3 tests (one per input length), append to `./mybench/tp4_long_prefill/results.csv`
5. Stop tp4 server, start PD server
6. Run 3 tests, append to `./mybench/pd_long_prefill/results.csv`
7. Merge both CSVs, print comparison table, highlight which mode wins in each config

---

## Troubleshooting

### bench_serving fails with "Connection refused"

- Server not running or wrong port. Verify with `req.py`.
- In PD mode, make sure the **router** is on port 8000.

### Output throughput is suspiciously low

- Check if `successful_requests` matches `num_prompts`. If not, some requests failed.
- Look at `max_itl_ms` — if it's very high (e.g., >5s), there may be queueing or scheduling issues.
- For PD mode, check TTFT — if it's >>1s, KV transfer overhead may dominate.

### CSV is hard to read

- Offer to convert to markdown table or Excel.
- Suggest pivot tables: group by `input_len`, sort by `output_tok_throughput`.

### Want to resume a partially completed suite

- The CSV file persists between runs. Just continue from where you left off.
- Check the CSV to see which configurations are missing, then run only those.

---

## CSV Template Reference

Here's a minimal CSV template for copy-paste:

```csv
timestamp,experiment_tag,input_len,output_len,num_prompts,max_concurrency,request_rate,duration_s,successful_requests,req_throughput,input_tok_throughput,output_tok_throughput,total_tok_throughput,concurrency,mean_e2e_ms,median_e2e_ms,p90_e2e_ms,p99_e2e_ms,mean_ttft_ms,median_ttft_ms,p99_ttft_ms,mean_tpot_ms,median_tpot_ms,p99_tpot_ms,mean_itl_ms,median_itl_ms,p95_itl_ms,p99_itl_ms,max_itl_ms
```

And a sample row (from the PD tp2 short workload):

```csv
2026-06-09T10:30:00,og_short_pdtp2,127,32,200,64,inf,38.17,200,5.24,666.40,169.06,835.45,53.62,10233.79,11055.53,12625.49,14904.87,9300.38,10147.72,13928.62,29.47,30.37,31.95,29.88,30.46,32.24,33.17,35.05
```
