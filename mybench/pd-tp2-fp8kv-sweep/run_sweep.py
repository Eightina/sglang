#!/usr/bin/env python3
"""Run PD tp2 sweep benchmark suite with FP8 KV cache."""

import subprocess
import re
import csv
from datetime import datetime
import time

# Experiment matrix
input_lens = [128, 256, 512, 1024, 2048, 4096]
output_lens = [32, 64, 128, 256, 512]
num_prompts = 100
max_concurrency = 32
request_rate = "inf"

# Fixed params
backend = "sglang"
host = "127.0.0.1"
port = 8000
model = "/root/.cache/modelscope/hub/models/qwen/qwen3___5-27b-fp8"
dataset = "random-ids"
output_dir = "./mybench/pd-tp2-fp8kv-sweep"
csv_file = f"{output_dir}/results.csv"

def parse_bench_output(output_text):
    """Parse bench_serving console output into a dict."""
    metrics = {}

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

def run_experiment(input_len, output_len, exp_num, total_exps):
    """Run a single experiment and return parsed metrics."""
    tag = f"pdtp2_fp8kv_i{input_len}_o{output_len}"
    output_file = f"{output_dir}/{tag}.jsonl"

    cmd = [
        "python3", "-m", "sglang.bench_serving",
        "--backend", backend,
        "--host", host,
        "--port", str(port),
        "--model", model,
        "--dataset-name", dataset,
        "--random-input-len", str(input_len),
        "--random-output-len", str(output_len),
        "--num-prompts", str(num_prompts),
        "--request-rate", request_rate,
        "--max-concurrency", str(max_concurrency),
        "--output-file", output_file,
    ]

    print(f"\n[{exp_num}/{total_exps}] Running: input={input_len}, output={output_len}")
    print(f"Command: {' '.join(cmd)}")

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"ERROR: Command failed with return code {result.returncode}")
        print(f"STDERR: {result.stderr}")
        return None

    metrics = parse_bench_output(result.stdout)

    # Print key metrics
    print(f"✓ Completed: duration={metrics.get('duration_s', 'N/A')}s, "
          f"output_throughput={metrics.get('output_tok_throughput', 'N/A')} tok/s, "
          f"mean_ttft={metrics.get('mean_ttft_ms', 'N/A')}ms, "
          f"mean_itl={metrics.get('mean_itl_ms', 'N/A')}ms")

    return metrics

def append_to_csv(metrics, input_len, output_len):
    """Append a row to the CSV file."""
    timestamp = datetime.now().isoformat()
    tag = f"pdtp2_fp8kv_i{input_len}_o{output_len}"

    row = [
        timestamp, tag, input_len, output_len, num_prompts, max_concurrency, request_rate,
        metrics.get("duration_s", ""),
        metrics.get("successful_requests", ""),
        metrics.get("req_throughput", ""),
        metrics.get("input_tok_throughput", ""),
        metrics.get("output_tok_throughput", ""),
        metrics.get("total_tok_throughput", ""),
        metrics.get("concurrency", ""),
        metrics.get("mean_e2e_ms", ""),
        metrics.get("median_e2e_ms", ""),
        metrics.get("p90_e2e_ms", ""),
        metrics.get("p99_e2e_ms", ""),
        metrics.get("mean_ttft_ms", ""),
        metrics.get("median_ttft_ms", ""),
        metrics.get("p99_ttft_ms", ""),
        metrics.get("mean_tpot_ms", ""),
        metrics.get("median_tpot_ms", ""),
        metrics.get("p99_tpot_ms", ""),
        metrics.get("mean_itl_ms", ""),
        metrics.get("median_itl_ms", ""),
        metrics.get("p95_itl_ms", ""),
        metrics.get("p99_itl_ms", ""),
        metrics.get("max_itl_ms", ""),
    ]

    with open(csv_file, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(row)

def main():
    """Run all experiments."""
    total_exps = len(input_lens) * len(output_lens)
    exp_num = 0

    # Write CSV header
    with open(csv_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "timestamp", "experiment_tag", "input_len", "output_len", "num_prompts",
            "max_concurrency", "request_rate", "duration_s", "successful_requests",
            "req_throughput", "input_tok_throughput", "output_tok_throughput",
            "total_tok_throughput", "concurrency", "mean_e2e_ms", "median_e2e_ms",
            "p90_e2e_ms", "p99_e2e_ms", "mean_ttft_ms", "median_ttft_ms", "p99_ttft_ms",
            "mean_tpot_ms", "median_tpot_ms", "p99_tpot_ms", "mean_itl_ms", "median_itl_ms",
            "p95_itl_ms", "p99_itl_ms", "max_itl_ms"
        ])

    print(f"Starting PD tp2 FP8 KV sweep: {len(input_lens)} input lengths × {len(output_lens)} output lengths = {total_exps} experiments")
    print(f"Results will be saved to: {csv_file}")

    for input_len in input_lens:
        for output_len in output_lens:
            exp_num += 1
            metrics = run_experiment(input_len, output_len, exp_num, total_exps)

            if metrics:
                append_to_csv(metrics, input_len, output_len)
                print(f"  → Results appended to {csv_file}")
            else:
                print(f"  → SKIPPED due to error")

    print(f"\n{'='*60}")
    print(f"All {total_exps} experiments completed!")
    print(f"Results saved to: {csv_file}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
