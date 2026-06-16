#!/usr/bin/env python3
"""Analyze decode scheduler profiling results from 16 vs 32 concurrency."""

import re
import sys
from pathlib import Path
from collections import defaultdict
import statistics

def parse_profile_line(line):
    """Parse a DECODE PROFILE OVERLAP line."""
    pattern = r'\[DECODE PROFILE OVERLAP\] recv: ([\d.]+)ms, process_input: ([\d.]+)ms, process_decode_queue: ([\d.]+)ms, war_barrier: ([\d.]+)ms, get_next_batch: ([\d.]+)ms, run_batch: ([\d.]+)ms, process_batch_result: ([\d.]+)ms, launch_batch_sample: ([\d.]+)ms, total: ([\d.]+)ms'
    match = re.search(pattern, line)
    if match:
        return {
            'recv': float(match.group(1)),
            'process_input': float(match.group(2)),
            'process_decode_queue': float(match.group(3)),
            'war_barrier': float(match.group(4)),
            'get_next_batch': float(match.group(5)),
            'run_batch': float(match.group(6)),
            'process_batch_result': float(match.group(7)),
            'launch_batch_sample': float(match.group(8)),
            'total': float(match.group(9)),
        }
    return None

def analyze_log_file(log_path):
    """Analyze a decode server log file."""
    data = defaultdict(list)

    with open(log_path, 'r') as f:
        for line in f:
            if 'DECODE PROFILE OVERLAP' in line:
                parsed = parse_profile_line(line)
                if parsed:
                    for stage, value in parsed.items():
                        data[stage].append(value)

    return data

def compute_stats(values):
    """Compute statistics for a list of values."""
    if not values:
        return {'min': 0, 'avg': 0, 'p50': 0, 'p99': 0, 'max': 0, 'count': 0}

    sorted_vals = sorted(values)
    n = len(sorted_vals)

    return {
        'min': sorted_vals[0],
        'avg': statistics.mean(sorted_vals),
        'p50': sorted_vals[n // 2],
        'p99': sorted_vals[int(n * 0.99)],
        'max': sorted_vals[-1],
        'count': n,
    }

def main():
    if len(sys.argv) != 3:
        print("Usage: python3 analyze_decode_profiling.py <16_conc_log> <32_conc_log>")
        sys.exit(1)

    log_16 = Path(sys.argv[1])
    log_32 = Path(sys.argv[2])

    print(f"Analyzing {log_16}...")
    data_16 = analyze_log_file(log_16)

    print(f"Analyzing {log_32}...")
    data_32 = analyze_log_file(log_32)

    # Compute statistics
    stats_16 = {stage: compute_stats(values) for stage, values in data_16.items()}
    stats_32 = {stage: compute_stats(values) for stage, values in data_32.items()}

    # Print comparison table
    stages = ['recv', 'process_input', 'process_decode_queue', 'war_barrier',
              'get_next_batch', 'run_batch', 'process_batch_result',
              'launch_batch_sample', 'total']

    print("\n" + "="*120)
    print("Decode Scheduler Profiling: 16 vs 32 Concurrency Comparison")
    print("="*120)
    print(f"\n{'Stage':<25} | {'16 Conc (ms)':^45} | {'32 Conc (ms)':^45} | {'Speedup':^10}")
    print("-"*120)
    print(f"{'':<25} | {'Min':>8} {'Avg':>8} {'P50':>8} {'P99':>8} {'Max':>8} | {'Min':>8} {'Avg':>8} {'P50':>8} {'P99':>8} {'Max':>8} | {'(P99)':>10}")
    print("-"*120)

    for stage in stages:
        s16 = stats_16[stage]
        s32 = stats_32[stage]

        if s16['p99'] > 0:
            speedup = s16['p99'] / s32['p99']
            speedup_str = f"{speedup:.2f}x"
        else:
            speedup_str = "N/A"

        print(f"{stage:<25} | "
              f"{s16['min']:>8.3f} {s16['avg']:>8.3f} {s16['p50']:>8.3f} {s16['p99']:>8.3f} {s16['max']:>8.3f} | "
              f"{s32['min']:>8.3f} {s32['avg']:>8.3f} {s32['p50']:>8.3f} {s32['p99']:>8.3f} {s32['max']:>8.3f} | "
              f"{speedup_str:>10}")

    print("="*120)

    # Print sample counts
    print(f"\nSample counts:")
    print(f"  16 concurrency: {stats_16['total']['count']:,} iterations")
    print(f"  32 concurrency: {stats_32['total']['count']:,} iterations")

    # Identify bottlenecks
    print("\n" + "="*120)
    print("Bottleneck Analysis (by P99 latency)")
    print("="*120)

    for label, stats in [("16 Concurrency", stats_16), ("32 Concurrency", stats_32)]:
        print(f"\n{label}:")
        sorted_stages = sorted(stats.items(), key=lambda x: x[1]['p99'], reverse=True)
        for i, (stage, s) in enumerate(sorted_stages[:5], 1):
            pct = (s['p99'] / stats['total']['p99'] * 100) if stats['total']['p99'] > 0 else 0
            print(f"  {i}. {stage:<25} P99={s['p99']:>8.3f}ms ({pct:>5.1f}%)")

if __name__ == '__main__':
    main()
