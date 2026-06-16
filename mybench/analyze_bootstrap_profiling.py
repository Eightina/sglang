#!/usr/bin/env python3
"""
Bootstrap Profiling 分析脚本

从 server log 中提取 bootstrap profiling 数据并生成报告。
"""

import re
import json
import sys
from pathlib import Path
from collections import defaultdict


def extract_profiling_data(log_file: Path) -> dict:
    """从日志文件中提取 profiling 数据"""
    profiles = defaultdict(list)

    # 匹配 profiling 日志行
    # 格式: [BOOTSTRAP PROFILE] <event>: <duration>ms, <details>
    pattern = r'\[BOOTSTRAP PROFILE\] (\w+): ([\d.]+)ms(.*)'

    with open(log_file, 'r') as f:
        for line in f:
            match = re.search(pattern, line)
            if match:
                event = match.group(1)
                duration = float(match.group(2))
                details = match.group(3).strip()

                # 解析详情
                detail_dict = {}
                if details:
                    for item in details.split(','):
                        if '=' in item:
                            key, value = item.split('=', 1)
                            key = key.strip()
                            value = value.strip()
                            try:
                                detail_dict[key] = float(value)
                            except ValueError:
                                detail_dict[key] = value

                profiles[event].append({
                    'duration_ms': duration,
                    'details': detail_dict
                })

    return dict(profiles)


def analyze_bootstrap_stages(profiles: dict) -> dict:
    """分析 bootstrap 各阶段耗时"""
    stages = defaultdict(list)

    for event, data_list in profiles.items():
        for data in data_list:
            duration = data['duration_ms']
            stages[event].append(duration)

    return dict(stages)


def print_summary(stages: dict):
    """打印汇总统计"""
    print("\n" + "="*80)
    print("Bootstrap Profiling Summary")
    print("="*80)

    total_samples = sum(len(v) for v in stages.values())
    print(f"\n总样本数: {total_samples}")

    print("\n各阶段耗时统计（毫秒）:")
    print("-" * 80)
    print(f"{'阶段':<50} {'平均':>10} {'最小':>10} {'最大':>10} {'样本数':>8}")
    print("-" * 80)

    for stage_name in sorted(stages.keys()):
        values = stages[stage_name]
        if values:
            avg = sum(values) / len(values)
            min_val = min(values)
            max_val = max(values)
            count = len(values)
            print(f"{stage_name:<50} {avg:>10.2f} {min_val:>10.2f} {max_val:>10.2f} {count:>8}")

    print("="*80)


def main():
    if len(sys.argv) < 2:
        print("Usage: python analyze_bootstrap_profiling.py <log_file>")
        print("Example: python analyze_bootstrap_profiling.py mybench/kv-transfer-measurement/20260613_103816/prefill_server.log")
        sys.exit(1)

    log_file = Path(sys.argv[1])
    if not log_file.exists():
        print(f"Error: File not found: {log_file}")
        sys.exit(1)

    print(f"Analyzing bootstrap profiling data from: {log_file}")

    profiles = extract_profiling_data(log_file)
    if not profiles:
        print("No profiling data found in log file.")
        print("Make sure profiling is enabled (SGLANG_BOOTSTRAP_PROFILE=1)")
        sys.exit(1)

    stages = analyze_bootstrap_stages(profiles)
    print_summary(stages)

    # 输出 JSON 格式
    output_file = log_file.parent / "bootstrap_profiling.json"
    with open(output_file, 'w') as f:
        json.dump({"stages": stages, "profiles": profiles}, f, indent=2)
    print(f"\nDetailed data saved to: {output_file}")


if __name__ == "__main__":
    main()
