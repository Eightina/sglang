#!/usr/bin/env python3
"""分析 KV transfer 数据"""

import re
import json
import sys
from pathlib import Path

def parse_prometheus_metrics(filepath):
    """解析 Prometheus metrics 文件"""
    metrics = {}
    if not Path(filepath).exists():
        return metrics

    with open(filepath) as f:
        content = f.read()

    # 提取 KV transfer 相关的 histograms
    patterns = [
        (r'sglang:kv_transfer_speed_gb_s_sum\s+([\d.]+)', 'speed_sum'),
        (r'sglang:kv_transfer_speed_gb_s_count\s+(\d+)', 'speed_count'),
        (r'sglang:kv_transfer_latency_ms_sum\s+([\d.]+)', 'latency_sum'),
        (r'sglang:kv_transfer_latency_ms_count\s+(\d+)', 'latency_count'),
        (r'sglang:kv_transfer_total_mb_sum\s+([\d.]+)', 'size_sum'),
        (r'sglang:kv_transfer_total_mb_count\s+(\d+)', 'size_count'),
    ]

    for pattern, key in patterns:
        match = re.search(pattern, content)
        if match:
            metrics[key] = float(match.group(1))

    return metrics

def parse_server_logs(filepath):
    """解析 server logs 中的 KV transfer 信息"""
    transfers = []
    if not Path(filepath).exists():
        return transfers

    with open(filepath) as f:
        content = f.read()

    # 匹配 ReqTimeStats 中的 transfer_speed 和 transfer_total
    # 例如: transfer_speed=12.34 GB/s, transfer_total=567.89 MB
    pattern = r'transfer_speed=([\d.]+) GB/s, transfer_total=([\d.]+) MB'
    matches = re.findall(pattern, content)

    for speed, size in matches:
        transfers.append({
            'speed_gb_s': float(speed),
            'total_mb': float(size),
        })

    return transfers

def parse_loads_snapshot(filepath):
    """解析 /v1/loads 快照"""
    if not Path(filepath).exists():
        return {}

    with open(filepath) as f:
        data = json.load(f)

    return data.get('disagg', {})

def main():
    output_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path('.')

    print("=" * 80)
    print("KV Transfer 分析报告")
    print("=" * 80)

    # 1. 分析 Prometheus metrics
    print("\n【1】Prometheus Metrics (Prefill Worker)")
    print("-" * 80)
    prefill_metrics = parse_prometheus_metrics(output_dir / 'prefill_metrics.txt')

    if prefill_metrics:
        speed_count = prefill_metrics.get('speed_count', 0)
        if speed_count > 0:
            avg_speed = prefill_metrics.get('speed_sum', 0) / speed_count
            avg_latency = prefill_metrics.get('latency_sum', 0) / prefill_metrics.get('latency_count', 1)
            avg_size = prefill_metrics.get('size_sum', 0) / prefill_metrics.get('size_count', 1)

            print(f"  请求数量: {int(speed_count)}")
            print(f"  平均传输速度: {avg_speed:.2f} GB/s")
            print(f"  平均传输延迟: {avg_latency:.2f} ms")
            print(f"  平均传输大小: {avg_size:.2f} MB")
        else:
            print("  无 KV transfer 数据")
    else:
        print("  未找到 metrics 文件")

    # 2. 分析 Server logs
    print("\n【2】Server Logs (逐请求分析)")
    print("-" * 80)
    transfers = parse_server_logs(output_dir / 'prefill_server.log')

    if transfers:
        speeds = [t['speed_gb_s'] for t in transfers]
        sizes = [t['total_mb'] for t in transfers]

        print(f"  请求数量: {len(transfers)}")
        print(f"  传输速度: 平均={sum(speeds)/len(speeds):.2f} GB/s, "
              f"最小={min(speeds):.2f} GB/s, 最大={max(speeds):.2f} GB/s")
        print(f"  传输大小: 平均={sum(sizes)/len(sizes):.2f} MB, "
              f"最小={min(sizes):.2f} MB, 最大={max(sizes):.2f} MB")

        # 计算 PCIe 带宽利用率（假设 PCIe 3.0 x16 = 32 GB/s）
        pcie_bandwidth = 32.0  # GB/s
        avg_speed = sum(speeds) / len(speeds)
        utilization = (avg_speed / pcie_bandwidth) * 100
        print(f"\n  PCIe 带宽利用率: {utilization:.1f}% (基于 PCIe 3.0 x16 = 32 GB/s)")

        if utilization < 30:
            print("  ⚠️  带宽利用率低，可能存在协议开销或调度瓶颈")
        elif utilization > 70:
            print("  ✅ 带宽利用率高，接近硬件极限")
        else:
            print("  ℹ️  带宽利用率中等")
    else:
        print("  未找到 KV transfer 日志")

    # 3. 分析 /v1/loads 快照
    print("\n【3】/v1/loads 快照（最新值）")
    print("-" * 80)
    loads = parse_loads_snapshot(output_dir / 'loads_snapshot.json')

    if loads:
        speed = loads.get('kv_transfer_speed_gb_s', 0)
        latency = loads.get('kv_transfer_latency_ms', 0)
        print(f"  最新传输速度: {speed:.2f} GB/s")
        print(f"  最新传输延迟: {latency:.2f} ms")
    else:
        print("  未找到 loads 快照")

    # 4. 总结
    print("\n" + "=" * 80)
    print("瓶颈诊断")
    print("=" * 80)

    if transfers:
        avg_speed = sum(speeds) / len(speeds)
        avg_latency_ms = (sum(sizes) / len(sizes)) / avg_speed  # 估算平均延迟

        print(f"\nKV Transfer 性能:")
        print(f"  平均速度: {avg_speed:.2f} GB/s")
        print(f"  估算平均延迟: {avg_latency_ms:.2f} ms")

        # 与 benchmark 结果对比
        benchmark_file = output_dir / 'benchmark_output.txt'
        if benchmark_file.exists():
            with open(benchmark_file) as f:
                benchmark_content = f.read()

            # 提取 TTFT
            ttft_match = re.search(r'Mean TTFT \(ms\)\|.*?\|.*?\|([\d.]+)', benchmark_content)
            if ttft_match:
                ttft = float(ttft_match.group(1))
                print(f"\n与 Benchmark 对比:")
                print(f"  Mean TTFT: {ttft:.2f} ms")
                print(f"  KV Transfer 延迟: {avg_latency_ms:.2f} ms")
                print(f"  KV Transfer 占比: {(avg_latency_ms / ttft * 100):.1f}%")

                if avg_latency_ms / ttft > 0.5:
                    print("\n  🔴 KV Transfer 是主要瓶颈（占 TTFT > 50%）")
                elif avg_latency_ms / ttft > 0.3:
                    print("\n  🟡 KV Transfer 是重要瓶颈（占 TTFT 30-50%）")
                else:
                    print("\n  🟢 KV Transfer 不是主要瓶颈（占 TTFT < 30%）")

    print("\n" + "=" * 80)

if __name__ == '__main__':
    main()
