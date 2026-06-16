#!/bin/bash
# 测量 KV Cache 传输耗时
# 通过 Prometheus metrics 和 server logs 获取 KV transfer 数据

set -e

# 代理设置
export http_proxy="http://lab22-squid.eng.xrvm.cn:3128"
export https_proxy="http://lab22-squid.eng.xrvm.cn:3128"
export HTTP_PROXY="http://lab22-squid.eng.xrvm.cn:3128"
export HTTPS_PROXY="http://lab22-squid.eng.xrvm.cn:3128"
export no_proxy=localhost,127.0.0.1,.local,.xrvm.cn,10.244.1.235,0.0.0.0

# SGLang 配置
export SGLANG_USE_MODELSCOPE=true
export HF_ENDPOINT=https://hf-mirror.com

# HPC-X UCX 配置
export PATH=/opt/hpcx/ucx/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:/opt/hpcx/ucx/lib:/opt/hpcx/ucx/lib/ucx:$LD_LIBRARY_PATH
export UCX_MODULE_DIR=/opt/hpcx/ucx/lib/ucx

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }
log_section() { echo -e "${BLUE}[SECTION]${NC} $1"; }

# 参数
INPUT_LEN=${1:-4096}
OUTPUT_LEN=${2:-512}
NUM_PROMPTS=${3:-10}
MAX_CONCURRENCY=${4:-32}
OUTPUT_DIR="./mybench/kv-transfer-measurement/$(date +%Y%m%d_%H%M%S)"

log_section "KV Transfer Measurement"
log_info "参数: input=$INPUT_LEN, output=$OUTPUT_LEN, prompts=$NUM_PROMPTS, concurrency=$MAX_CONCURRENCY"
log_info "输出目录: $OUTPUT_DIR"

mkdir -p "$OUTPUT_DIR"

# 清理函数
cleanup() {
    log_info "清理进程..."
    pkill -f "sglang.launch_server" 2>/dev/null || true
    pkill -f "sglang_router" 2>/dev/null || true
    sleep 2
}

# 启动 prefill worker（带 metrics）
start_prefill_worker() {
    log_info "启动 prefill worker（带 metrics）..."

    python3 -m sglang.launch_server \
        --config ./pyscripts/q-prefilltp2.yaml \
        --enable-metrics \
        --enable-request-time-stats-logging \
        > "${OUTPUT_DIR}/prefill_server.log" 2>&1 &

    PREFILL_PID=$!
    echo $PREFILL_PID > "${OUTPUT_DIR}/prefill.pid"
    log_info "Prefill worker PID: $PREFILL_PID"
}

# 启动 decode worker（带 metrics）
start_decode_worker() {
    log_info "启动 decode worker（带 metrics）..."

    python3 -m sglang.launch_server \
        --config ./pyscripts/q-decodetp2.yaml \
        --enable-metrics \
        --enable-request-time-stats-logging \
        > "${OUTPUT_DIR}/decode_server.log" 2>&1 &

    DECODE_PID=$!
    echo $DECODE_PID > "${OUTPUT_DIR}/decode.pid"
    log_info "Decode worker PID: $DECODE_PID"
}

# 启动 router
start_router() {
    log_info "启动 router..."

    python3 -m sglang_router.launch_router \
        --pd-disaggregation \
        --prefill "http://0.0.0.0:30000" \
        --decode "http://0.0.0.0:30001" \
        --policy round_robin \
        --host 0.0.0.0 \
        --port 8000 \
        > "${OUTPUT_DIR}/router.log" 2>&1 &

    ROUTER_PID=$!
    echo $ROUTER_PID > "${OUTPUT_DIR}/router.pid"
    log_info "Router PID: $ROUTER_PID"
}

# 等待服务就绪
wait_for_servers() {
    log_info "等待服务启动..."
    local MAX_WAIT=300
    local WAITED=0

    # 等待 prefill worker (30000) 就绪
    log_info "等待 prefill worker (port 30000)..."
    while ! curl -s http://127.0.0.1:30000/health > /dev/null 2>&1; do
        sleep 2
        WAITED=$((WAITED + 2))
        if [ $WAITED -ge $MAX_WAIT ]; then
            log_error "Prefill worker 启动超时（${MAX_WAIT}s）"
            cleanup
            exit 1
        fi
        printf "."
    done
    echo ""
    log_info "Prefill worker 已就绪（等待了 ${WAITED}s）"

    # 等待 decode worker (30001) 就绪
    log_info "等待 decode worker (port 30001)..."
    while ! curl -s http://127.0.0.1:30001/health > /dev/null 2>&1; do
        sleep 2
        WAITED=$((WAITED + 2))
        if [ $WAITED -ge $MAX_WAIT ]; then
            log_error "Decode worker 启动超时（${MAX_WAIT}s）"
            cleanup
            exit 1
        fi
        printf "."
    done
    echo ""
    log_info "Decode worker 已就绪（等待了 ${WAITED}s）"

    # 等待 router (8000) 就绪
    log_info "等待 router (port 8000)..."
    while ! curl -s http://127.0.0.1:8000/health > /dev/null 2>&1; do
        sleep 2
        WAITED=$((WAITED + 2))
        if [ $WAITED -ge $MAX_WAIT ]; then
            log_error "Router 启动超时（${MAX_WAIT}s）"
            cleanup
            exit 1
        fi
        printf "."
    done
    echo ""
    log_info "所有服务已就绪（总共等待了 ${WAITED}s）"
    sleep 5
}

# 运行 benchmark
run_benchmark() {
    log_info "运行 benchmark..."

    python3 -m sglang.bench_serving \
        --host 127.0.0.1 \
        --port 8000 \
        --dataset-name random-ids \
        --random-input-len "$INPUT_LEN" \
        --random-output-len "$OUTPUT_LEN" \
        --num-prompts "$NUM_PROMPTS" \
        --max-concurrency "$MAX_CONCURRENCY" \
        --output-file "${OUTPUT_DIR}/benchmark_results.jsonl" \
        | tee "${OUTPUT_DIR}/benchmark_output.txt"

    log_info "Benchmark 完成"
}

# 收集 Prometheus metrics
collect_prometheus_metrics() {
    log_info "收集 Prometheus metrics..."

    # 从 prefill worker 获取 metrics
    curl -s http://127.0.0.1:30000/metrics > "${OUTPUT_DIR}/prefill_metrics.txt" 2>/dev/null || true

    # 从 decode worker 获取 metrics
    curl -s http://127.0.0.1:30001/metrics > "${OUTPUT_DIR}/decode_metrics.txt" 2>/dev/null || true

    log_info "Prometheus metrics 已保存"
}

# 收集 /v1/loads 快照
collect_loads_snapshot() {
    log_info "收集 /v1/loads 快照..."

    curl -s "http://127.0.0.1:8000/v1/loads?include=all" > "${OUTPUT_DIR}/loads_snapshot.json" 2>/dev/null || true

    log_info "Loads snapshot 已保存"
}

# 分析 KV transfer 数据
analyze_kv_transfer() {
    log_section "KV Transfer 分析"

    cat > "${OUTPUT_DIR}/analysis.py" << 'PYEOF'
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
PYEOF

    chmod +x "${OUTPUT_DIR}/analysis.py"
    python3 "${OUTPUT_DIR}/analysis.py" "${OUTPUT_DIR}" | tee "${OUTPUT_DIR}/kv_transfer_analysis.txt"
}

# 主流程
main() {
    log_section "开始测量"

    # 清理旧进程
    cleanup

    # 启动服务
    start_prefill_worker
    start_decode_worker
    start_router

    # 等待就绪
    wait_for_servers

    # 运行 benchmark
    run_benchmark

    # 收集数据
    collect_prometheus_metrics
    collect_loads_snapshot

    # 停止服务
    log_info "停止服务..."
    cleanup
    sleep 3

    # 分析数据
    analyze_kv_transfer

    # 输出总结
    log_section "测量完成"
    log_info "输出文件:"
    log_info "  - Benchmark 结果: ${OUTPUT_DIR}/benchmark_output.txt"
    log_info "  - Prefill server log: ${OUTPUT_DIR}/prefill_server.log"
    log_info "  - Prometheus metrics: ${OUTPUT_DIR}/prefill_metrics.txt"
    log_info "  - Loads snapshot: ${OUTPUT_DIR}/loads_snapshot.json"
    log_info "  - KV transfer 分析: ${OUTPUT_DIR}/kv_transfer_analysis.txt"
    log_info ""
    log_info "查看分析结果:"
    log_info "  cat ${OUTPUT_DIR}/kv_transfer_analysis.txt"
}

# 运行主流程
main
