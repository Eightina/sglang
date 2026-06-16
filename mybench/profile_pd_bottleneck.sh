#!/bin/bash
# PD Mode Profiling Script
# 用于诊断 PD disaggregation 模式的性能瓶颈
#
# 使用方法:
#   ./profile_pd_bottleneck.sh [test_name] [input_len] [output_len] [num_prompts] [max_concurrency]
#
# 示例:
#   ./profile_pd_bottleneck.sh single 4096 512 10 1       # 单请求：清晰的时间线，分析各阶段
#   ./profile_pd_bottleneck.sh concurrent 4096 512 100 32 # 高并发：真实负载，分析 batch/流水线效果
#   ./profile_pd_bottleneck.sh quick 1024 128 5 1         # 快速测试

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


# 参数解析
TEST_NAME=${1:-"profile_$(date +%Y%m%d_%H%M%S)"}
INPUT_LEN=${2:-4096}
OUTPUT_LEN=${3:-512}
NUM_PROMPTS=${4:-10}
MAX_CONCURRENCY=${5:-32}  # 默认使用 32（与 benchmark 一致）
OUTPUT_DIR="./mybench/profiling/${TEST_NAME}"

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# 创建输出目录
mkdir -p "$OUTPUT_DIR"
log_info "输出目录: $OUTPUT_DIR"

# 检查 nsys 是否安装
if ! command -v nsys &> /dev/null; then
    log_error "nsys 未安装。请安装 NVIDIA Nsight Systems:"
    log_error "  sudo apt-get install nsight-systems"
    log_error "  或从 https://developer.nvidia.com/nsight-systems 下载"
    exit 1
fi

# 检查服务是否已经在运行
check_server_running() {
    if curl -s http://127.0.0.1:8000/health > /dev/null 2>&1; then
        return 0
    else
        return 1
    fi
}

# 清理函数
cleanup() {
    log_info "清理进程..."
    pkill -f "sglang.launch_server" 2>/dev/null || true
    pkill -f "sglang_router" 2>/dev/null || true
    sleep 2
}

# 启动 prefill worker（带 profiling）
start_prefill_worker() {
    log_info "启动 prefill worker（带 profiling）..."

    # 读取配置
    local CONFIG_FILE="./pyscripts/q-prefilltp2.yaml"
    if [ ! -f "$CONFIG_FILE" ]; then
        log_error "配置文件不存在: $CONFIG_FILE"
        exit 1
    fi

    # 使用 nsys profile
    nsys profile \
        --trace=cuda,nvtx,osrt \
        --cuda-memory-usage=true \
        --output="${OUTPUT_DIR}/prefill_profile" \
        --force-overwrite=true \
        --stats=true \
        python3 -m sglang.launch_server \
            --config "$CONFIG_FILE" \
            > "${OUTPUT_DIR}/prefill_server.log" 2>&1 &

    PREFILL_PID=$!
    echo $PREFILL_PID > "${OUTPUT_DIR}/prefill.pid"
    log_info "Prefill worker PID: $PREFILL_PID"
}

# 启动 decode worker（带 profiling）
start_decode_worker() {
    log_info "启动 decode worker（带 profiling）..."

    local CONFIG_FILE="./pyscripts/q-decodetp2.yaml"
    if [ ! -f "$CONFIG_FILE" ]; then
        log_error "配置文件不存在: $CONFIG_FILE"
        exit 1
    fi

    nsys profile \
        --trace=cuda,nvtx,osrt \
        --cuda-memory-usage=true \
        --output="${OUTPUT_DIR}/decode_profile" \
        --force-overwrite=true \
        --stats=true \
        python3 -m sglang.launch_server \
            --config "$CONFIG_FILE" \
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
    local MAX_WAIT=120
    local WAITED=0

    while ! check_server_running; do
        sleep 2
        WAITED=$((WAITED + 2))
        if [ $WAITED -ge $MAX_WAIT ]; then
            log_error "服务启动超时（${MAX_WAIT}s）"
            log_error "检查日志:"
            log_error "  tail -50 ${OUTPUT_DIR}/prefill_server.log"
            log_error "  tail -50 ${OUTPUT_DIR}/decode_server.log"
            cleanup
            exit 1
        fi
        printf "."
    done
    echo ""
    log_info "服务已就绪（等待了 ${WAITED}s）"
    sleep 5  # 额外等待稳定
}

# 运行 benchmark
run_benchmark() {
    log_info "运行 benchmark: input=$INPUT_LEN, output=$OUTPUT_LEN, prompts=$NUM_PROMPTS, concurrency=$MAX_CONCURRENCY"

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

# 生成统计报告
generate_stats() {
    log_info "生成统计报告..."

    # Prefill worker 统计
    if [ -f "${OUTPUT_DIR}/prefill_profile.nsys-rep" ]; then
        log_info "Prefill worker 统计:"
        nsys stats \
            --report cuda_gpu_kern_sum \
            --report nvtx_sum \
            "${OUTPUT_DIR}/prefill_profile.nsys-rep" \
            > "${OUTPUT_DIR}/prefill_stats.txt" 2>&1 || true

        echo "" >> "${OUTPUT_DIR}/prefill_stats.txt"
        echo "=== CUDA Kernel Summary ===" >> "${OUTPUT_DIR}/prefill_stats.txt"
        nsys stats \
            --report cuda_gpu_kern_sum \
            "${OUTPUT_DIR}/prefill_profile.nsys-rep" \
            >> "${OUTPUT_DIR}/prefill_stats.txt" 2>&1 || true
    fi

    # Decode worker 统计
    if [ -f "${OUTPUT_DIR}/decode_profile.nsys-rep" ]; then
        log_info "Decode worker 统计:"
        nsys stats \
            --report cuda_gpu_kern_sum \
            --report nvtx_sum \
            "${OUTPUT_DIR}/decode_profile.nsys-rep" \
            > "${OUTPUT_DIR}/decode_stats.txt" 2>&1 || true

        echo "" >> "${OUTPUT_DIR}/decode_stats.txt"
        echo "=== CUDA Kernel Summary ===" >> "${OUTPUT_DIR}/decode_stats.txt"
        nsys stats \
            --report cuda_gpu_kern_sum \
            "${OUTPUT_DIR}/decode_profile.nsys-rep" \
            >> "${OUTPUT_DIR}/decode_stats.txt" 2>&1 || true
    fi
}

# 分析瓶颈
analyze_bottleneck() {
    log_info "分析性能瓶颈..."

    cat > "${OUTPUT_DIR}/analysis.md" << 'EOF'
# PD Mode 性能瓶颈分析报告

## 1. 时间分解

从 benchmark 输出中提取关键指标：

EOF

    # 提取 benchmark 结果
    if [ -f "${OUTPUT_DIR}/benchmark_output.txt" ]; then
        echo '```' >> "${OUTPUT_DIR}/analysis.md"
        grep -E "(Mean TTFT|Mean ITL|Mean TPOT|Output token throughput)" \
            "${OUTPUT_DIR}/benchmark_output.txt" \
            >> "${OUTPUT_DIR}/analysis.md" || true
        echo '```' >> "${OUTPUT_DIR}/analysis.md"
    fi

    cat >> "${OUTPUT_DIR}/analysis.md" << 'EOF'

## 2. Prefill Worker 分析

### CUDA Kernel 耗时分布

EOF

    if [ -f "${OUTPUT_DIR}/prefill_stats.txt" ]; then
        echo '```' >> "${OUTPUT_DIR}/analysis.md"
        grep -A 20 "CUDA Kernel Summary" "${OUTPUT_DIR}/prefill_stats.txt" \
            >> "${OUTPUT_DIR}/analysis.md" || echo "无数据" >> "${OUTPUT_DIR}/analysis.md"
        echo '```' >> "${OUTPUT_DIR}/analysis.md"
    else
        echo "无 profiling 数据" >> "${OUTPUT_DIR}/analysis.md"
    fi

    cat >> "${OUTPUT_DIR}/analysis.md" << 'EOF'

## 3. Decode Worker 分析

### CUDA Kernel 耗时分布

EOF

    if [ -f "${OUTPUT_DIR}/decode_stats.txt" ]; then
        echo '```' >> "${OUTPUT_DIR}/analysis.md"
        grep -A 20 "CUDA Kernel Summary" "${OUTPUT_DIR}/decode_stats.txt" \
            >> "${OUTPUT_DIR}/analysis.md" || echo "无数据" >> "${OUTPUT_DIR}/analysis.md"
        echo '```' >> "${OUTPUT_DIR}/analysis.md"
    else
        echo "无 profiling 数据" >> "${OUTPUT_DIR}/analysis.md"
    fi

    cat >> "${OUTPUT_DIR}/analysis.md" << 'EOF'

## 4. 瓶颈诊断指南

### 如何判断瓶颈在哪里？

#### 场景 1: Prefill 计算瓶颈
- **症状**: prefill_stats.txt 中 CUDA kernel 耗时占比 > 80%
- **特征**: GPU-Util 持续 > 90%
- **解决**:
  - 使用更小的 TP size（减少 AllReduce）
  - 启用 chunked prefill（`--chunked-prefill-size 2048`）
  - 使用 FP8 计算（`--quantization fp8`）

#### 场景 2: KV Transfer 瓶颈
- **症状**: TTFT 远大于 ITL × output_len
- **特征**:
  - PCIe 带宽利用率低（< 30%）
  - NVTX 中 KV transfer 相关 marker 耗时长
- **解决**:
  - 启用 FP8 KV cache（`--kv-cache-dtype fp8_e4m3`）
  - 优化 NIXL 配置（调整 buffer size）
  - 升级到 NVLink 硬件

#### 场景 3: Decode 计算瓶颈
- **症状**: ITL 很高（> 50ms）
- **特征**: decode_stats.txt 中 attention kernel 耗时长
- **解决**:
  - 使用 FlashDecoding（自动启用）
  - 减少 batch size
  - 使用 speculative decoding

#### 场景 4: 调度开销
- **症状**: CPU 时间占比高，GPU 利用率低
- **特征**: prefill_stats.txt 中 osrt (OS runtime) 占比 > 20%
- **解决**:
  - 减少 `--scheduler-recv-interval`
  - 优化 Python 代码（减少 GIL 竞争）

## 5. 带宽计算

### PCIe 带宽利用率

从 nsys stats 中提取 NCCL/NIXL 相关 kernel：

```bash
# 查找通信相关 kernel
grep -i "nccl\|allreduce\|allgather\|send\|recv" prefill_stats.txt
```

计算公式：
```
实际带宽 = 传输数据量 / 传输时间
利用率 = 实际带宽 / 理论带宽（PCIe 3.0 x16 = 32 GB/s）
```

### AllReduce 带宽

对于 TP=2，每次 AllReduce 传输：
```
数据量 = 模型参数 / TP_size × 4 bytes (FP32)
       = 27B / 2 × 4 = 54 GB（对于 27B 模型）

实际带宽 = 54 GB / AllReduce 时间
利用率 = 实际带宽 / 32 GB/s
```

## 6. 下一步行动

根据分析结果，选择对应的优化方向：

1. **如果 KV transfer 是瓶颈**:
   - 查看 `mybench/comparison-tp4-vs-pdtp2.md` 第 8 节
   - 优先实现请求级流水线（P0）
   - 考虑 FP8 KV cache（P1）

2. **如果 Prefill 是瓶颈**:
   - 调整 `--chunked-prefill-size`
   - 优化 attention backend

3. **如果 Decode 是瓶颈**:
   - 检查 batch size 是否过大
   - 考虑 speculative decoding

EOF

    log_info "分析报告已生成: ${OUTPUT_DIR}/analysis.md"
}

# 主流程
main() {
    log_info "=== PD Mode Profiling Script ==="
    log_info "测试名称: $TEST_NAME"
    log_info "参数: input=$INPUT_LEN, output=$OUTPUT_LEN, prompts=$NUM_PROMPTS"

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

    # 停止服务（触发 nsys 生成报告）
    log_info "停止服务，生成 profiling 报告..."
    cleanup
    sleep 5  # 等待 nsys 完成

    # 生成统计
    generate_stats

    # 分析瓶颈
    analyze_bottleneck

    # 输出总结
    log_info "=== Profiling 完成 ==="
    log_info "输出文件:"
    log_info "  - Benchmark 结果: ${OUTPUT_DIR}/benchmark_output.txt"
    log_info "  - Prefill profile: ${OUTPUT_DIR}/prefill_profile.nsys-rep"
    log_info "  - Decode profile: ${OUTPUT_DIR}/decode_profile.nsys-rep"
    log_info "  - Prefill stats: ${OUTPUT_DIR}/prefill_stats.txt"
    log_info "  - Decode stats: ${OUTPUT_DIR}/decode_stats.txt"
    log_info "  - 分析报告: ${OUTPUT_DIR}/analysis.md"
    log_info ""
    log_info "查看报告:"
    log_info "  cat ${OUTPUT_DIR}/analysis.md"
    log_info ""
    log_info "查看详细 stats:"
    log_info "  cat ${OUTPUT_DIR}/prefill_stats.txt"
    log_info "  cat ${OUTPUT_DIR}/decode_stats.txt"
}

# 运行主流程
main
