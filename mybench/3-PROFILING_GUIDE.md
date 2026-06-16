# PD Mode Profiling 使用指南

## 快速开始

### 1. 运行 Profiling 脚本

```bash
# 基本用法
./mybench/profile_pd_bottleneck.sh

# 指定参数
./mybench/profile_pd_bottleneck.sh test1 4096 512 10
# 参数说明：
#   test1: 测试名称（用于创建输出目录）
#   4096: input length
#   512: output length
#   10: 请求数量

# 快速测试（少量请求）
./mybench/profile_pd_bottleneck.sh quick 1024 128 5
```

### 2. 查看结果

脚本会在 `./mybench/profiling/<test_name>/` 目录下生成：

```
profiling/test1/
├── benchmark_output.txt          # Benchmark 输出（包含 TTFT/ITL/吞吐等）
├── prefill_profile.nsys-rep      # Prefill worker 的 profiling 数据
├── decode_profile.nsys-rep       # Decode worker 的 profiling 数据
├── prefill_stats.txt             # Prefill worker 的统计信息（命令行友好）
├── decode_stats.txt              # Decode worker 的统计信息
├── analysis.md                   # 自动生成的分析报告
├── prefill_server.log            # Prefill 服务日志
├── decode_server.log             # Decode 服务日志
└── router.log                    # Router 日志
```

### 3. 快速查看瓶颈

```bash
# 查看分析报告
cat ./mybench/profiling/test1/analysis.md

# 查看 benchmark 结果
cat ./mybench/profiling/test1/benchmark_output.txt

# 查看 prefill worker 的 kernel 耗时
cat ./mybench/profiling/test1/prefill_stats.txt

# 查看 decode worker 的 kernel 耗时
cat ./mybench/profiling/test1/decode_stats.txt
```

## 无 GUI 分析方法

### 方法 1: 使用 nsys stats（推荐）

`nsys stats` 可以从 `.nsys-rep` 文件生成命令行友好的统计报告。

```bash
# 进入 profiling 目录
cd ./mybench/profiling/test1/

# 查看 CUDA kernel 耗时排名（按时间排序）
nsys stats --report cuda_gpu_kern_sum prefill_profile.nsys-rep

# 查看 NVTX marker 耗时（用于识别不同阶段）
nsys stats --report nvtx_sum prefill_profile.nsys-rep

# 查看 CUDA API 调用
nsys stats --report cuda_api_sum prefill_profile.nsys-rep

# 查看 OS runtime 调用
nsys stats --report osrt_sum prefill_profile.nsys-rep

# 导出为 CSV 格式（方便用 awk/grep 处理）
nsys stats --report cuda_gpu_kern_sum --format csv prefill_profile.nsys-rep > kernels.csv
```

### 方法 2: 使用 nsys export（生成文本格式）

```bash
# 导出为 SQLite 数据库（可以用 sqlite3 查询）
nsys export --type=sqlite prefill_profile.nsys-rep -o prefill.db

# 用 sqlite3 查询
sqlite3 prefill.db <<EOF
-- 查询耗时最长的 10 个 CUDA kernel
SELECT name, SUM(end - start) / 1000000 as total_ms
FROM CUPTI_ACTIVITY_KIND_KERNEL
GROUP BY name
ORDER BY total_ms DESC
LIMIT 10;

-- 查询 NCCL 相关 kernel（通信）
SELECT name, COUNT(*) as count, SUM(end - start) / 1000000 as total_ms
FROM CUPTI_ACTIVITY_KIND_KERNEL
WHERE name LIKE '%nccl%' OR name LIKE '%allreduce%'
GROUP BY name;

-- 查询 NVTX 标记（识别不同阶段）
SELECT text, SUM(end - start) / 1000000 as total_ms
FROM NVTX_EVENTS
GROUP BY text
ORDER BY total_ms DESC;
EOF
```

### 方法 3: 实时监控带宽使用率

```bash
# 方法 A: 使用 nvidia-smi（简单但粗糙）
watch -n 0.5 nvidia-smi

# 方法 B: 使用 DCGM（更精确）
# 安装 DCGM
sudo apt-get install datacenter-gpu-manager

# 启动 DCGM 服务
sudo systemctl start nvidia-dcgm

# 监控 PCIe 带宽（每 1 秒刷新）
dcgmi dmon -e 1011,1012 -d 1000
# 1011: PCIe TX bytes (发送)
# 1012: PCIe RX bytes (接收)

# 监控 GPU 利用率
dcgmi dmon -e 1015,1016 -d 1000
# 1015: SM utilization
# 1016: Memory utilization
```

### 方法 4: 使用 Python 脚本分析

创建一个简单的分析脚本：

```python
#!/usr/bin/env python3
# analyze_profile.py

import sqlite3
import pandas as pd

def analyze_prefill(db_path):
    """分析 prefill worker 的瓶颈"""
    conn = sqlite3.connect(db_path)
    
    # 1. 查询 CUDA kernel 耗时
    kernel_df = pd.read_sql("""
        SELECT name, 
               COUNT(*) as count,
               SUM(end - start) / 1000000 as total_ms,
               AVG(end - start) / 1000000 as avg_ms
        FROM CUPTI_ACTIVITY_KIND_KERNEL
        GROUP BY name
        ORDER BY total_ms DESC
    """, conn)
    
    print("=== Top 10 CUDA Kernels (by total time) ===")
    print(kernel_df.head(10).to_string(index=False))
    
    # 2. 查询通信相关 kernel
    comm_df = pd.read_sql("""
        SELECT name, 
               COUNT(*) as count,
               SUM(end - start) / 1000000 as total_ms
        FROM CUPTI_ACTIVITY_KIND_KERNEL
        WHERE name LIKE '%nccl%' OR name LIKE '%allreduce%' OR name LIKE '%allgather%'
        GROUP BY name
        ORDER BY total_ms DESC
    """, conn)
    
    print("\n=== Communication Kernels (NCCL/AllReduce) ===")
    if len(comm_df) > 0:
        print(comm_df.to_string(index=False))
        total_comm_time = comm_df['total_ms'].sum()
        print(f"\nTotal communication time: {total_comm_time:.2f} ms")
    else:
        print("No communication kernels found")
    
    # 3. 查询 NVTX 标记
    nvtx_df = pd.read_sql("""
        SELECT text,
               COUNT(*) as count,
               SUM(end - start) / 1000000 as total_ms
        FROM NVTX_EVENTS
        WHERE text IS NOT NULL
        GROUP BY text
        ORDER BY total_ms DESC
    """, conn)
    
    print("\n=== NVTX Markers (Stage Identification) ===")
    print(nvtx_df.head(20).to_string(index=False))
    
    # 4. 计算瓶颈分析
    total_kernel_time = kernel_df['total_ms'].sum()
    comm_time = comm_df['total_ms'].sum() if len(comm_df) > 0 else 0
    compute_time = total_kernel_time - comm_time
    
    print("\n=== Bottleneck Analysis ===")
    print(f"Total kernel time: {total_kernel_time:.2f} ms")
    print(f"  - Compute time: {compute_time:.2f} ms ({compute_time/total_kernel_time*100:.1f}%)")
    print(f"  - Communication time: {comm_time:.2f} ms ({comm_time/total_kernel_time*100:.1f}%)")
    
    if comm_time / total_kernel_time > 0.3:
        print("\n⚠️  Communication overhead is high (>30%)")
        print("   → Consider: FP8 KV cache, request-level pipelining")
    else:
        print("\n✓ Communication overhead is acceptable")
    
    conn.close()

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python3 analyze_profile.py <db_path>")
        print("Example: python3 analyze_profile.py prefill.db")
        sys.exit(1)
    
    analyze_prefill(sys.argv[1])
```

使用方法：

```bash
# 导出 profiling 数据为 SQLite
nsys export --type=sqlite prefill_profile.nsys-rep -o prefill.db

# 运行分析脚本
python3 analyze_profile.py prefill.db
```

## 常见瓶颈诊断

### 场景 1: TTFT 很高（> 5s）

**诊断步骤**：

```bash
# 1. 查看 benchmark 结果
grep "Mean TTFT" benchmark_output.txt
# 输出: Mean TTFT (ms):  10794.35

# 2. 查看 prefill worker 的 kernel 耗时
nsys stats --report cuda_gpu_kern_sum prefill_profile.nsys-rep | head -20

# 3. 查找通信相关 kernel
nsys stats --report cuda_gpu_kern_sum prefill_profile.nsys-rep | grep -i "nccl\|allreduce"

# 4. 计算通信占比
total_time=$(nsys stats --report cuda_gpu_kern_sum prefill_profile.nsys-rep | awk '/Total/ {print $3}')
comm_time=$(nsys stats --report cuda_gpu_kern_sum prefill_profile.nsys-rep | grep -i "nccl\|allreduce" | awk '{sum+=$3} END {print sum}')
echo "Communication ratio: $(echo "scale=2; $comm_time / $total_time * 100" | bc)%"
```

**可能的原因**：
- KV transfer 耗时过长（PCIe 带宽不足或协议开销）
- Prefill 计算量大（input 太长）
- 调度延迟

### 场景 2: ITL 很高（> 50ms）

**诊断步骤**：

```bash
# 1. 查看 benchmark 结果
grep "Mean ITL" benchmark_output.txt

# 2. 查看 decode worker 的 kernel 耗时
nsys stats --report cuda_gpu_kern_sum decode_profile.nsys-rep | head -20

# 3. 查找 attention 相关 kernel
nsys stats --report cuda_gpu_kern_sum decode_profile.nsys-rep | grep -i "attention\|flash"

# 4. 检查 batch size
grep "max_running_requests" decode_server.log
```

**可能的原因**：
- Decode batch size 过大
- Attention kernel 效率低
- 显存带宽不足

### 场景 3: 吞吐量低

**诊断步骤**：

```bash
# 1. 查看 benchmark 结果
grep "Output token throughput" benchmark_output.txt

# 2. 查看 GPU 利用率
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv -l 1

# 3. 查看 prefill 和 decode 的 kernel 耗时
echo "=== Prefill ==="
nsys stats --report cuda_gpu_kern_sum prefill_profile.nsys-rep | grep "Total"
echo "=== Decode ==="
nsys stats --report cuda_gpu_kern_sum decode_profile.nsep | grep "Total"

# 4. 检查是否有大量 CPU 开销
nsys stats --report osrt_sum prefill_profile.nsys-rep | head -20
```

**可能的原因**：
- GPU 利用率低（调度问题）
- CPU 瓶颈（Python GIL）
- 显存不足（频繁 swap）

## 高级技巧

### 1. 自定义 NVTX 标记

在 SGLang 代码中添加自定义 NVTX 标记，便于识别不同阶段：

```python
# python/sglang/srt/disaggregation/prefill.py

import torch.cuda.nvtx as nvtx

def process_prefill_chunk(self):
    nvtx.range_push("prefill_compute")
    # ... prefill 计算 ...
    nvtx.range_pop()
    
    nvtx.range_push("kv_transfer")
    self.send_kv_chunk(...)
    nvtx.range_pop()
```

### 2. 使用 CUDA Event 精确计时

```python
# python/sglang/srt/disaggregation/prefill.py

import torch

def process_prefill_chunk(self):
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    start_event.record()
    # ... prefill 计算 ...
    end_event.record()
    torch.cuda.synchronize()
    
    compute_time = start_event.elapsed_time(end_event)
    logger.info(f"Prefill compute time: {compute_time:.3f} ms")
```

### 3. 使用 py-spy 分析 Python 代码

```bash
# 安装 py-spy
pip install py-spy

# 对运行中的服务进行 CPU profiling
py-spy record -o profile.svg --pid $(cat prefill.pid)

# 生成火焰图（可以用浏览器打开）
# 查看 Python 代码的瓶颈
```

### 4. 使用 DCGM Exporter + Prometheus

```bash
# 启动 DCGM Exporter
docker run -d --gpus all -p 9400:9400 \
  nvcr.io/nvidia/k8s/dcgm-exporter:3.3.0-3.2.0-ubuntu22.04

# 查询 metrics
curl http://localhost:9400/metrics | grep DCGM

# 关键 metrics:
# - DCGM_FI_DEV_PCIE_TX_THROUGHPUT: PCIe 发送带宽
# - DCGM_FI_DEV_PCIE_RX_THROUGHPUT: PCIe 接收带宽
# - DCGM_FI_DEV_GPU_UTILIZATION: GPU 利用率
```

## 故障排查

### 问题 1: nsys 命令失败

```bash
# 错误: "nsys: command not found"
# 解决: 安装 Nsight Systems
sudo apt-get update
sudo apt-get install nsight-systems

# 或从 NVIDIA 官网下载:
# https://developer.nvidia.com/nsight-systems
```

### 问题 2: Profiling 文件太大

```bash
# 减少 profiling 范围
nsys profile \
  --trace=cuda,nvtx \  # 只 trace CUDA 和 NVTX，不 trace OS runtime
  --cuda-memory-usage=false \  # 不追踪显存分配
  --output=profile \
  python3 your_script.py
```

### 问题 3: 服务启动失败

```bash
# 检查端口是否被占用
lsof -i :30000
lsof -i :30001
lsof -i :8000

# 杀掉占用进程
kill -9 <PID>

# 检查日志
tail -50 ./mybench/profiling/test1/prefill_server.log
tail -50 ./mybench/profiling/test1/decode_server.log
```

## 参考资源

- [Nsight Systems 文档](https://docs.nvidia.com/nsight-systems/)
- [DCGM 用户指南](https://docs.nvidia.com/datacenter/dcgm/latest/user-guide/)
- [SGLang Profiling 文档](https://docs.sglang.ai/)
- [CUDA Profiling 最佳实践](https://developer.nvidia.com/blog/how-profile-optimize-applications-nsight-compute/)
