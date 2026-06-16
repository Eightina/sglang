# PD 模式 KV Transfer 瓶颈 Profiling 指南

## 1. 问题背景

FP8 KV cache 测试结果显示，KV transfer 时间几乎没有减少（TTFT 基本不变），这与预期（减半）差距很大。需要通过 profiling 确定瓶颈所在。

## 2. Profiling 目标

需要回答以下问题：
1. **PCIe 带宽利用率**：实际传输带宽 vs 理论带宽（PCIe 3.0 x16 = ~32 GB/s）
2. **传输延迟组成**：握手、数据传输、同步等待各占多少时间
3. **CPU vs GPU 瓶颈**：是 CPU 拷贝开销还是 GPU 计算等待
4. **NIXL 协议开销**：有多少时间花在协议交互而非数据传输

## 3. Profiling 方法

### 3.1 方法一：使用 nsys (NVIDIA Nsight Systems)

**最全面的 profiling 工具**，可以分析 GPU/CPU 活动、CUDA kernel、内存拷贝。

```bash
# 1. 启动 prefill worker 并 profile
nsys profile \
  --output=pd-prefill-profile \
  --trace=cuda,nvtx,osrt \
  --cuda-memory-usage=true \
  --sample=none \
  python3 -m sglang.launch_server --config ./pyscripts/q-prefilltp2.yaml

# 2. 启动 decode worker 并 profile
nsys profile \
  --output=pd-decode-profile \
  --trace=cuda,nvtx,osrt \
  --cuda-memory-usage=true \
  --sample=none \
  python3 -m sglang.launch_server --config ./pyscripts/q-decodetp2.yaml

# 3. 运行 benchmark（只需一个实验即可）
python3 -m sglang.bench_serving \
  --host 127.0.0.1 --port 8000 \
  --dataset-name random-ids \
  --random-input-len 4096 \
  --random-output-len 512 \
  --num-prompts 10 \
  --max-concurrency 1 \
  --request-rate inf

# 4. 查看结果
nsys stats pd-prefill-profile.nsys-rep
nsys stats pd-decode-profile.nsys-rep

# 5. 在 GUI 中打开（推荐）
nsys-ui pd-prefill-profile.nsys-rep
```

**在 nsys UI 中关注**：
- **CUDA Memcpy**：查看 GPU→CPU 和 CPU→GPU 的内存拷贝时间和带宽
- **NVTX Markers**：SGLang 代码中的计时标记（如 `prefill`, `kv_transfer`, `decode`）
- **OS Runtime**：CPU 端的系统调用和线程活动
- **PCIe 传输**：查看实际的 PCIe 带宽利用率

### 3.2 方法二：在代码中添加计时点

**最精确的方法**，可以测量每个阶段的耗时。

```python
# 在 python/sglang/srt/disaggregation/prefill.py 中添加计时

import time

def send_kv_chunk(self, req, last_chunk=False, end_idx=None):
    start_time = time.perf_counter()
    
    # 准备数据
    prep_start = start_time
    page_size = self.token_to_kv_pool_allocator.page_size
    # ... 原有代码 ...
    prep_end = time.perf_counter()
    
    # 实际传输
    transfer_start = prep_end
    req.disagg_kv_sender.send(page_indices, state_indices)
    transfer_end = time.perf_counter()
    
    # 记录时间
    total_time = (transfer_end - start_time) * 1000
    prep_time = (prep_end - prep_start) * 1000
    transfer_time = (transfer_end - transfer_start) * 1000
    
    print(f"[KV Transfer] req={req.rid}, chunk_size={len(page_indices)} pages, "
          f"prep={prep_time:.2f}ms, transfer={transfer_time:.2f}ms, total={total_time:.2f}ms")
    
    req.start_send_idx = end_idx
```

**在 decode 端也添加计时**：

```python
# 在 python/sglang/srt/disaggregation/decode.py 中

def pop_transferred(self, rids_to_check=None):
    start_time = time.perf_counter()
    
    # 轮询检查传输状态
    polls = self._poll_with_metadata_gate()
    poll_time = (time.perf_counter() - start_time) * 1000
    
    # 提交传输结果
    commit_start = time.perf_counter()
    for i, (decode_req, poll) in enumerate(zip(self.queue, polls)):
        if poll == KVPoll.Success:
            self._commit_transfer_to_req(decode_req)
    commit_time = (time.perf_counter() - commit_start) * 1000
    
    print(f"[Decode Transfer] poll={poll_time:.2f}ms, commit={commit_time:.2f}ms")
```

### 3.3 方法三：监控 PCIe 带宽

**实时监控 PCIe 带宽利用率**。

```bash
# 方法 1: 使用 nvidia-smi（每秒更新）
watch -n 1 nvidia-smi dmon -s p

# 方法 2: 使用 nv-bandwidth-test（需要安装）
# https://developer.nvidia.com/nv-bandwidth-test
nv-bandwidth-test --testcase 5  # PCIe bandwidth test

# 方法 3: 使用 nsys 的 PCIe 追踪
nsys profile --trace=pciex \
  python3 -m sglang.launch_server --config ./pyscripts/q-prefilltp2.yaml
```

**计算带宽利用率**：

```python
# 在代码中计算实际带宽
def calculate_bandwidth(kv_cache_size_bytes, transfer_time_ms):
    """
    kv_cache_size_bytes: KV cache 大小（字节）
    transfer_time_ms: 传输时间（毫秒）
    """
    bandwidth_gbps = (kv_cache_size_bytes / 1e9) / (transfer_time_ms / 1000)
    pcie_bandwidth_gbps = 32  # PCIe 3.0 x16 理论带宽
    utilization = bandwidth_gbps / pcie_bandwidth_gbps * 100
    
    print(f"实际带宽: {bandwidth_gbps:.2f} GB/s")
    print(f"PCIe 带宽利用率: {utilization:.1f}%")
    return bandwidth_gbps, utilization

# 示例：input=4096, 27B 模型, FP16 KV
# KV cache 大小 ≈ 4096 tokens × 64 layers × 2 (K+V) × 4096 hidden_dim × 2 bytes = ~8 GB
# 如果传输时间 = 10s，则带宽 = 0.8 GB/s，利用率 = 2.5%
```

### 3.4 方法四：分析 NIXL 传输日志

**查看 NIXL 的详细传输日志**。

```bash
# 启用 NIXL 详细日志
export NIXL_LOG_LEVEL=DEBUG
export NIXL_LOG_FILE=/tmp/nixl-transfer.log

# 启动服务
python3 -m sglang.launch_server --config ./pyscripts/q-prefilltp2.yaml

# 运行 benchmark
python3 -m sglang.bench_serving ...

# 查看日志
tail -f /tmp/nixl-transfer.log
```

**在日志中关注**：
- Transfer request/response 的时间戳
- 每次传输的数据量
- 握手和同步的延迟
- 错误和重试

### 3.5 方法五：使用 SGLang 内置的 metrics

**SGLang 已经有一些内置的 metrics**。

```bash
# 启用 metrics
python3 -m sglang.launch_server \
  --config ./pyscripts/q-prefilltp2.yaml \
  --enable-metrics

# 访问 metrics endpoint
curl http://localhost:30000/metrics
```

**关注的 metrics**：
- `sglang:prefill_latency_ms`: Prefill 延迟
- `sglang:decode_latency_ms`: Decode 延迟
- `sglang:queue_time_ms`: 队列等待时间

**添加自定义 metrics**（需要在代码中）：

```python
# 在 scheduler.py 中添加
from prometheus_client import Histogram

KV_TRANSFER_TIME = Histogram(
    'sglang_kv_transfer_time_ms',
    'Time spent on KV cache transfer',
    buckets=[10, 50, 100, 500, 1000, 5000, 10000]
)

# 在 send_kv_chunk 中记录
KV_TRANSFER_TIME.observe(transfer_time_ms)
```

## 4. Profiling 实验设计

### 4.1 基线测试（当前状态）

```bash
# 1. 启动服务（不 profile）
python3 -m sglang.launch_server --config ./pyscripts/q-prefilltp2.yaml
python3 -m sglang.launch_server --config ./pyscripts/q-decodetp2.yaml

# 2. 运行单个请求，观察日志
python3 -m sglang.bench_serving \
  --host 127.0.0.1 --port 8000 \
  --dataset-name random-ids \
  --random-input-len 4096 \
  --random-output-len 512 \
  --num-prompts 1 \
  --max-concurrency 1

# 3. 记录以下指标：
# - TTFT (from bench_serving output)
# - 服务端日志中的 prefill_time, kv_transfer_time, decode_time
```

### 4.2 对比测试：FP16 vs FP8

```bash
# FP16 KV cache（基线）
python3 -m sglang.launch_server --config ./pyscripts/q-prefilltp2.yaml
# 运行 benchmark，记录 TTFT

# FP8 KV cache
python3 -m sglang.launch_server \
  --config ./pyscripts/q-prefilltp2.yaml \
  --kv-cache-dtype fp8_e4m3
# 运行 benchmark，记录 TTFT

# 对比：如果 TTFT 几乎不变，说明瓶颈不在带宽
```

### 4.3 对比测试：不同 input length

```bash
# 测试不同 input length，观察 TTFT 与 input length 的关系
for input_len in 512 1024 2048 4096; do
    python3 -m sglang.bench_serving \
      --host 127.0.0.1 --port 8000 \
      --dataset-name random-ids \
      --random-input-len $input_len \
      --random-output-len 512 \
      --num-prompts 10 \
      --max-concurrency 1 \
      --output-file results_i${input_len}.jsonl
done

# 分析：如果 TTFT 与 input length 线性相关，说明瓶颈在传输
# 如果 TTFT 与 input length 无关，说明瓶颈在协议开销
```

## 5. 预期结果和诊断

### 5.1 如果 PCIe 带宽利用率很低（<10%）

**诊断**：瓶颈在协议开销或同步等待，而非数据传输。

**可能原因**：
1. NIXL 的握手和元数据交换占用大量时间
2. Decode 端轮询间隔过长（`--disaggregation-decode-polling-interval`）
3. Prefill 端在等待 decode 端的确认
4. CPU 端的序列化/反序列化开销

**解决方案**：
- 减少 polling interval：`--disaggregation-decode-polling-interval 1`
- 启用 overlap 模式（如果支持）
- 优化 NIXL 配置（batch size, buffer size）

### 5.2 如果 PCIe 带宽利用率很高（>80%）

**诊断**：瓶颈在 PCIe 带宽，FP8 KV cache 应该有效。

**可能原因**：
- FP8 没有真正生效（检查配置）
- FP8 的量化/反量化开销抵消了传输节省
- 传输的数据量没有减半（检查实际传输的 bytes）

**解决方案**：
- 验证 FP8 KV cache 确实生效（检查日志）
- 使用更激进的量化（INT4）
- 考虑 NVLink 硬件升级

### 5.3 如果 CPU 占用率很高

**诊断**：瓶颈在 CPU 端的拷贝或序列化。

**可能原因**：
- GPU→CPU→GPU 的多次拷贝
- Python 端的序列化开销
- 线程竞争和锁

**解决方案**：
- 启用 GPUDirect RDMA（绕过 CPU）
- 使用 zero-copy 传输
- 优化 CPU 端的并行度

## 6. 快速诊断脚本

```bash
#!/bin/bash
# quick-diagnose.sh - 快速诊断 PD 模式瓶颈

echo "=== PD Mode Bottleneck Diagnosis ==="

# 1. 检查 PCIe 带宽
echo "1. Checking PCIe bandwidth..."
nvidia-smi dmon -s p -d 1 -c 5 > pcie_stats.txt
cat pcie_stats.txt

# 2. 运行单个请求并记录时间
echo "2. Running single request test..."
python3 -m sglang.bench_serving \
  --host 127.0.0.1 --port 8000 \
  --dataset-name random-ids \
  --random-input-len 4096 \
  --random-output-len 512 \
  --num-prompts 1 \
  --max-concurrency 1 \
  --output-file single_req.jsonl 2>&1 | tee bench_output.txt

# 3. 提取关键指标
echo "3. Extracting metrics..."
grep -E "TTFT|ITL|Output token throughput" bench_output.txt

# 4. 计算理论传输时间
echo "4. Theoretical transfer time calculation..."
python3 << 'EOF'
# 假设 27B 模型, input=4096, FP16 KV
tokens = 4096
layers = 64
hidden_dim = 4096
kv_pairs = 2  # K and V
bytes_per_element = 2  # FP16

kv_size_bytes = tokens * layers * hidden_dim * kv_pairs * bytes_per_element
kv_size_gb = kv_size_bytes / 1e9

pcie_bandwidth_gbps = 32  # PCIe 3.0 x16
theoretical_time_s = kv_size_gb / pcie_bandwidth_gbps

print(f"KV cache size: {kv_size_gb:.2f} GB")
print(f"Theoretical transfer time (PCIe 3.0 x16): {theoretical_time_s*1000:.2f} ms")
print(f"If actual TTFT >> {theoretical_time_s*1000:.0f} ms, bottleneck is NOT bandwidth")
EOF

echo "=== Diagnosis complete ==="
```

## 7. 下一步行动

1. **运行 nsys profile**：获取最全面的性能数据
2. **添加代码计时点**：精确测量每个阶段的耗时
3. **监控 PCIe 带宽**：确认带宽利用率
4. **对比 FP16 vs FP8**：验证 FP8 是否真的生效
5. **根据结果调整优化方向**

完成 profiling 后，根据结果更新优化策略。
