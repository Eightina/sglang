# TP4 单机 PCIe 环境优化计划

**日期**: 2026-06-16  
**硬件环境**: 单机 4× RTX PRO 4000 Blackwell (PCIe, 无 NVLink)  
**目标**: 在 TP=4 配置下优化推理性能，找到有竞争力的优化点

---

## 1. 背景与问题

### 1.1 当前状态

在 TP=4 PCIe 环境下，SGLang 的 AllReduce 通信存在明显的性能瓶颈：

- **CustomAllReduce V2**: 严格要求 NVLink，PCIe 环境无法使用
- **CustomAllReduce V1**: 仅支持 TP=2 的 PCIe P2P，TP=4 需要 NVLink
- **FlashInfer AllReduce Fusion**: 理论上支持 PCIe，但实际效果存疑
- **NCCL**: 作为 fallback，在小消息（< 256KB）时延迟高（2-5ms）

### 1.2 性能分析

从之前的 profiling 数据：

```
Decode 阶段（最关键）：
- 实测 ITL = 32ms/step
- 纯计算时间 ≈ 8.6ms（27B params × 32 tokens）
- 非计算开销 = 32 - 8.6 = ~23ms (72%)

这 23ms 包括：
- AllReduce (TP=4, 128 次/step) ← 主要部分
- KV cache 读写
- Kernel launch 开销
- Memory-bound 操作（layernorm 等）
```

**关键发现**: 72% 的 decode step 时间不在计算上，AllReduce 通信是其中最大的可优化项。

---

## 2. 优化方向

### 2.1 方向 A：PCIe Shared Memory AllReduce（最有竞争力）

**动机**: NCCL 在小消息时的延迟高，因为它要走 GPU → PCIe switch → GPU 的路径。但如果 4 张 GPU 共享同一个 NUMA node，可以用 CPU shared memory 做中转。

**实现思路**:
```
当前 NCCL 路径:
  GPU 0 → PCIe → PCIe Switch → PCIe → GPU 1/2/3
  延迟: 2-5ms (每次 AllReduce)

Shared Memory 路径:
  GPU 0 → cudaMemcpy(D2H) → CPU shared memory → cudaMemcpy(H2D) → GPU 1/2/3
  延迟: 可能更低（CPU memcpy 对小块数据延迟更低）
```

**实施步骤**:
1. 启动时分配一个 CPU shared memory buffer（通过 `mmap` + `shm_open`）
2. 4 个 TP rank 的进程共享这个 buffer
3. AllReduce 时：每个 rank `cudaMemcpy(D2H)` 到 shared buffer → barrier → `cudaMemcpy(H2D)` 读回
4. 对小 tensor（< 256KB）使用这个路径，大 tensor 走 NCCL

**参考代码**: SGLang 已有的 `CustomAllReduce V1`（`python/sglang/srt/distributed/device_communicators/custom_all_reduce.py`），它用 CUDA IPC 做 P2P，可以改用 shared memory。

**预期收益**: AllReduce 延迟从 2-5ms 降到 0.5-1ms，decode ITL 降低 10-20%。

### 2.2 方向 B：FlashInfer AllReduce Fusion PCIe 适配

**动机**: FlashInfer AllReduce Fusion 理论上支持 PCIe，但实际效果存疑。需要确认：
1. Workspace 是否创建成功
2. Fusion 是否真的被使用
3. 在 PCIe 上是否有实际收益

**诊断步骤**:
```bash
# 1. 检查 workspace 是否创建成功
grep -i "flashinfer\|allreduce.*fusion\|workspace" prefill_server.log

# 期望看到的（成功）：
# "FlashInfer workspace initialized for rank X, backend trtllm"

# 可能看到的（失败）：
# "Failed to initialize FlashInfer workspace: ..."
# 或
# "FlashInfer workspace preflight: cuMemCreate probe failed"

# 2. 检查 fusion 是否真的被使用
grep -i "allreduce_fusion\|needs_allreduce" prefill_server.log

# 3. 对比 NCCL 调用次数
# 如果 fusion 生效，ncclKernel 调用次数应该比没开 fusion 少
```

**可能的优化**:
- 如果 workspace 创建失败 → 分析原因，fix preflight check
- 如果 fusion 没效果 → profile AllReduce 实际耗时，分析为什么
- 实现 PCIe-specific 的 fusion 优化

**预期收益**: 如果 fusion 在 PCIe 上有效，AllReduce 调用次数减少 50%，decode ITL 降低 5-10%。

### 2.3 方向 C：AllReduce-Compute Overlap

**动机**: 当前 AllReduce 是阻塞的，GPU 等 AllReduce 完成后才开始下一个计算。可以把 AllReduce 放到独立 CUDA stream，与下一层计算重叠。

**实现思路**:
```python
# 当前: 串行
all_reduce(layer_N_output)          # 等 3ms
compute_layer_N+1_input_layernorm   # 等 0.5ms
compute_layer_N+1_qkv_proj          # 等 2ms

# 优化: 重叠
all_reduce_async(layer_N_output, stream=comm_stream)   # 后台执行
compute_layer_N+1_input_layernorm(residual)             # 不依赖 AllReduce 结果
wait(comm_stream)                                        # 在这里同步
compute_layer_N+1_qkv_proj(all_reduced_hidden)          # 依赖 AllReduce 结果
```

**修改位置**: `python/sglang/srt/layers/communicator.py` 中的 `prepare_mlp()` 和 `prepare_attn()` 逻辑。

**预期收益**: AllReduce 延迟被部分隐藏，decode ITL 降低 5-15%。

### 2.4 方向 D：NCCL 参数调优

**动机**: NCCL 有多种算法和协议，不同配置在不同硬件上性能差异很大。

**测试矩阵**:
```bash
# 算法选择
export NCCL_ALGO=Ring      # vs Tree, CollNetDirect

# 协议选择
export NCCL_PROTO=Simple    # vs LL, LL128

# 禁用 P2P（强制走 PCIe）
export NCCL_P2P_DISABLE=1

# 禁用 SHM（强制走 PCIe）
export NCCL_SHM_DISABLE=1

# Buffer 大小
export NCCL_BUFFSIZE=4194304  # 4MB, 默认 4MB
```

**预期收益**: 10-20% 的 AllReduce 性能提升。

---

## 3. 实施计划

### Phase 1: Profiling 与诊断（Week 1）

**目标**: 量化 AllReduce 在 decode step 中的实际开销，确认优化方向。

**任务**:
1. **nsys profile decode step**
   ```bash
   nsys profile --trace=cuda,nvtx,osrt \
       python3 -m sglang.launch_server \
       --model qwen/qwen3.5-27b-fp8 --tp 4 \
       --max-running-requests 32
   ```
   - 查看 `ncclKernel_AllReduce` 调用次数、每次耗时、数据量
   - 计算 AllReduce 占 decode ITL 的比例

2. **检查 FlashInfer AllReduce Fusion 状态**
   ```bash
   grep -i "flashinfer\|allreduce.*fusion\|workspace" prefill_server.log
   ```
   - 确认 workspace 是否创建成功
   - 确认 fusion 是否被使用

3. **对比 NCCL 不同配置**
   - 测试 `NCCL_ALGO=Ring` vs `Tree`
   - 测试 `NCCL_PROTO=Simple` vs `LL`
   - 对比 ITL 和吞吐

**产出**: Profiling 报告，明确 AllReduce 占比和当前配置状态。

### Phase 2: 快速优化（Week 2）

**目标**: 基于 profiling 结果，实施低风险的优化。

**任务**:
1. **如果 FlashInfer fusion 失败** → 分析原因，尝试 fix
2. **如果 NCCL 参数调优有效** → 确定最优配置
3. **实现 AllReduce-Compute Overlap**（如果 Phase 1 证明有价值）

**产出**: 优化后的配置或代码，对比 baseline 的性能提升。

### Phase 3: 深度优化（Week 3-4）

**目标**: 实现 PCIe Shared Memory AllReduce（如果 Phase 1-2 证明有价值）。

**任务**:
1. 参考 `CustomAllReduce V1` 代码结构
2. 实现小 tensor 的 shared memory 路径
3. Benchmark 对比 NCCL vs shared memory
4. 集成到 SGLang，提交 PR

**产出**: PCIe Shared Memory AllReduce 实现，性能对比报告。

---

## 4. 预期成果

### 4.1 最差情况

- 一份详细的 AllReduce profiling 报告
- NCCL 参数调优建议
- 明确说明为什么某些优化在 PCIe 上无效

**面试价值**: "我深入分析了 TP AllReduce 在 PCIe 上的开销，发现 X，排除了 Y，最终确定了 Z 是最优方案。"

### 4.2 中等情况

- NCCL 参数调优带来 10-20% 的 AllReduce 性能提升
- FlashInfer fusion 的 PCIe 适配或修复

**面试价值**: "我通过参数调优和配置修复，将 decode ITL 降低了 X%，在 PCIe 环境下达到了接近 NVLink 的性能。"

### 4.3 最好情况

- 实现 PCIe Shared Memory AllReduce
- 提交 PR 到 SGLang 主仓库
- Decode ITL 降低 10-20%

**面试价值**: "我发现 SGLang 在 PCIe 环境下缺少 AllReduce 优化，实现了 PCIe Shared Memory AllReduce，将 decode ITL 降低了 X%，并提交到主仓库。"

---

## 5. 关键文件

### 5.1 AllReduce 相关代码

| 文件 | 职责 |
|------|------|
| `python/sglang/srt/distributed/parallel_state.py` | `GroupCoordinator.all_reduce()` 调度链 |
| `python/sglang/srt/layers/communicator.py` | Transformer block 级别的通信编排 |
| `python/sglang/srt/layers/flashinfer_comm_fusion.py` | FlashInfer AllReduce Fusion 实现 |
| `python/sglang/srt/distributed/device_communicators/custom_all_reduce.py` | CustomAllReduce V1 |
| `python/sglang/srt/distributed/device_communicators/custom_all_reduce_v2.py` | CustomAllReduce V2 |

### 5.2 Model Forward Pass

| 文件 | 职责 |
|------|------|
| `python/sglang/srt/models/qwen3.py` | Qwen3 dense model forward |
| `python/sglang/srt/models/qwen3_5.py` | Qwen3.5 MoE model forward |
| `python/sglang/srt/layers/linear.py` | `RowParallelLinear.forward()` 中的 AllReduce |

### 5.3 环境变量

| 变量 | 用途 |
|------|------|
| `SGLANG_OPT_USE_CUSTOM_ALL_REDUCE_V2` | 启用 CustomAllReduce V2（默认 True） |
| `NCCL_ALGO` | NCCL 算法选择（Ring/Tree/CollNetDirect） |
| `NCCL_PROTO` | NCCL 协议选择（Simple/LL/LL128） |
| `NCCL_P2P_DISABLE` | 禁用 P2P（强制走 PCIe） |
| `NCCL_SHM_DISABLE` | 禁用 SHM（强制走 PCIe） |
| `NCCL_BUFFSIZE` | NCCL buffer 大小 |

---

## 6. 测试命令

### 6.1 Baseline 测试

```bash
# 启动 TP=4 服务
python3 -m sglang.launch_server \
    --model qwen/qwen3.5-27b-fp8 \
    --tp 4 \
    --max-running-requests 32 \
    --enable-metrics \
    --enable-request-time-stats-logging

# 运行 benchmark
python3 -m sglang.bench_serving \
    --host 127.0.0.1 \
    --port 30000 \
    --dataset-name random-ids \
    --random-input-len 4096 \
    --random-output-len 512 \
    --num-prompts 100 \
    --max-concurrency 32
```

### 6.2 Profiling 测试

```bash
# nsys profile
nsys profile --trace=cuda,nvtx,osrt \
    python3 -m sglang.launch_server \
    --model qwen/qwen3.5-27b-fp8 \
    --tp 4 \
    --max-running-requests 32

# 查看 nsys 报告
nsys stats report.nsys-rep
```

### 6.3 NCCL 参数调优测试

```bash
# 测试不同 NCCL 配置
for algo in Ring Tree; do
    for proto in Simple LL; do
        export NCCL_ALGO=$algo
        export NCCL_PROTO=$proto
        echo "Testing NCCL_ALGO=$algo NCCL_PROTO=$proto"
        python3 -m sglang.bench_serving ...
    done
done
```

---

## 7. 总结

### 7.1 核心问题

在 TP=4 PCIe 环境下，SGLang 对 AllReduce 没有任何优化，完全 fallback 到 NCCL。NCCL 在小消息时延迟高（2-5ms），导致 decode ITL 中 72% 的时间不在计算上。

### 7.2 优化路径

| 优先级 | 方向 | 类型 | 预期收益 |
|--------|------|------|----------|
| **P0** | Profiling AllReduce 开销 | 诊断 | 明确优化方向 |
| **P0** | NCCL 参数调优 | 配置 | 10-20% |
| **P1** | FlashInfer fusion 适配 | 配置/代码 | 5-10% |
| **P1** | AllReduce-Compute Overlap | 代码 | 5-15% |
| **P2** | PCIe Shared Memory AllReduce | 代码 | 10-20% |

### 7.3 核心竞争力

这个项目展示了：
1. **深度 profiling 能力**: 用 nsys 定位性能瓶颈
2. **系统级理解**: 理解 TP AllReduce 的通信机制
3. **代码实现能力**: 实现 PCIe-specific 的优化
4. **工程实践**: 提交 PR 到开源项目

**最终目标**: 在 PCIe 环境下，将 decode ITL 降低 10-20%，并提交 PR 到 SGLang 主仓库。
