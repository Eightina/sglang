# Decode Scheduler Profiling 结果分析

**日期**: 2026-06-14  
**目标**: 识别 Decode 端调度器的性能瓶颈，对比 16 vs 32 并发，分析 `pop_bootstrapped_poll` 耗时增加的根本原因  
**数据来源**: 
- 16 并发: `mybench/kv-transfer-measurement/20260614_133930/decode_server.log`
- 32 并发: `mybench/kv-transfer-measurement/20260614_134131/decode_server.log`

---

## 1. 核心发现

### 1.1 意外结果：process_batch_result 是真正的瓶颈

根据 `8-bootstrap-opt-plan.md` 的方案 F，我们预期会发现 `run_batch`（GPU forward）占用 90% 时间，但实际结果完全不同：

| 阶段 | 16 并发 P99 | 32 并发 P99 | 占比 (32 conc) |
|------|-------------|-------------|----------------|
| **process_batch_result** | **29.27ms** | **30.38ms** | **93.6%** 🔴 |
| run_batch | 0.93ms | 0.97ms | 3.0% |
| process_decode_queue | 0.48ms | 0.53ms | 1.6% |
| recv | 0.31ms | 0.33ms | 1.0% |
| get_next_batch | 0.17ms | 0.18ms | 0.6% |
| process_input | 0.00ms | 0.00ms | 0.0% |
| war_barrier | 0.03ms | 0.03ms | 0.1% |
| **total** | **31.05ms** | **32.44ms** | **100%** |

### 1.2 关键洞察

1. **process_batch_result 占据绝对主导**：
   - 16 并发：94.3% 的 P99 延迟
   - 32 并发：93.6% 的 P99 延迟
   - 这是 GPU forward 完成后的后处理阶段

2. **16 vs 32 并发差异极小**：
   - P99 延迟：31.05ms vs 32.44ms（仅差 4.5%）
   - 说明瓶颈不是并发压力导致的，而是固有的架构问题

3. **run_batch 不是瓶颈**：
   - GPU forward 只占 3% 的 P99 延迟
   - 与预期完全相反

---

## 2. 详细对比数据

### 2.1 各阶段完整统计（16 并发）

| Stage | Min | Avg | P50 | P99 | Max | Count |
|-------|-----|-----|-----|-----|-----|-------|
| recv | 0.080 | 0.157 | 0.150 | 0.310 | 36.150 | 163,326 |
| process_input | 0.000 | 0.000 | 0.000 | 0.000 | 1.170 | 163,326 |
| process_decode_queue | 0.010 | 0.033 | 0.020 | 0.480 | 34.980 | 163,326 |
| war_barrier | 0.010 | 0.011 | 0.010 | 0.030 | 0.170 | 163,326 |
| get_next_batch | 0.000 | 0.020 | 0.000 | 0.170 | 32.550 | 163,326 |
| run_batch | 0.000 | 0.018 | 0.000 | 0.930 | 150.680 | 163,326 |
| process_batch_result | 0.000 | 0.518 | 0.020 | 29.270 | 32.130 | 163,326 |
| launch_batch_sample | 0.000 | 0.000 | 0.000 | 0.000 | 0.030 | 163,326 |
| **total** | **0.160** | **0.762** | **0.220** | **31.050** | **169.070** | **163,326** |

### 2.2 各阶段完整统计（32 并发）

| Stage | Min | Avg | P50 | P99 | Max | Count |
|-------|-----|-----|-----|-----|-----|-------|
| recv | 0.080 | 0.157 | 0.160 | 0.330 | 56.380 | 209,723 |
| process_input | 0.000 | 0.000 | 0.000 | 0.000 | 0.810 | 209,723 |
| process_decode_queue | 0.010 | 0.034 | 0.010 | 0.530 | 35.550 | 209,723 |
| war_barrier | 0.010 | 0.012 | 0.010 | 0.030 | 0.500 | 209,723 |
| get_next_batch | 0.000 | 0.031 | 0.000 | 0.180 | 33.790 | 209,723 |
| run_batch | 0.000 | 0.021 | 0.000 | 0.970 | 148.060 | 209,723 |
| process_batch_result | 0.000 | 0.613 | 0.020 | 30.380 | 33.330 | 209,723 |
| launch_batch_sample | 0.000 | 0.000 | 0.000 | 0.000 | 0.030 | 209,723 |
| **total** | **0.160** | **0.871** | **0.220** | **32.440** | **164.900** | **209,723** |

### 2.3 并发对比（P99 延迟）

| Stage | 16 Conc P99 | 32 Conc P99 | 差异 | 说明 |
|-------|-------------|-------------|------|------|
| recv | 0.31ms | 0.33ms | +6.5% | 几乎无变化 |
| process_input | 0.00ms | 0.00ms | 0% | 无变化 |
| process_decode_queue | 0.48ms | 0.53ms | +10.4% | 轻微增加 |
| war_barrier | 0.03ms | 0.03ms | 0% | 无变化 |
| get_next_batch | 0.17ms | 0.18ms | +5.9% | 几乎无变化 |
| run_batch | 0.93ms | 0.97ms | +4.3% | 几乎无变化 |
| **process_batch_result** | **29.27ms** | **30.38ms** | **+3.8%** | **几乎无变化** |
| **total** | **31.05ms** | **32.44ms** | **+4.5%** | **几乎无变化** |

---

## 3. 瓶颈深度分析

### 3.1 process_batch_result 是什么？

`process_batch_result` 是 GPU forward 完成后的后处理阶段，包括：

1. **Token 采样结果处理**：
   - 从 GPU 读取采样结果
   - 更新请求状态（已生成的 token 数、finish reason 等）

2. **KV Cache 管理**：
   - 更新 radix tree（如果启用）
   - 释放已完成的请求的 KV cache

3. **请求完成通知**：
   - 将生成的 token 发送回 TokenizerManager
   - 触发 streaming response

4. **Batch 清理**：
   - 移除已完成的请求
   - 准备下一个 iteration

### 3.2 为什么 process_batch_result 这么慢？

**可能的原因**：

1. **ZMQ 通信延迟**：
   - Scheduler 需要将所有生成的 token 通过 ZMQ 发送给 TokenizerManager
   - 如果 batch size 很大（32 并发），每次 iteration 要发送 32 个 token
   - TokenizerManager 可能处理不过来，导致 ZMQ 队列堆积

2. **Tokenizer 解码延迟**：
   - TokenizerManager 需要将 token IDs 解码为文本
   - 如果启用了 streaming，每次 iteration 都要解码并发送 HTTP response
   - Qwen3.5-27B 的 vocabulary 很大，解码可能较慢

3. **HTTP Response 延迟**：
   - TokenizerManager 通过 FastAPI 发送 HTTP response
   - 如果客户端处理慢，会导致 backpressure
   - 32 并发意味着 32 个并发的 HTTP 连接

4. **同步等待**：
   - `process_batch_result` 可能需要等待所有 TP rank 完成
   - 如果 TP=2，需要等待 2 个 GPU 都完成 forward

### 3.3 为什么 16 vs 32 并发差异很小？

**关键发现**：P99 延迟几乎相同（31.05ms vs 32.44ms）

**解释**：

1. **瓶颈不在并发压力**：
   - 如果是并发压力导致的瓶颈，32 并发应该比 16 并发慢很多
   - 但实际上差异只有 4.5%

2. **瓶颈在固有的架构限制**：
   - ZMQ 通信、Tokenizer 解码、HTTP response 都是固有开销
   - 这些开销不随并发数线性增长

3. **P99 是尾部延迟**：
   - P99 反映的是最慢的 1% 的请求
   - 这些请求可能遇到了 GC、网络抖动等偶发问题
   - 这些问题不随并发数变化

---

## 4. pop_bootstrapped_poll 耗时增加的根本原因

### 4.1 问题背景

在 `bootstrap-profiling-results.md` 中，我们发现 32 并发下 `pop_bootstrapped_poll` 的 max 延迟从 12.64ms 增加到 92.75ms（**7.3× 增加**）。这是 bootstrap 总耗时 5338ms 的主要原因。

**核心问题**：为什么 32 并发下 `pop_bootstrapped_poll` 耗时增加这么多？Prefill 和 Decode 之间的 all_reduce 握手同步是如何造成这个延迟的？

### 4.2 代码路径分析

#### Prefill 端：`pop_bootstrapped()` → `poll_and_all_reduce_attn_cp_tp_group()`

```python
# python/sglang/srt/disaggregation/prefill.py:319
def pop_bootstrapped(self, ...):
    # Step 1: 本地 poll 每个 sender 的状态（CPU 操作，很快）
    polls = [poller.poll() for poller in pollers]  # 每个 poller 是 NixlKVSender

    # Step 2: 在 Prefill 的 TP ranks 之间做 all_reduce（同步屏障！）
    dist.all_reduce(tensor_to_reduce, op=dist.ReduceOp.MIN, group=attn_tp_cpu_group)

    # Step 3: 在 Prefill 的 CP ranks 之间做 all_reduce（同步屏障！）
    dist.all_reduce(tensor_to_reduce, op=dist.ReduceOp.MIN, group=attn_cp_cpu_group)
```

#### Decode 端：`_pop_bootstrapped()` → `poll_and_all_reduce()`

```python
# python/sglang/srt/disaggregation/decode.py:651
def _pop_bootstrapped(self, ...):
    # Step 1: 本地 poll 每个 receiver 的状态
    polls = [poller.poll() for poller in pollers]  # 每个 poller 是 NixlKVReceiver

    # Step 2: 在 Decode 的 TP ranks 之间做 all_reduce（同步屏障！）
    dist.all_reduce(tensor_to_reduce, op=dist.ReduceOp.MIN, group=gloo_group)
```

#### 关键发现：Prefill 和 Decode 之间**没有**直接的 all_reduce 同步

- Prefill 端的 all_reduce 只在 **Prefill 的 TP ranks**（GPU 0, 1）之间进行
- Decode 端的 all_reduce 只在 **Decode 的 TP ranks**（GPU 2, 3）之间进行
- **Prefill 和 Decode 之间通过 NIXL/UCX 数据通道通信，不通过 all_reduce**

### 4.3 `pop_bootstrapped_poll` 耗时增加的直接原因

`pop_bootstrapped_poll` 的耗时 = `poll_and_all_reduce_attn_cp_tp_group()` 的耗时，包括：

1. **本地 poll 阶段**（`_poll_with_failure_injection`）：
   - 遍历 queue 中所有 sender，调用 `sender.poll()` → `kv_mgr.check_status(bootstrap_room)`
   - 这是纯 CPU 操作，很快（微秒级）

2. **TP all_reduce 阶段**（`dist.all_reduce(group=attn_tp_cpu_group)`）：
   - **这是一个同步屏障**：TP rank 0 和 TP rank 1 必须同时到达这个点
   - 如果 TP rank 1 还在处理上一个 iteration 的 `run_batch` 或 `process_batch_result`，TP rank 0 必须等待
   - 使用 Gloo 后端（CPU all_reduce），通过 PCIe 通信

3. **CP all_reduce 阶段**（`dist.all_reduce(group=attn_cp_cpu_group)`）：
   - 第二次同步屏障（如果 attn_cp_size > 1）

**直接原因**：32 并发下，Prefill 端的某个 TP rank 的 scheduler loop 更慢，导致 all_reduce 屏障等待时间增加。

### 4.4 为什么 32 并发下 Prefill 端 TP rank 更慢？

**因果链**：

```
32 并发 → Prefill scheduler loop 更忙 → 某个 TP rank 的 iteration 耗时更长
→ 该 TP rank 到达 all_reduce 屏障的时间更晚
→ 另一个 TP rank 等待时间更长
→ pop_bootstrapped_poll 耗时增加
```

具体来说：

1. **Prefill scheduler loop 的总耗时增加**：
   - 32 并发下，每次 iteration 要处理更多的请求
   - `run_batch`（GPU forward）处理更大的 batch
   - `process_batch_result` 处理更多的 token

2. **TP rank 之间的负载不均衡**：
   - TP rank 0 和 TP rank 1 的 scheduler loop 可能不完全同步
   - 某个 rank 可能因为 GC、CUDA stream 同步等原因稍慢
   - 在高并发下，这种微小的不均衡被放大

3. **all_reduce 是同步屏障**：
   - `dist.all_reduce` 必须等待所有 rank 到达
   - 最慢的 rank 决定了所有 rank 的等待时间
   - 这就是为什么 P99 没有变化（大多数 poll 很快），但 max 增加了 7.3 倍（少数 poll 遇到了极端等待）

### 4.5 与 Decode 侧的间接关系

虽然 Prefill 和 Decode 之间没有直接的 all_reduce 同步，但 Decode 侧的性能问题会**间接影响** `pop_bootstrapped_poll`：

#### 间接影响路径 1：Bootstrap 状态转换慢

```
Decode 端 process_batch_result 慢（30ms P99）
→ Decode 端 scheduler loop 处理新 bootstrap 请求慢
→ Decode 端 receiver.init() 延迟
→ Decode 端 NIXL receiver 状态转换慢（Bootstrapping → WaitingForInput）
→ Prefill 端 sender.poll() 返回 Bootstrapping 的次数更多
→ Prefill 端需要更多次 poll 才能完成 bootstrap
→ 每次 poll 都有 all_reduce 开销
→ 累积延迟增加
```

#### 间接影响路径 2：Prefill 端 scheduler loop 竞争

```
Decode 端处理慢 → KV transfer 等待时间更长
→ Prefill 端 inflight queue 堆积
→ Prefill 端 process_disagg_prefill_inflight_queue 更慢
→ Prefill 端 scheduler loop 总耗时增加
→ TP rank 到达 all_reduce 屏障的时间差更大
→ pop_bootstrapped_poll 耗时增加
```

#### 间接影响路径 3：HTTP bootstrap info 获取慢

```
Decode 端 scheduler loop 慢 → Decode 端处理 HTTP bootstrap info 请求慢
→ Prefill 端 _get_bootstrap_info_from_server 延迟增加
→ Prefill 端 create_sender 延迟增加
→ Prefill 端 bootstrap queue 堆积
→ pop_bootstrapped 的 queue_size 更大
→ poll 更多 sender → all_reduce tensor 更大 → all_reduce 更慢
```

### 4.6 为什么 P99 不变但 max 增加了 7.3 倍？

| 指标 | 16 并发 | 32 并发 | 变化 |
|------|---------|---------|------|
| P50 | 0.36ms | 0.36ms | 0% |
| P99 | 0.74ms | 0.74ms | 0% |
| **max** | **12.64ms** | **92.75ms** | **+7.3×** |

**解释**：

1. **大多数 poll 操作不受影响**：
   - P50 和 P99 完全相同
   - 89.2% 的 poll 在 0.5ms 内完成
   - 说明大多数时候 TP rank 之间的同步很快

2. **少数 poll 操作遇到极端等待**：
   - max 从 12.64ms 增加到 92.75ms
   - 这些极端情况发生在 TP rank 之间的负载不均衡最严重的时候
   - 32 并发下，这种极端情况出现的概率更高

3. **累积效应**：
   - 单个 poll 的 max 延迟 = 92.75ms
   - 但 bootstrap 需要 ~12000 次 poll
   - 即使只有 0.2% 的 poll 超过 1ms，累积起来也是巨大的延迟

---

## 5. 优化建议（更新）

### 5.1 原方案 F 的假设

原方案 F 假设：

> "如果 `run_batch` 占用了 90% 的时间，那么 CPU 侧的 receiver 创建会被延迟"

**实际情况**：

- `run_batch` 只占 3%
- `process_batch_result` 占 93.6%
- 假设不成立

### 5.2 新的优化方向

#### 方向 1：优化 Prefill 端 TP all_reduce 同步（最高优先级）

**目标**：减少 `pop_bootstrapped_poll` 的 max 延迟

**方法**：

1. **减少 poll 频率**：
   - 当前每次 scheduler iteration 都调用 `pop_bootstrapped()`
   - 可以每 N 次 iteration 才调用一次
   - 减少 all_reduce 的次数

2. **异步 all_reduce**：
   - 将 all_reduce 移到后台线程
   - 不阻塞 scheduler loop

3. **减少 queue_size**：
   - 更大的 queue 意味着更多的 sender 需要 poll
   - all_reduce tensor 更大，all_reduce 更慢
   - 可以通过更快的 bootstrap 完成来减少 queue_size

#### 方向 2：优化 process_batch_result（次优先级）

**目标**：将 process_batch_result 从 30ms 降到 5ms

**方法**：

1. **异步 Tokenizer 解码**：
   - 将 token 解码移到后台线程
   - Scheduler 不需要等待解码完成

2. **批量发送 token**：
   - 不要每次 iteration 都发送 token
   - 累积多个 token 后一次性发送

3. **减少 ZMQ 通信次数**：
   - 使用 batch notify（已有 `batch_notify_size=16`）
   - 每 16 个 token 才发送一次通知

4. **启用 overlap scheduling**：
   - 让 `process_batch_result` 与下一个 `run_batch` 重叠
   - 需要修改 scheduler 架构

#### 方向 3：优化 Bootstrap 等待（次优先级）

**目标**：将 bootstrap 从 5338ms 降到 1000ms

**方法**：

1. **启用 Optimistic Prefill**：
   - `--optimistic-prefill-retries 3`
   - 允许 bootstrap 未完成时先开始 forward

2. **调优 Chunked Prefill Size**：
   - `--chunked-prefill-size 4096`
   - 减少 chunk 数量，减少 bootstrap 等待次数

#### 方向 4：优化 Decode 端 GPU 利用率（低优先级）

**目标**：提高 Decode 端的 GPU 利用率

**方法**：

1. **增加 batch size**：
   - `--max-running-requests 32`
   - 让 Decode 端一次处理更多请求

2. **启用 continuous batching**：
   - 让 Decode 端动态调整 batch size
   - 避免 GPU 空闲

---

## 6. 下一步行动

### 6.1 立即行动（本周）

1. **深入分析 process_batch_result**：
   - 添加更细粒度的 profiling
   - 识别 process_batch_result 内部的瓶颈（ZMQ？Tokenizer？HTTP？）

2. **测试异步 Tokenizer 解码**：
   - 修改 TokenizerManager，将解码移到后台线程
   - 测量对 process_batch_result 的影响

### 6.2 短期行动（下周）

1. **启用 Optimistic Prefill**：
   - 测试 `--optimistic-prefill-retries 3`
   - 测量对 bootstrap 延迟的影响

2. **调优 Chunked Prefill Size**：
   - 测试 `--chunked-prefill-size 4096`
   - 测量对 TTFT 的影响

### 6.3 中期行动（第三周）

1. **实现 overlap scheduling**：
   - 让 process_batch_result 与 run_batch 重叠
   - 需要修改 scheduler 架构

2. **优化 ZMQ 通信**：
   - 使用 batch notify
   - 减少通信次数

---

## 7. 总结

### 7.1 核心发现

1. **Decode 端 scheduler loop 的瓶颈是 process_batch_result（93.6%）**，不是 run_batch（3%）
2. **16 vs 32 并发差异极小（4.5%）**，说明瓶颈是固有的架构问题，不是并发压力
3. **原方案 F 的假设不成立**，需要重新设计优化方案
4. **`pop_bootstrapped_poll` 耗时增加的根本原因是 Prefill 端 TP all_reduce 同步屏障**，不是 Decode 端直接造成的
5. **Decode 端通过间接路径影响 Prefill 端**：Decode 慢 → Prefill scheduler loop 更忙 → TP rank 负载不均衡更严重 → all_reduce 等待更久

### 7.2 关键指标

| 指标 | 16 并发 | 32 并发 | 差异 |
|------|---------|---------|------|
| Scheduler loop P99 | 31.05ms | 32.44ms | +4.5% |
| process_batch_result P99 | 29.27ms | 30.38ms | +3.8% |
| run_batch P99 | 0.93ms | 0.97ms | +4.3% |
| Scheduler iterations | 163,326 | 209,723 | +28.4% |
| pop_bootstrapped_poll max | 12.64ms | 92.75ms | +7.3× |

### 7.3 优化优先级

1. **P0**: 优化 Prefill 端 TP all_reduce 同步（减少 poll 频率、异步 all_reduce）
2. **P1**: 优化 process_batch_result（异步 Tokenizer、批量发送、overlap scheduling）
3. **P2**: 优化 Bootstrap 等待（Optimistic Prefill、Chunked Prefill Size）
4. **P3**: 优化 Decode 端 GPU 利用率（增加 batch size、continuous batching）

---

## 附录 A：Profiling 方法

### A.1 添加 Profiling 代码

在 `python/sglang/srt/disaggregation/decode.py` 的 `event_loop_overlap_disagg_decode` 函数中添加：

```python
def event_loop_overlap_disagg_decode(self: Scheduler):
    """An overlap scheduler loop for decode worker in disaggregation mode."""
    import time

    while True:
        t0 = time.perf_counter()

        # Receive requests
        recv_reqs = self.request_receiver.recv_requests()
        t1 = time.perf_counter()

        self.process_input_requests(recv_reqs)
        t2 = time.perf_counter()

        self.process_decode_queue()
        t3 = time.perf_counter()

        if self._engine_paused:
            continue

        # WAR barrier
        self.tp_worker.wait_war_barrier()
        t4 = time.perf_counter()

        # Get the next batch to run
        batch = self.get_next_disagg_decode_batch_to_run()
        t5 = time.perf_counter()

        self.cur_batch = batch

        # Launch the current batch
        if batch:
            result = self.run_batch(batch)
            t6 = time.perf_counter()

            self.process_batch_result(batch, result)
            t7 = time.perf_counter()

            # Launch batch sample
            self.tp_worker.launch_batch_sample()
            t8 = time.perf_counter()

            logger.info(
            f"[DECODE PROFILE OVERLAP] "
            f"recv: {(t1-t0)*1000:.2f}ms, "
            f"process_input: {(t2-t1)*1000:.2f}ms, "
            f"process_decode_queue: {(t3-t2)*1000:.2f}ms, "
            f"war_barrier: {(t4-t3)*1000:.2f}ms, "
            f"get_next_batch: {(t5-t4)*1000:.2f}ms, "
            f"run_batch: {(t6-t5)*1000:.2f}ms, "
            f"process_batch_result: {(t7-t6)*1000:.2f}ms, "
            f"launch_batch_sample: {(t8-t7)*1000:.2f}ms, "
            f"total: {(t8-t0)*1000:.2f}ms"
            )
        else:
            self.on_idle()
            t6 = time.perf_counter()

            logger.info(
            f"[DECODE PROFILE OVERLAP] "
            f"recv: {(t1-t0)*1000:.2f}ms, "
            f"process_input: {(t2-t1)*1000:.2f}ms, "
            f"process_decode_queue: {(t3-t2)*1000:.2f}ms, "
            f"war_barrier: {(t4-t3)*1000:.2f}ms, "
            f"get_next_batch: {(t5-t4)*1000:.2f}ms, "
            f"idle: {(t6-t5)*1000:.2f}ms, "
            f"total: {(t6-t0)*1000:.2f}ms"
            )

        self.last_batch = batch
```

### A.2 分析脚本

使用 `mybench/analyze_decode_profiling.py` 分析 profiling 结果：

```bash
python3 mybench/analyze_decode_profiling.py \
    mybench/kv-transfer-measurement/20260614_133930/decode_server.log \
    mybench/kv-transfer-measurement/20260614_134131/decode_server.log
```

### A.3 关键代码路径

#### Prefill 端 all_reduce 路径

```
pop_bootstrapped()                                    # prefill.py:319
  → poll_and_all_reduce_attn_cp_tp_group()            # utils.py:118
    → poll_and_all_reduce(attn_tp_cpu_group)          # utils.py:96
      → _poll_with_failure_injection(pollers)         # utils.py:62
        → sender.poll() → kv_mgr.check_status()       # nixl/conn.py:1946
      → dist.all_reduce(group=attn_tp_cpu_group)      # 同步屏障！
    → dist.all_reduce(group=attn_cp_cpu_group)        # 同步屏障！
```

#### Decode 端 all_reduce 路径

```
_pop_bootstrapped()                                   # decode.py:640
  → poll_and_all_reduce(gloo_group)                   # utils.py:96
    → _poll_with_failure_injection(pollers)           # utils.py:62
      → receiver.poll() → kv_mgr.check_status()       # nixl/conn.py:2072
    → dist.all_reduce(group=gloo_group)               # 同步屏障！
```

---

**文档版本**: v2.0  
**最后更新**: 2026-06-14
