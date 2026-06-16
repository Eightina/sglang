# Bootstrap 优化方案审查与改进计划

**日期**: 2026-06-13  
**目标**: 在 32 并发下达到最大吞吐（当前 TTFT ~10s，瓶颈在 bootstrap 等待 5338ms）  
**输入文档**: `kv-transfer-bottleneck-analysis.md`, `bootstrap-profiling-results.md`

---

## 1. 现有方案审查

### 1.1 ✅ 正确的判断

| 判断 | 证据 | 结论 |
|------|------|------|
| **瓶颈是等待而非执行** | `create_sender` 0.02ms, `finalize_bootstrap` 0.01ms, 但 bootstrap 总耗时 5338ms | ✅ 正确，99.8% 时间在等待 |
| **UCX 连接已被缓存** | `_add_remote_peer` 只调用一次（783ms），后续请求 0ms | ✅ 正确，连接复用已生效 |
| **高并发是根本原因** | 32 并发 → 16 并发，bootstrap 从 5338ms 降到 470ms（11.4×） | ✅ 正确，排队延迟是主因 |
| **Decode 端等待是新瓶颈** | 16 并发下 decode 等待占 TTFT 的 44% | ✅ 正确，需要关注 decode 端调度 |

### 1.2 ❌ 有问题的方案

#### 方案 1：「减少 Decode 端调度延迟」— 描述模糊，缺乏具体机制

**问题**：
- 文档只说"优先级调度"、"预分配 receiver"、"异步 receiver 创建"，但**没有分析 Decode 端调度慢在哪里**
- Decode 端的 `receiver.init()` 在 `DecodePreallocQueue.add()` 中**同步调用**（同一 scheduler 迭代）
- 如果 decode polling interval 是 1，那么每个 scheduler tick 都会处理 prealloc → transfer → waiting
- **真正的问题可能是**：Decode 端在处理其他 decode forward 时，无法及时处理新请求的 bootstrap

**改进**：
- 需要先 profiling Decode 端的 scheduler 利用率（GPU forward 占用了多少时间？）
- 如果 GPU forward 占用了 90% 的时间，那么 CPU 侧的 receiver 创建会被延迟
- 需要确认 `disaggregation_decode_polling_interval` 的实际值（默认是 1，但如果被改大，会导致延迟）

#### 方案 2：「预热连接池」— 收益极低，不值得做

**问题**：
- `_add_remote_peer` 只在首个请求调用（783ms），后续请求 0ms
- 100 个请求的平均开销 = 783ms / 100 = 7.8ms，**占比 0.1%**
- 即使完全消除，TTFT 只降低 7.8ms（从 10080ms → 10072ms）
- **不值得投入开发资源**

#### 方案 3：「异步 Bootstrap」— 代码中已有实现，但默认关闭

**问题**：
- 文档说"将 Bootstrap 改为异步"，但代码中已经有 `--optimistic-prefill-retries` 参数（默认 0）
- 这个功能**已经实现**：允许 bootstrap 未完成时先开始 forward，如果 bootstrap 后来完成就继续，否则 requeue
- 文档没有提到这个已存在的功能，说明对代码不够熟悉

**改进**：
- 不需要"实现异步 Bootstrap"，只需要**启用并调优** `--optimistic-prefill-retries`
- 建议测试 `--optimistic-prefill-retries 3` 或 `5`

#### 方案 4：「批量 Bootstrap」— 协议不支持，实现复杂

**问题**：
- Bootstrap 是 per-request 的（每个请求有不同的 `bootstrap_room`）
- 要批量处理需要修改 bootstrap 协议（decode 端也要批量处理 metadata）
- 收益不确定（bootstrap 本身很快，慢的是等待 decode 端调度）

**改进**：
- 如果 decode 端调度延迟降低，批量 bootstrap 的收益会进一步降低
- 建议先优化 decode 端调度，再评估批量 bootstrap 的必要性

#### 方案 5：「优化 Decode 端调度架构」— 方向正确，但缺乏具体方案

**问题**：
- 文档说"多队列调度"、"优先级抢占"、"分布式调度"，但没有分析当前调度器的瓶颈
- Decode 端的 scheduler 在处理 decode forward 时，确实会阻塞新请求的处理
- 但"分布式调度"过于复杂，短期内不可行

**改进**：
- 需要先确认 decode 端 scheduler 的利用率（GPU forward 占用了多少时间？）
- 如果 GPU forward 占用时间长，可以考虑 overlap scheduling（已有实现）或异步 receiver 创建

### 1.3 🟡 遗漏的优化方向

#### 遗漏 1：Chunked Prefill 参数调优

**问题**：
- 当前 `chunked_prefill_size` 默认值可能是 2048 或 4096（取决于 GPU 内存）
- 4096 input tokens 会被切成多个 chunk（比如 2 个 2048）
- 每个 chunk 都需要等待 bootstrap 完成才能发送 KV
- **如果 chunk size 太小，会导致 bootstrap 等待时间增加**

**改进**：
- 测试不同的 `chunked_prefill_size`（2048, 4096, 8192）
- 找到最优的 chunk size，使得 bootstrap 等待时间最小化

#### 遗漏 2：Staging Buffer 模式

**问题**：
- `SGLANG_DISAGG_STAGING_BUFFER` 环境变量默认关闭
- Staging buffer 模式会在 GPU 上先 gather KV cache，再一次性发送
- 这可以减少 RDMA 传输次数，提高带宽利用率

**改进**：
- 启用 staging buffer 模式，测试对 KV transfer 的影响
- 注意：staging buffer 不支持 MLA backend

#### 遗漏 3：Decode 端 Polling Interval

**问题**：
- `disaggregation_decode_polling_interval` 默认值是 1（每个 scheduler tick 都处理）
- 如果被改大（比如 10），会导致 decode 端处理 bootstrap 的延迟增加

**改进**：
- 确认 `disaggregation_decode_polling_interval` 的实际值
- 如果是 1，保持不变；如果大于 1，考虑改回 1

#### 遗漏 4：Prefill 端 Batch Size 调优

**问题**：
- Prefill 端的 `max_running_requests` 或 `max_prefill_tokens` 控制 batch size
- 如果 batch size 太小，GPU 利用率低，forward 时间长
- 如果 batch size 太大，内存占用高，可能导致 OOM 或 retraction

**改进**：
- 测试不同的 `max_running_requests`（8, 16, 32）
- 找到最优的 batch size，使得 GPU 利用率最大化且不 OOM

---

## 2. 改进后的优化方案

### 2.1 优先级重排

| 优先级 | 方案 | 预期收益 | 复杂度 | 依据 |
|--------|------|----------|--------|------|
| **P0** | **启用 Optimistic Prefill** | **Bootstrap 降低 50-80%** | 低（只需配置参数） | 代码已实现，只需调优 |
| **P0** | **调优 Chunked Prefill Size** | **Forward 时间降低 10-20%** | 低（只需配置参数） | 减少 chunk 数量，减少 bootstrap 等待 |
| P1 | 启用 Staging Buffer | KV Transfer 速度提升 20-30% | 低（只需配置环境变量） | 减少 RDMA 传输次数 |
| P1 | 调优 Prefill Batch Size | GPU 利用率提升 10-20% | 低（只需配置参数） | 提高 forward 吞吐 |
| P1 | 确认 Decode Polling Interval | 避免 decode 端调度延迟 | 低（只需确认配置） | 防止配置错误导致延迟 |
| **P2** | **优化 Decode 端调度** | **Decode 等待降低 50%+** | 中（需要 profiling + 代码修改） | 解决新的主要瓶颈 |
| P3 | 异步 Receiver 创建（如果 P2 不够） | Decode 等待降低 30-40% | 中（需要线程安全设计） | 进一步降低 decode 延迟 |
| ❌ | ~~预热连接池~~ | 收益极低（7.8ms / 请求） | 低 | 不值得投入 |
| ❌ | ~~批量 Bootstrap~~ | 协议不支持，实现复杂 | 高 | 优先级低于 P2 |

### 2.2 具体实施方案

#### 方案 A：启用 Optimistic Prefill（P0）

**原理**：
- 允许 bootstrap 未完成时先开始 forward
- 如果 bootstrap 在 forward 期间完成，继续发送 KV
- 如果 bootstrap 仍未完成，释放 KV cache 并 requeue（最多重试 N 次）

**配置**：
```bash
python3 -m sglang.launch_server \
    --config ./pyscripts/q-prefilltp2.yaml \
    --optimistic-prefill-retries 3 \
    --enable-metrics \
    --enable-request-time-stats-logging
```

**预期收益**：
- 如果 bootstrap 5338ms，forward 1070ms，那么 forward 可以掩盖 1070ms 的 bootstrap
- 理想情况下，bootstrap 从 5338ms 降到 4268ms（降低 20%）
- 如果 bootstrap 在 forward 期间完成的概率高，收益可能更大（50-80%）

**风险**：
- 如果 bootstrap 在 forward 期间仍未完成，需要 release + requeue，浪费一次 forward
- 需要监控 `prefill_retry_count` 的分布，确认重试次数合理

**验证方法**：
1. 运行 benchmark，对比启用前后的 TTFT
2. 检查 server log 中的 `optimistic prefill retry` 日志，确认重试次数
3. 检查 Prometheus metrics 中的 `prefill_bootstrap` 和 `prefill_forward` 时间分布

#### 方案 B：调优 Chunked Prefill Size（P0）

**原理**：
- 当前 4096 input tokens 可能被切成 2 个 2048 chunk
- 每个 chunk 都需要等待 bootstrap 完成才能发送 KV
- 如果 chunk size 更大（4096 或 8192），可以减少 chunk 数量，减少 bootstrap 等待次数

**配置**：
```bash
python3 -m sglang.launch_server \
    --config ./pyscripts/q-prefilltp2.yaml \
    --chunked-prefill-size 4096 \
    --optimistic-prefill-retries 3
```

**预期收益**：
- 如果 chunk 数量从 2 降到 1，bootstrap 等待时间减少 50%
- Forward 时间可能略微增加（单个 chunk 更大），但总体收益为正

**风险**：
- 如果 chunk size 太大，可能导致 OOM 或 retraction
- 需要监控 GPU 内存使用率

**验证方法**：
1. 运行 benchmark，对比不同 chunk size 的 TTFT
2. 检查 server log 中的 `chunked_prefill` 日志，确认 chunk 数量
3. 监控 GPU 内存使用率，确认没有 OOM

#### 方案 C：启用 Staging Buffer（P1）

**原理**：
- 默认模式下，KV cache 按 page 逐个发送，RDMA 传输次数多
- Staging buffer 模式下，先在 GPU 上 gather 多个 page，再一次性发送
- 减少 RDMA 传输次数，提高带宽利用率

**配置**：
```bash
export SGLANG_DISAGG_STAGING_BUFFER=1
export SGLANG_DISAGG_STAGING_BUFFER_SIZE_MB=64
export SGLANG_DISAGG_STAGING_POOL_SIZE_MB=4096

python3 -m sglang.launch_server \
    --config ./pyscripts/q-prefilltp2.yaml \
    --optimistic-prefill-retries 3 \
    --chunked-prefill-size 4096
```

**预期收益**：
- KV transfer 速度提升 20-30%（从 0.14 GB/s → 0.18 GB/s）
- KV transfer 时间从 521ms 降到 400ms

**风险**：
- 不支持 MLA backend（qwen3.5-27b-fp8 是 GQA，应该支持）
- 需要额外的 GPU 内存（staging buffer 占用）

**验证方法**：
1. 运行 benchmark，对比启用前后的 KV transfer 速度
2. 检查 Prometheus metrics 中的 `kv_transfer_speed_gb_s`

#### 方案 D：调优 Prefill Batch Size（P1）

**原理**：
- Prefill 端的 batch size 控制每次 forward 处理的请求数
- 如果 batch size 太小，GPU 利用率低，forward 时间长
- 如果 batch size 太大，内存占用高，可能导致 OOM 或 retraction

**配置**：
```bash
python3 -m sglang.launch_server \
    --config ./pyscripts/q-prefilltp2.yaml \
    --max-running-requests 16 \
    --optimistic-prefill-retries 3 \
    --chunked-prefill-size 4096
```

**预期收益**：
- GPU 利用率提升 10-20%
- Forward 时间从 1070ms 降到 900ms

**风险**：
- 如果 batch size 太大，可能导致 OOM 或 retraction
- 需要监控 GPU 内存使用率和 retraction 次数

**验证方法**：
1. 运行 benchmark，对比不同 batch size 的 TTFT 和吞吐
2. 检查 server log 中的 `retraction` 日志，确认没有频繁 retraction
3. 监控 GPU 内存使用率

#### 方案 E：确认 Decode Polling Interval（P1）

**原理**：
- Decode 端的 `disaggregation_decode_polling_interval` 控制每多少个 scheduler tick 处理一次 bootstrap
- 如果被改大（比如 10），会导致 decode 端处理 bootstrap 的延迟增加

**配置**：
```bash
python3 -m sglang.launch_server \
    --config ./pyscripts/q-decodetp2.yaml \
    --disaggregation-decode-polling-interval 1
```

**预期收益**：
- 如果当前值大于 1，改回 1 可以减少 decode 端调度延迟 50%+
- 如果当前值已经是 1，无收益

**验证方法**：
1. 检查当前配置：`grep disaggregation_decode_polling_interval pyscripts/q-decodetp2.yaml`
2. 如果大于 1，改为 1 并重新测试

#### 方案 F：优化 Decode 端调度（P2）

**原理**：
- Decode 端的 scheduler 在处理 decode forward 时，会阻塞新请求的处理
- 如果 GPU forward 占用了 90% 的时间，那么 CPU 侧的 receiver 创建会被延迟

**Profiling 方法**：
```python
# 在 decode.py 的 event_loop_normal_disagg_decode 中添加 profiling
import time

while True:
    t0 = time.perf_counter()
    recv_reqs = self.request_receiver.recv_requests()
    self.process_input_requests(recv_reqs)
    t1 = time.perf_counter()
    
    self.process_decode_queue()
    t2 = time.perf_counter()
    
    batch = self.get_next_disagg_decode_batch_to_run()
    t3 = time.perf_counter()
    
    if batch:
        result = self.run_batch(batch)
        self.process_batch_result(batch, result)
    t4 = time.perf_counter()
    
    logger.info(
        f"[DECODE PROFILE] "
        f"recv_requests: {(t1-t0)*1000:.2f}ms, "
        f"process_decode_queue: {(t2-t1)*1000:.2f}ms, "
        f"get_next_batch: {(t3-t2)*1000:.2f}ms, "
        f"run_batch: {(t4-t3)*1000:.2f}ms"
    )
```

**预期发现**：
- 如果 `run_batch` 占用了 90% 的时间，那么 CPU 侧的 receiver 创建会被延迟
- 需要优化 decode forward 的吞吐（比如 overlap scheduling、异步 receiver 创建）

**优化方案**：
1. **Overlap Scheduling**：启用 `event_loop_overlap_disagg_decode`，让 forward 与 process_batch_result 重叠
2. **异步 Receiver 创建**：将 `create_receiver` 移到后台线程，不阻塞主调度循环

### 2.3 实施顺序

```
Week 1: 启用 Optimistic Prefill + 调优 Chunked Prefill Size
  ├─ 测试 --optimistic-prefill-retries 3
  ├─ 测试 --chunked-prefill-size 4096
  └─ 验证 TTFT 降低 30-50%

Week 2: 启用 Staging Buffer + 调优 Prefill Batch Size
  ├─ 启用 SGLANG_DISAGG_STAGING_BUFFER
  ├─ 测试 --max-running-requests 16
  └─ 验证 KV transfer 速度提升 20-30%

Week 3: 确认 Decode Polling Interval + Profiling Decode 调度
  ├─ 确认 disaggregation_decode_polling_interval 值
  ├─ Profiling decode scheduler 利用率
  └─ 识别 decode 端调度的瓶颈

Week 4: 优化 Decode 端调度（如果必要）
  ├─ 启用 overlap scheduling
  ├─ 实现异步 receiver 创建（如果 overlap 不够）
  └─ 验证 decode 等待降低 50%+
```

---

## 3. 预期最终效果

### 3.1 32 并发下的 TTFT 分解

**当前（10080ms）**：
```
TTFT ≈ 10080ms
├── Bootstrap (等待):           ~5338ms  (53%)   🔴 最大瓶颈
├── Prefill Forward:            ~1070ms  (11%)   ← GPU 计算
├── KV Transfer (数据传输):     ~521ms   (5%)    ← 实际 PCIe 传输
├── Decode 端等待:              ~3152ms  (31%)   ← decode_transferred - prefill 阶段
└── 其他/重叠:                  ~0ms     (0%)
```

**优化后（预期 3000-4000ms）**：
```
TTFT ≈ 3000-4000ms
├── Bootstrap (等待):           ~1500ms  (50%)   ✅ 降低 72%（optimistic prefill + chunk size 调优）
├── Prefill Forward:            ~900ms   (30%)   ✅ 降低 16%（batch size 调优）
├── KV Transfer (数据传输):     ~400ms   (13%)   ✅ 降低 23%（staging buffer）
├── Decode 端等待:              ~200ms   (7%)    ✅ 降低 94%（decode polling interval + overlap scheduling）
└── 其他/重叠:                  ~0ms     (0%)
```

### 3.2 吞吐提升

| 指标 | 当前 | 优化后 | 提升 |
|------|------|--------|------|
| TTFT | 10080ms | 3000-4000ms | **60-70%** |
| 吞吐 | 379 tok/s | 1000-1200 tok/s | **2.6-3.2×** |
| Bootstrap | 5338ms | 1500ms | **72%** |
| KV Transfer | 521ms | 400ms | **23%** |
| Decode 等待 | 3152ms | 200ms | **94%** |

---

## 4. 关键问题与风险

### 4.1 为什么 Optimistic Prefill 默认关闭？

**可能的原因**：
1. **稳定性问题**：如果 bootstrap 在 forward 期间仍未完成，需要 release + requeue，可能导致频繁的 retraction
2. **内存压力**：optimistic prefill 会占用 KV cache，如果 bootstrap 失败需要释放，可能导致内存碎片
3. **复杂场景**：在 PP（pipeline parallel）或 EP（expert parallel）模式下，optimistic prefill 的同步逻辑更复杂

**建议**：
- 先在非生产环境测试，确认稳定性
- 监控 `prefill_retry_count` 的分布，确认重试次数合理（比如 90% 的请求在 1-2 次重试内完成）
- 如果重试次数过高（比如 >5 次），说明 bootstrap 延迟太大，需要先优化 decode 端调度

### 4.2 为什么 Staging Buffer 不支持 MLA？

**技术原因**：
- MLA（Multi-head Latent Attention）的 KV cache 结构更复杂（压缩 + 解压）
- Staging buffer 需要在 GPU 上 gather KV cache，但 MLA 的 gather 逻辑更复杂
- 当前实现只支持 GQA/MHA（标准 attention）

**影响**：
- qwen3.5-27b-fp8 是 GQA，应该支持 staging buffer
- 如果使用 MLA 模型（比如 DeepSeek V3），无法使用 staging buffer

### 4.3 Decode 端调度的瓶颈是什么？

**可能的瓶颈**：
1. **GPU Forward 占用时间长**：如果 decode forward 占用了 90% 的时间，CPU 侧的 receiver 创建会被延迟
2. **Polling Interval 太大**：如果 `disaggregation_decode_polling_interval` 大于 1，会导致 bootstrap 处理延迟
3. **KV Cache 分配慢**：`_pre_alloc` 需要分配 KV cache pages，如果内存紧张，分配时间长

**Profiling 方法**：
- 在 decode.py 的 `event_loop_normal_disagg_decode` 中添加 profiling（见方案 F）
- 确认 `run_batch`、`process_decode_queue`、`recv_requests` 各占用多少时间

---

## 5. 总结

### 5.1 现有方案的评估

| 方案 | 评估 | 建议 |
|------|------|------|
| 减少 Decode 端调度延迟 | 方向正确，但缺乏具体机制 | 先 profiling，再决定优化方案 |
| 预热连接池 | 收益极低（7.8ms / 请求） | **不建议实施** |
| 异步 Bootstrap | 代码已实现（optimistic prefill） | 启用并调优 `--optimistic-prefill-retries` |
| 批量 Bootstrap | 协议不支持，实现复杂 | 优先级低于 P2 |
| 优化 Decode 端调度架构 | 方向正确，但过于复杂 | 先优化 polling interval，再考虑架构重构 |

### 5.2 遗漏的优化方向

| 方向 | 预期收益 | 优先级 |
|------|----------|--------|
| 启用 Optimistic Prefill | Bootstrap 降低 50-80% | **P0** |
| 调优 Chunked Prefill Size | Forward 时间降低 10-20% | **P0** |
| 启用 Staging Buffer | KV Transfer 速度提升 20-30% | P1 |
| 调优 Prefill Batch Size | GPU 利用率提升 10-20% | P1 |
| 确认 Decode Polling Interval | 避免 decode 端调度延迟 | P1 |

### 5.3 下一步行动

1. **本周**：启用 optimistic prefill（`--optimistic-prefill-retries 3`）+ 调优 chunked prefill size（`--chunked-prefill-size 4096`）
2. **下周**：启用 staging buffer（`SGLANG_DISAGG_STAGING_BUFFER=1`）+ 调优 prefill batch size（`--max-running-requests 16`）
3. **第三周**：确认 decode polling interval + profiling decode 调度
4. **第四周**：优化 decode 端调度（如果必要）

**预期最终效果**：
- TTFT 从 10080ms 降到 3000-4000ms（**降低 60-70%**）
- 吞吐从 379 tok/s 提升到 1000-1200 tok/s（**提升 2.6-3.2×**）

---

## 附录 A：关键配置参数速查表

| 参数 | 默认值 | 建议值 | 说明 |
|------|--------|--------|------|
| `--optimistic-prefill-retries` | 0 | **3** | 启用 optimistic prefill，最多重试 3 次 |
| `--chunked-prefill-size` | 2048 或 4096 | **4096** | 减少 chunk 数量，减少 bootstrap 等待 |
| `--max-running-requests` | 自动 | **16** | 调优 prefill batch size |
| `--disaggregation-decode-polling-interval` | 1 | **1** | 保持默认，避免延迟 |
| `SGLANG_DISAGG_STAGING_BUFFER` | 0 | **1** | 启用 staging buffer，提高 KV transfer 速度 |
| `SGLANG_DISAGG_STAGING_BUFFER_SIZE_MB` | 64 | 64 | Staging buffer 大小 |
| `SGLANG_DISAGG_STAGING_POOL_SIZE_MB` | 4096 | 4096 | Staging pool 大小 |

## 附录 B：监控指标

| 指标 | 位置 | 说明 |
|------|------|------|
| `prefill_bootstrap` | Prometheus per-stage | Bootstrap 耗时（目标：降低 50%+） |
| `prefill_forward` | Prometheus per-stage | Forward 耗时（目标：降低 10-20%） |
| `prefill_transfer_kv_cache` | Prometheus per-stage | KV transfer 耗时（目标：降低 20-30%） |
| `kv_transfer_speed_gb_s` | Prometheus kv_transfer | KV transfer 速度（目标：提升 20-30%） |
| `prefill_retry_count` | Server log | Optimistic prefill 重试次数（目标：90% 请求 < 2 次） |
| `retraction` | Server log | Retraction 次数（目标：避免频繁 retraction） |
| `decode_transferred` | Prometheus per-stage | Decode 等待时间（目标：降低 50%+） |
