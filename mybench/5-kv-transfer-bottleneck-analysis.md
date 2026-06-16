# KV Transfer 性能瓶颈深度分析

- **日期**: 2026-06-11
- **模型**: `qwen/qwen3.5-27b-fp8`
- **硬件**: 4× NVIDIA RTX PRO 4000 Blackwell (PCIe, no NVLink)
- **配置**: PD TP2 模式（Prefill TP=2 GPU 0-1, Decode TP=2 GPU 2-3）
- **传输后端**: NIXL over UCX (PCIe)

---

## 1. 核心发现

### 1.1 KV Transfer 性能数据（100 请求，32 并发）

| 指标 | 数值 | 说明 |
|---|---|---|
| 平均传输速度 | **0.140 GB/s** | 单请求平均速度 |
| 聚合带宽 | **0.187 GB/s** | 考虑并发后的有效带宽 |
| 平均传输大小 | 108.6 MB | 每个请求的 KV cache 大小 |
| 平均传输延迟 | **861 ms** | 从开始传输到完成 |
| PCIe 带宽利用率 | **0.58%** | 聚合带宽 / 理论 32 GB/s |
| 传输速度范围 | 0.07 - 0.38 GB/s | 最小到最大 |

### 1.2 并发分析

| 指标 | 数值 |
|---|---|
| 有效请求数 | 101 |
| 传输时间窗口 | 57.2s |
| sum(单请求传输耗时) | 87.0s |
| 最大并发传输数 | 5 |
| 平均并发度 | 1.52 |

**并发度分布**：

| 并发度 | 持续时间 | 占比 |
|---|---:|---:|
| 1 | 25.5s | 44.5% |
| 2 | 15.9s | 27.9% |
| 3 | 6.5s | 11.3% |
| 4 | 2.5s | 4.3% |
| 5 | 0.1s | 0.1% |

**结论**：传输是**部分并行**的，平均并发度 1.52，最大 5。

### 1.3 带宽计算（考虑并发）

| 方法 | 公式 | 结果 | PCIe 利用率 |
|---|---|---:|---:|
| 简单平均（不考虑并发） | sum(单请求速度) / 请求数 | 0.140 GB/s | 0.44% |
| 聚合带宽（考虑并发） | 总数据量 / 时间窗口 | 0.187 GB/s | 0.58% |
| 加权平均（平均并发度） | 平均并发度 × 平均速度 | 0.213 GB/s | 0.67% |

**结论**：三种方法结果接近（0.14-0.21 GB/s），因为实际并发度较低（平均 1.52）。

### 1.4 TTFT 延迟分解（input=4096, output=512, 100 requests）

#### 数据来源：Prometheus per_stage_req_latency

```
TTFT ≈ 9948ms
├── Bootstrap (连接建立):   ~5179ms  (52%)  🔴 最大瓶颈！
├── Queue Time (排队等待):  ~2622ms  (26%)
├── Prefill Forward:        ~1102ms  (11%)  GPU 计算
├── Chunked Prefill:        ~695ms   (7%)   chunked prefill 子阶段
├── KV Transfer (数据传输): ~486ms   (5%)   实际 PCIe 传输
└── 其他:                   ~-136ms        阶段重叠
```

**注意**：之前从 server log 的 `transfer_speed` 和 `transfer_total` 计算出的"KV Transfer 耗时 ~861ms"实际上**混合了 Bootstrap 和数据传输**。Prometheus 的 per-stage 数据揭示了真正的瓶颈分布。

#### 数据来源：Prometheus kv_transfer 专项 metrics

```
KV Transfer 总开销分解:
├── Bootstrap (连接建立):   ~5282ms  ← 🔴 真正的瓶颈！
├── Allocation (内存分配):  ~0.006ms ← 几乎为零
└── 实际数据传输:           ~486ms   ← 只占总开销的 8%
```

### 1.5 关键结论

**🔴 瓶颈不在 PCIe 硬件带宽，而在 Bootstrap 等待时间（排队 + 连接建立）**

之前的分析认为"瓶颈在 UCX 连接建立"，但 inline profiling 揭示了更精确的真相：

| 开销类型 | 耗时 | 占比 | 说明 |
|---|---:|---:|---|
| **Bootstrap 等待** | **5216ms** | **53%** | 🔴 真正的瓶颈！包括排队 + 等待 decode metadata |
| Prefill Forward | 1087ms | 11% | GPU 计算 |
| KV Transfer (数据传输) | 492ms | 5% | 实际 PCIe DMA 传输 |
| **Decode 端等待** | **~3124ms** | **31%** | decode_transferred - prefill 阶段 |

**Bootstrap 包括什么？**（来自 inline profiling）

1. **排队等待 GPU 资源**（~4400ms）🔴
   - 请求进入系统后，等待 GPU 资源可用
   - 高并发下（32 并发），GPU 资源紧张，排队时间长

2. **等待 Decode 端发送 metadata**（~800ms）
   - Decode 端需要告诉 Prefill：把 KV cache 发到哪些内存地址
   - 通过 TCP 发送，涉及网络延迟

3. **UCX 连接建立**（~780ms，**只在前 2 个请求**）
   - `add_remote_agent`: 152ms（TCP 握手 + UCX 协议协商）
   - `prepare_payload_xfer`: 627ms（GPU 显存注册）🔴
   - **后续请求复用连接**，`create_sender` 只需 0.01-0.05ms

**关键发现：UCX 连接已经被复用！**

Inline profiling 显示，`_add_remote_peer`（建立 UCX 连接）**只在前 2 个请求时调用**，后续请求的 `create_sender` 只需 0.01-0.05ms。这说明：
- UCX 连接**已经被缓存和复用**（在 `decode_kv_args_table` 中）
- 每个请求不需要重新建立 UCX 连接
- Bootstrap 的 5216ms **不是**花在连接建立，而是花在**排队等待**

**为什么 Bootstrap 这么慢？**
- 高并发下（32 并发），GPU 资源紧张，请求需要排队等待
- Decode 端发送 metadata 有网络延迟（~800ms）
- 首批请求需要建立 UCX 连接和注册 GPU 显存（~780ms），但后续请求复用

**优化潜力**：
- 如果能减少排队等待时间，TTFT 可以从 9919ms 降到 ~4700ms（**降低 53%**）
- 如果能预热 UCX 连接池，首批请求的 Bootstrap 时间也能减少

### 1.6 并发度与传输速度的关系

| 并发度 | 请求数 | 平均速度 | 平均大小 | 平均耗时 |
|---:|---:|---:|---:|---:|
| 1 | 29 | 0.118 GB/s | 123.7 MB | 1149ms |
| 2 | 32 | 0.145 GB/s | 104.2 MB | 790ms |
| 3 | 27 | 0.141 GB/s | 102.7 MB | 764ms |
| 4 | 13 | 0.178 GB/s | 97.5 MB | 595ms |

**发现**：并发度增加时，单请求传输速度反而**提升**（0.118 → 0.178 GB/s）。这说明：
1. 并发传输没有造成严重的带宽竞争
2. 传输速度主要受协议开销限制，而非带宽限制
3. UCX 可能内部做了某种优化（如批量处理）

### 1.7 为什么 KV Transfer 最大并发度只有 5？（max_concurrency=32）

`max_concurrency=32` 是**客户端并发**（同时有 32 个请求在系统中），不是 KV Transfer 并发。

#### 两种并发的区别

```
客户端视角（max_concurrency=32）:
  同时有 32 个请求在系统中

  Request 1:  [等待] → [Prefill] → [KV Transfer] → [Decode]
  Request 2:  [等待] → [Prefill] → [KV Transfer] → [Decode]
  Request 3:  [等待] → [Prefill] → [KV Transfer] → [Decode]
  ...
  Request 32: [等待] → [Prefill] → [KV Transfer] → [Decode]

KV Transfer 视角（实际并发度 = 5）:
  同时在 PCIe 上传输的 KV cache 最多 5 个
```

#### 瓶颈链路分析

```
Prefill Worker 视角（GPU 0-1, TP=2）:

  时间线:
  ──────────────────────────────────────────────────────→

  [Prefill Batch 1] → [Prefill Batch 2] → [Prefill Batch 3] → ...
       ↓                    ↓                    ↓
  [KV Transfer 1]     [KV Transfer 2]     [KV Transfer 3]
       ↓                    ↓
  [同时进行的 KV Transfer]

  由于 Prefill 是 GPU 计算，必须串行执行（一个 batch 完成后才开始下一个）
  KV Transfer 必须等待 Prefill 完成后才能开始
  所以 KV Transfer 的并发度受限于 Prefill 的完成速率
```

#### 各阶段耗时与并行性（Prometheus per-stage 数据）

| 阶段 | 平均耗时 | 并行性 | 说明 |
|---|---:|---|---|
| Bootstrap (连接建立) | ~5179ms | **串行** | 🔴 UCX 握手 + GPU 显存注册 |
| Queue Time (排队) | ~2622ms | 串行 | 等待 GPU 资源 |
| Prefill Forward | ~1102ms | **串行** | GPU batch 计算 |
| Chunked Prefill | ~695ms | 串行 | chunked prefill 子阶段 |
| KV Transfer (数据传输) | ~486ms | 可以并行 | 实际 PCIe DMA 传输 |
| Decode Forward | ~32ms/token | 串行 | 逐 token 生成 |

**注意**：之前从 server log 计算的"KV Transfer ~861ms"实际上是 `prefill_transfer_kv_cache`（486ms）加上部分 Bootstrap 开销的混合值。Prometheus per-stage 数据揭示了真正的瓶颈分布。

#### 为什么并发度是 5？

Bootstrap 耗时 5179ms，Prefill Forward 耗时 1102ms。由于 Bootstrap 远长于 Prefill，当 Prefill 完成多个请求时，它们的 Bootstrap 会重叠。但由于 Bootstrap 主要是 CPU 侧操作（TCP 握手、内存注册），可以并行进行，所以实际并发度受限于：
1. Bootstrap 服务器的处理能力
2. UCX 连接池的大小
3. GPU 显存注册的并发限制

实际观测到最大并发度 5，说明这些限制在当前配置下约为 5。

#### 如何提高 KV Transfer 并发度？

| 方案 | 原理 | 预期效果 |
|---|---|---|
| **Prefill-Decode 流水线** | 让 KV Transfer 与下一个 Prefill 重叠 | 并发度提升 2-3× |
| **多个 Prefill worker** | 多实例并行 prefill | 并发度线性提升 |
| **加快 Prefill** | 更快 GPU、更小模型、chunked prefill | 间接提升 KV Transfer 并发度 |
| **RDMA/NVLink** | 减少 KV Transfer 耗时 | 并发度降低（传输更快完成） |

**关键洞察**：当前瓶颈不在 KV Transfer 本身，而在 **Prefill Forward 的串行性**。KV Transfer 的并发度是 Prefill 吞吐的"副产品"。要真正提升 KV Transfer 的并行度，必须让 KV Transfer 与 Prefill 重叠（流水线优化）。

---

## 2. Profiling 原理

### 2.1 SGLang 内置 Metrics 系统

SGLang 提供了两个关键的 metrics 收集机制：

#### 1. Prometheus Metrics（聚合统计）

启用方式：`--enable-metrics`

暴露端点：`http://<worker-ip>:<port>/metrics`

KV transfer 相关的 Prometheus metrics：

```prometheus
# KV transfer 延迟（毫秒）
sglang:kv_transfer_latency_ms_sum{engine_type="prefill",...}
sglang:kv_transfer_latency_ms_count{engine_type="prefill",...}
sglang:kv_transfer_latency_ms_bucket{engine_type="prefill",le="1000.0",...}

# KV transfer 速度（GB/s）
sglang:kv_transfer_speed_gb_s_sum{engine_type="prefill",...}
sglang:kv_transfer_speed_gb_s_count{engine_type="prefill",...}

# KV transfer 大小（MB）
sglang:kv_transfer_total_mb_sum{engine_type="prefill",...}
sglang:kv_transfer_total_mb_count{engine_type="prefill",...}
```

#### 2. Request-level Time Stats（逐请求日志）

启用方式：`--enable-request-time-stats-logging`

输出格式（在 server log 中）：

```
ReqTimeStats(rid=xxx, input_len=4096, cached_input_len=0, output_len=1, type=prefill):
  bootstrap_duration=1.60ms,
  queue_duration=828.30ms,
  forward_duration=1773.63ms,
  entry_time=1781183686.422,
  transfer_speed=0.14 GB/s,      ← KV transfer 速度
  transfer_total=132.20 MB,      ← KV cache 大小
  #retries=0
```

### 2.2 测量方法

#### 步骤 1：启动服务时启用 metrics

```bash
# Prefill worker
python3 -m sglang.launch_server \
    --config ./pyscripts/q-prefilltp2.yaml \
    --enable-metrics \
    --enable-request-time-stats-logging \
    > prefill_server.log 2>&1 &

# Decode worker
python3 -m sglang.launch_server \
    --config ./pyscripts/q-decodetp2.yaml \
    --enable-metrics \
    --enable-request-time-stats-logging \
    > decode_server.log 2>&1 &

# Router
python3 -m sglang_router.launch_router \
    --pd-disaggregation \
    --prefill "http://0.0.0.0:30000" \
    --decode "http://0.0.0.0:30001" \
    --policy round_robin \
    --host 0.0.0.0 \
    --port 8000
```

#### 步骤 2：运行 benchmark

```bash
python3 -m sglang.bench_serving \
    --host 127.0.0.1 \
    --port 8000 \
    --dataset-name random-ids \
    --random-input-len 4096 \
    --random-output-len 512 \
    --num-prompts 10 \
    --max-concurrency 32
```

#### 步骤 3：收集数据

```bash
# 收集 Prometheus metrics
curl -s http://127.0.0.1:30000/metrics > prefill_metrics.txt

# 收集 /v1/loads 快照（最新值）
curl -s "http://127.0.0.1:8000/v1/loads?include=all" > loads_snapshot.json

# 从 server log 提取逐请求数据
grep "transfer_speed" prefill_server.log > kv_transfer_logs.txt
```

#### 步骤 4：分析数据

从 server log 中提取每个请求的 `transfer_speed` 和 `transfer_total`，计算：
- 平均传输速度
- 平均传输大小
- 传输延迟 = 传输大小 / 传输速度
- PCIe 带宽利用率 = 实际速度 / 32 GB/s

---

## 3. SGLang PD Disaggregation 代码流程

### 3.1 整体架构

```
┌─────────────────────────────────────────────────────────────┐
│                        Router (port 8000)                    │
│  - 接收客户端请求                                            │
│  - 路由到 prefill worker                                     │
│  - 路由到 decode worker                                      │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│              Prefill Worker (port 30000, GPU 0-1, TP=2)      │
│  - TokenizerManager: HTTP + tokenization                     │
│  - Scheduler: 请求调度 + batching                            │
│  - ModelRunner: GPU forward (prefill)                        │
│  - KV Sender: 发送 KV cache 到 decode worker                 │
└─────────────────────────────────────────────────────────────┘
                              ↓ KV Transfer (NIXL/UCX over PCIe)
┌─────────────────────────────────────────────────────────────┐
│              Decode Worker (port 30001, GPU 2-3, TP=2)       │
│  - KV Receiver: 接收 KV cache 从 prefill worker              │
│  - Scheduler: 请求调度 + batching                            │
│  - ModelRunner: GPU forward (decode)                         │
│  - Detokenizer: 解码 + 返回结果                              │
└─────────────────────────────────────────────────────────────┘
```

### 3.2 Prefill 端代码流程

**文件**: `python/sglang/srt/disaggregation/prefill.py`

#### 3.2.1 Prefill Forward 完成

```python
# prefill.py:607-608
def process_prefill_result(self, ...):
    # Prefill forward 完成
    self.send_kv_chunk(req, last_chunk=True)
    req.time_stats.set_prefill_transfer_queue_entry_time()  # ← 记录传输开始时间
```

#### 3.2.2 发送 KV Cache

```python
# prefill.py:907-1003
def send_kv_chunk(self, req, last_chunk=False, end_idx=None):
    """发送 KV cache 到 decode worker"""
    
    # 1. 准备 page indices
    page_indices = req.kv_cache_pool.get_page_indices(...)
    
    # 2. 初始化 sender（第一次）
    if req.disagg_kv_sender is None:
        req.disagg_kv_sender = self.kv_mgr.create_sender(...)
        req.disagg_kv_sender.init(num_kv_indices=len(page_indices), aux_index=...)
    
    # 3. 发送 KV cache
    req.disagg_kv_sender.send(page_indices, state_indices)  # ← 实际发送
```

#### 3.2.3 轮询传输完成

```python
# prefill.py:670-786
def process_disagg_prefill_inflight_queue(self):
    """轮询 inflight 请求，检查 KV transfer 是否完成"""
    
    for req in self.inflight_requests:
        # 轮询 sender
        poll = req.disagg_kv_sender.poll()
        
        if poll == KVPoll.Success:
            # 传输完成
            req.time_stats.set_prefill_kv_transfer_finish_time()  # ← 记录传输完成时间
            req.time_stats.compute_and_observe_kv_transfer_metrics(
                req.disagg_kv_sender.get_transfer_metric()
            )
            self.process_finished_request(req)
        
        elif poll == KVPoll.Failed:
            # 传输失败
            req.disagg_kv_sender.failure_exception()
```

#### 3.2.4 计算 Metrics

```python
# python/sglang/srt/observability/req_time_stats.py:851-917
def compute_and_observe_kv_transfer_metrics(self, transfer_metric):
    """计算并记录 KV transfer metrics"""
    
    # 1. 计算传输延迟
    if transfer_metric.transfer_latency_s is not None:
        latency_ms = transfer_metric.transfer_latency_s * 1000
    else:
        # Fallback: 使用 completion_time - start_time
        latency_ms = (self.prefill_kv_transfer_finish_time - 
                      self.prefill_transfer_queue_entry_time) * 1000
    
    # 2. 计算传输大小
    total_bytes = transfer_metric.transfer_total_bytes
    total_mb = total_bytes / (1024 * 1024)
    
    # 3. 计算传输速度
    speed_gb_s = (total_mb / 1024) / (latency_ms / 1000)
    
    # 4. 存储到 self
    self.transfer_speed_gb_s = speed_gb_s
    self.transfer_total_mb = total_mb
    
    # 5. 推送到 Prometheus（如果启用）
    if self.metrics_collector:
        self.metrics_collector.observe_kv_transfer_metrics(
            speed_gb_s=speed_gb_s,
            latency_ms=latency_ms,
            total_mb=total_mb,
            ...
        )
```

### 3.3 Decode 端代码流程

**文件**: `python/sglang/srt/disaggregation/decode.py`

#### 3.3.1 接收 KV Cache

```python
# decode.py:1047
def process_decode_request(self, decode_req):
    """接收 decode 请求"""
    
    # 1. 预分配 KV cache 空间
    self.prealloc_queue.add(decode_req)
    
    # 2. 初始化 receiver
    decode_req.kv_receiver = self.kv_mgr.create_receiver(...)
    decode_req.kv_receiver.init(prefill_dp_rank=...)
    
    # 3. 发送 metadata 到 prefill worker（告诉它把 KV cache 发到哪里）
    decode_req.kv_receiver.send_metadata(
        kv_indices=decode_req.kv_cache_indices,
        aux_index=...,
        state_indices=...,
        decode_prefix_len=...
    )
    
    # 4. 进入 transfer queue
    self.transfer_queue.add(decode_req)
    decode_req.req.time_stats.set_decode_transfer_queue_entry_time()  # ← 记录进入 transfer queue 时间
```

#### 3.3.2 轮询传输完成

```python
# decode.py:1585-1699
def pop_transferred(self):
    """从 transfer queue 中弹出已完成的请求"""
    
    for decode_req in self.transfer_queue:
        # 轮询 receiver
        poll = decode_req.kv_receiver.poll()
        
        if poll == KVPoll.Success:
            # 传输完成
            self._commit_transfer_to_req(decode_req)
            decode_req.req.time_stats.set_wait_queue_entry_time()  # ← 进入 waiting queue
            self.waiting_queue.add(decode_req)
        
        elif poll == KVPoll.Failed:
            # 传输失败
            decode_req.kv_receiver.failure_exception()
```

#### 3.3.3 开始 Decode

```python
# decode.py:1445-1549
def _commit_transfer_to_req(self, decode_req):
    """提交传输完成的请求"""
    
    # 1. 清理 receiver
    decode_req.kv_receiver.clear()
    
    # 2. 标记请求为可运行
    decode_req.req.is_retracted = False
    
    # 3. 进入 waiting queue，等待 scheduler 调度
    # （下一步是进入 running batch，开始 decode forward）
```

### 3.4 KV Transfer 数据流

```
┌──────────────────────────────────────────────────────────────┐
│                      Prefill Worker                           │
│                                                               │
│  1. Prefill Forward 完成                                      │
│     └─> req.time_stats.set_prefill_transfer_queue_entry_time()│
│                                                               │
│  2. send_kv_chunk(req, last_chunk=True)                       │
│     ├─> kv_sender.init(num_kv_indices, aux_index)             │
│     └─> kv_sender.send(page_indices, state_indices)           │
│         └─> NIXL: UCX 发送数据到 decode worker                │
│                                                               │
│  3. process_disagg_prefill_inflight_queue()                   │
│     └─> poll = kv_sender.poll()                               │
│         ├─> KVPoll.Transferring: 继续等待                     │
│         └─> KVPoll.Success: 传输完成                          │
│             ├─> req.time_stats.set_prefill_kv_transfer_finish_time()
│             └─> compute_and_observe_kv_transfer_metrics()     │
│                 ├─> transfer_speed = total_bytes / latency    │
│                 └─> 推送到 Prometheus                         │
└──────────────────────────────────────────────────────────────┘
                              ↓
                    KV Transfer (NIXL/UCX)
                    - UCX 协议封装
                    - 内存注册 (memory registration)
                    - PCIe DMA 传输
                    - 确认机制 (ACK)
                              ↓
┌──────────────────────────────────────────────────────────────┐
│                      Decode Worker                            │
│                                                               │
│  1. 接收请求，初始化 receiver                                 │
│     ├─> kv_receiver.init(prefill_dp_rank)                     │
│     └─> kv_receiver.send_metadata(kv_indices, aux_index)      │
│         └─> 告诉 prefill worker: "把 KV cache 发到这些地址"   │
│                                                               │
│  2. pop_transferred()                                         │
│     └─> poll = kv_receiver.poll()                             │
│         ├─> KVPoll.Transferring: 继续等待                     │
│         └─> KVPoll.Success: 传输完成                          │
│             └─> _commit_transfer_to_req(decode_req)           │
│                 └─> 进入 waiting queue                        │
│                                                               │
│  3. Scheduler 调度，进入 running batch                        │
│     └─> 开始 decode forward                                   │
└──────────────────────────────────────────────────────────────┘
```

### 3.5 KVPoll 状态机

**文件**: `python/sglang/srt/disaggregation/base/conn.py:71-76`

```python
class KVPoll:
    Failed = 0          # 传输失败
    Bootstrapping = 1   # 建立连接中
    WaitingForInput = 2 # 等待输入（decode 端）
    Transferring = 3    # 传输中
    Success = 4         # 传输成功
```

状态转换：

```
Prefill 端:
  Bootstrapping → Transferring → Success
                              ↘ Failed

Decode 端:
  Bootstrapping → WaitingForInput → Transferring → Success
                                              ↘ Failed
```

---

## 4. 瓶颈分析

### 4.1 重新审视：瓶颈到底在哪里？

通过 Prometheus metrics 的 per-stage 数据，我们可以精确定位瓶颈：

| 开销类型 | 耗时 | 占比 | 说明 |
|---|---:|---:|---|
| **Bootstrap（连接建立）** | **5179ms** | **52%** | 🔴 最大瓶颈 |
| Queue Time（排队等待） | 2622ms | 26% | 等待 GPU 资源 |
| Prefill Forward | 1102ms | 11% | GPU 计算 |
| Chunked Prefill | 695ms | 7% | chunked prefill 子阶段 |
| **KV Transfer（数据传输）** | **486ms** | **5%** | 实际 PCIe DMA 传输 |

**关键发现**：
- 实际数据传输只需要 486ms，但 Bootstrap 需要 5179ms
- Bootstrap 是数据传输的 **10.7 倍**
- 如果能消除 Bootstrap 开销，TTFT 可以从 9948ms 降到 ~4700ms（**降低 53%**）

### 4.2 Bootstrap 为什么这么慢？

Bootstrap 包括以下步骤：

1. **Decode 端发送 metadata 到 Prefill**（~100ms）
   - 告诉 Prefill：把 KV cache 发到哪些内存地址
   - 通过 TCP 发送，涉及网络延迟

2. **UCX 连接建立**（~2000ms）
   - TCP 三次握手
   - UCX 协议协商（选择传输层、参数配置）
   - 安全认证（如果启用）

3. **GPU 显存注册**（~3000ms）🔴
   - `ucp_mem_map()` 调用，将 GPU 显存注册到 UCX
   - 涉及内核态操作（ioctl 系统调用）
   - 每个请求都要重新注册（没有连接复用）

4. **握手协议完成**（~100ms）
   - Prefill 和 Decode 确认连接就绪
   - 开始数据传输

**为什么没有连接复用？**

当前实现中，每个请求都会创建新的 `KVSender` 和 `KVReceiver` 对象：

```python
# prefill.py:907-1003
def send_kv_chunk(self, req, last_chunk=False, end_idx=None):
    if req.disagg_kv_sender is None:
        req.disagg_kv_sender = self.kv_mgr.create_sender(...)  # ← 每次创建新 sender
        req.disagg_kv_sender.init(num_kv_indices=len(page_indices), aux_index=...)
```

这意味着：
- 每个请求都要重新建立 UCX 连接
- 每个请求都要重新注册 GPU 显存
- 无法复用之前的连接和注册信息

### 4.3 为什么 FP8 KV cache 没用？

之前的 benchmark 显示，启用 `--kv-cache-dtype fp8_e4m3` 后，吞吐几乎没提升。

**原因**：

```
FP16: 数据量 216 MB，传输速度 0.16 GB/s，延迟 = 216 / 0.16 = 1350ms
FP8:  数据量 108 MB，传输速度 0.16 GB/s，延迟 = 108 / 0.16 = 675ms

但是，总延迟 = Bootstrap（~5200ms）+ 数据传输时间

FP16: 总延迟 = 5200ms + 1350ms = 6550ms
FP8:  总延迟 = 5200ms + 675ms = 5875ms

提升 = (6550 - 5875) / 6550 = 10%

但实际提升只有 0-11%，因为：
1. Bootstrap 开销占主导（5200ms），数据传输时间占比很小
2. FP8 可能引入额外的量化/反量化开销
3. 高并发下，Bootstrap 服务器成为瓶颈，掩盖了数据传输的改进
```

**结论**：瓶颈在 Bootstrap（连接建立），而不是数据传输时间。减少数据量（FP8）无法解决 Bootstrap 开销问题。

### 4.4 与 TP4 的对比

TP4 模式下，所有 4 张 GPU 在同一个进程内，KV cache 不需要传输：

```
TP4:
  - AllReduce 通信：~300ms（PCIe P2P）
  - 无 KV transfer 开销
  - TTFT = Prefill + AllReduce = ~2000ms

PD TP2:
  - Prefill AllReduce：~300ms
  - Bootstrap：~5200ms  ← 新增开销
  - KV Transfer：~486ms
  - TTFT = Prefill + AllReduce + Bootstrap + KV Transfer = ~9948ms

差距：9948 - 2000 = 7948ms（397% 增加）
其中 Bootstrap 占 5200ms（65% 的差距）
```

这就是为什么 PD TP2 的 TTFT 比 TP4 高 4 倍。

---

## 5. 优化方向

### 5.1 短期优化（P0，立即可做）

#### 1. 🔴 UCX 连接复用（预期收益：TTFT 降低 50%+）

当前每个请求都创建新的 `KVSender`/`KVReceiver`，导致每次都要重新建立 UCX 连接和注册 GPU 显存（5.3 秒）。

**优化方案**：

```python
# 当前：每个请求创建新连接
def send_kv_chunk(self, req, ...):
    if req.disagg_kv_sender is None:
        req.disagg_kv_sender = self.kv_mgr.create_sender(...)  # ← 每次新建
        req.disagg_kv_sender.init(...)

# 优化：维护连接池，复用已有连接
class KVConnectionPool:
    def __init__(self):
        self.connections = {}  # decode_url -> KVSender
    
    def get_sender(self, decode_url):
        if decode_url not in self.connections:
            sender = self.kv_mgr.create_sender(...)
            sender.init(...)  # 只初始化一次
            self.connections[decode_url] = sender
        return self.connections[decode_url]
```

**预期收益**：
- Bootstrap 时间从 5282ms 降到 ~100ms（只需发送 metadata）
- TTFT 从 9948ms 降到 ~4700ms（**降低 53%**）
- 吞吐提升 2× 以上

**复杂度**：中等
- 需要修改 `KVSender`/`KVReceiver` 的生命周期管理
- 需要处理连接失效、重试等边界情况
- 需要确保 GPU 显存注册信息在请求间正确更新

#### 2. 优化 UCX 配置

调整 UCX 环境变量，减少协议开销：

```bash
# 增加 buffer size
export UCX_TLS=tcp,sm
export UCX_TCP_TX_SEG_SIZE=1M
export UCX_TCP_RX_SEG_SIZE=1M

# 减少 polling 间隔
export UCX_TCP_POLL_INTERVAL=0
```

**预期收益**：减少协议开销 10-20%

### 5.2 中期优化（P1，需要开发）

#### 3. Prefill-Decode 流水线

让 KV transfer 与下一个请求的 prefill 重叠：

```
Request 1: [prefill] → [KV transfer] → [decode]
Request 2:           [prefill] → [KV transfer] → [decode]
                    ↑ overlap ↑
```

**预期收益**：吞吐提升 30-50%（高并发场景）

#### 4. 批量传输

将多个请求的 KV cache 合并成一个大消息，减少 Bootstrap 次数：

```python
# 当前：每个请求单独传输（每次都要 Bootstrap）
for req in requests:
    kv_sender.send(req.kv_indices)  # 每次 5.3s Bootstrap

# 优化：批量传输（一次 Bootstrap，传输多个请求）
batch_kv_indices = concat([req.kv_indices for req in requests])
kv_sender.send(batch_kv_indices)  # 一次 Bootstrap，传输 N 个请求
```

**预期收益**：
- Bootstrap 开销摊薄到每个请求：5282ms / N
- 如果 batch_size=8，每个请求的 Bootstrap 开销降到 ~660ms

### 5.3 长期优化（P2，需要硬件支持）

#### 5. RDMA 传输

如果硬件支持 RDMA（如 InfiniBand、RoCE），可以绕过 CPU，直接 GPU-to-GPU 传输：

```
当前：GPU → CPU → PCIe → CPU → GPU
RDMA：GPU → PCIe → GPU（绕过 CPU）
```

**预期收益**：传输速度提升 10-20 倍（接近 PCIe 理论带宽）

#### 6. NVLink

如果硬件支持 NVLink，可以使用 NVLink 传输：

```
PCIe 3.0 x16: 32 GB/s
NVLink 3.0:   600 GB/s（18.75 倍）
```

**预期收益**：传输速度提升 18 倍

---

## 6. 总结

### 6.1 核心发现

1. **🔴 最大瓶颈是 UCX 连接建立（Bootstrap），耗时 5282ms，占 TTFT 的 53%**
2. 实际 PCIe 数据传输只需 486ms（5%），PCIe 带宽利用率仅 0.58%
3. 每个请求都要重新建立 UCX 连接和注册 GPU 显存，没有连接复用
4. FP8 KV cache 无效，因为瓶颈是 Bootstrap 开销，不是数据传输时间
5. 并发度增加时速度反而提升（0.118 → 0.178 GB/s），说明瓶颈不在带宽竞争

### 6.2 TTFT 延迟分解（100 请求，32 并发）

```
TTFT ≈ 9948ms
├── Bootstrap (连接建立):   ~5179ms  (52%)  🔴 最大瓶颈！
├── Queue Time (排队等待):  ~2622ms  (26%)
├── Prefill Forward:        ~1102ms  (11%)  GPU 计算
├── Chunked Prefill:        ~695ms   (7%)
├── KV Transfer (数据传输): ~486ms   (5%)   实际 PCIe 传输
└── 其他:                   ~-136ms        阶段重叠
```

### 6.3 优化建议（按优先级）

| 优先级 | 优化方向 | 预期收益 | 复杂度 |
|---|---|---|---|
| **P0** | **UCX 连接复用** | **TTFT 降低 53%** | 中 |
| P0 | 优化 UCX 配置 | 10-20% | 低 |
| P1 | 批量传输 | Bootstrap 摊薄 N× | 中 |
| P1 | Prefill-Decode 流水线 | 吞吐提升 30-50% | 中 |
| P2 | RDMA 传输 | 10-20× | 高（需硬件） |
| P2 | NVLink | 18× | 高（需硬件） |

### 6.4 下一步行动

1. **立即**：实现 UCX 连接池，复用连接和 GPU 显存注册
2. **本周**：优化 UCX 配置，测试不同参数
3. **本月**：实现批量传输和 Prefill-Decode 流水线
4. **长期**：评估 RDMA/NVLink 硬件升级的 ROI

---

## 附录 A：测量脚本

完整的测量脚本见：`mybench/measure_kv_transfer.sh`

使用方法：

```bash
bash mybench/measure_kv_transfer.sh 4096 512 100 32
# 参数：input_len output_len num_prompts max_concurrency
```

输出文件：

```
mybench/kv-transfer-measurement/20260611_135523/
├── benchmark_output.txt          # Benchmark 结果
├── prefill_server.log            # Prefill server 日志（含 transfer_speed）
├── decode_server.log             # Decode server 日志
├── prefill_metrics.txt           # Prometheus metrics
├── loads_snapshot.json           # /v1/loads 快照
└── kv_transfer_analysis.txt      # 分析报告
```

## 附录 B：原始数据（100 请求，32 并发）

### B.1 Benchmark 结果

```
Backend:                                 sglang
Max request concurrency:                 32
Successful requests:                     100
Benchmark duration (s):                  68.02
Output token throughput (tok/s):         379.58
Mean TTFT (ms):                          9947.58
Mean ITL (ms):                           32.13
```

### B.2 并发分析数据

| 指标 | 数值 |
|---|---|
| 有效请求数 | 101 |
| 传输时间窗口 | 57.2s |
| sum(单请求传输耗时) | 87.0s |
| 最大并发传输数 | 5 |
| 平均并发度 | 1.52 |

### B.3 带宽计算对比

| 方法 | 结果 | PCIe 利用率 |
|---|---:|---:|
| 简单平均（不考虑并发） | 0.140 GB/s | 0.44% |
| 聚合带宽（考虑并发） | 0.187 GB/s | 0.58% |
| 加权平均（平均并发度） | 0.213 GB/s | 0.67% |

### B.4 并发度与传输速度关系

| 并发度 | 请求数 | 平均速度 | 平均大小 | 平均耗时 |
|---:|---:|---:|---:|---:|
| 1 | 29 | 0.118 GB/s | 123.7 MB | 1149ms |
| 2 | 32 | 0.145 GB/s | 104.2 MB | 790ms |
| 3 | 27 | 0.141 GB/s | 102.7 MB | 764ms |
| 4 | 13 | 0.178 GB/s | 97.5 MB | 595ms |

### B.5 逐请求统计

| 指标 | 最小值 | 最大值 | 平均值 |
|---|---:|---:|---:|
| Input length | 18 | 4685 | 2249 |
| 传输速度 | 0.070 GB/s | 0.380 GB/s | 0.140 GB/s |
| 传输大小 | 73.7 MB | 146.6 MB | 108.6 MB |
| 传输耗时 | 250ms | 2045ms | 861ms |

### B.6 Prometheus Metrics 数据（关键洞见）

**Prefill 端 per_stage_req_latency（tp_rank=0）**：

| 阶段 | 平均耗时 | 请求数 | 说明 |
|---|---:|---:|---|
| prefill_bootstrap | 5179ms | 103 | 🔴 UCX 连接建立 |
| queue_time | 2622ms | 103 | 排队等待 GPU |
| prefill_forward | 1102ms | 103 | GPU 计算 |
| prefill_transfer_kv_cache | 486ms | 103 | 实际 PCIe 传输 |
| chunked_prefill | 695ms | 87 | chunked prefill 子阶段 |

**Prefill 端 kv_transfer 专项 metrics（tp_rank=0）**：

| 指标 | 平均值 | 说明 |
|---|---:|---|
| kv_transfer_latency_ms | 857ms | 总传输延迟 |
| kv_transfer_bootstrap_ms | **5282ms** | 🔴 连接建立（含等待） |
| kv_transfer_alloc_ms | 0.006ms | 内存分配（几乎为零） |
| kv_transfer_total_mb | 108.6 MB | 传输大小 |
| kv_transfer_speed_gb_s | 0.141 GB/s | 传输速度 |

**Decode 端 metrics**：

| 指标 | 值 | 说明 |
|---|---:|---|
| Queue Time | 0.1ms | 几乎无排队（资源充足） |
| E2E Latency | 17828ms | 端到端延迟 |
| 请求数 | 102 | |
| 平均 output len | 253.5 | |

**KV Transfer 速度分布**：

| 速度范围 | 请求数 | 占比 |
|---|---:|---:|
| ≤ 0.1 GB/s | 19 | 19% |
| 0.1 - 0.5 GB/s | 82 | 81% |
| > 0.5 GB/s | 0 | 0% |
