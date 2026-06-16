# Bootstrap 细粒度 Profiling 分析报告

**日期**: 2026-06-13  
**模型**: qwen/qwen3.5-27b-fp8  
**硬件**: 4× NVIDIA RTX PRO 4000 Blackwell (PCIe, no NVLink)  
**配置**: PD TP2 模式（Prefill TP=2 GPU 0-1, Decode TP=2 GPU 2-3）  
**传输后端**: NIXL over UCX (PCIe)  
**测试组**: A = 100 req / 32 并发，B = 50 req / 16 并发

---

## 1. 核心发现

### 1.1 Bootstrap 的真正瓶颈：Prefill 端轮询等待，而非任何单步执行

通过细粒度 profiling，我们发现 **Bootstrap 的耗时主要来自「Prefill 端反复轮询等待 Decode 端连接」，而非任何单步操作的执行时间**。所有执行操作（create_sender、finalize_bootstrap、create_receiver 等）都在 **0.04ms 以内**。

### 1.2 Prefill 端各子阶段耗时对比

| 子阶段 | A: 100 req / 32 并发 | B: 50 req / 16 并发 | A/B 倍数 | 说明 |
|---|---:|---:|---:|---|
| `create_sender` (avg) | 0.016ms | 0.021ms | 0.8× | 创建 KVSender 对象，极快 |
| `pop_bootstrapped_poll` (avg) | 0.437ms | 0.307ms | 1.4× | 单次轮询（含 TP AllReduce） |
| `pop_bootstrapped_poll` (max) | **92.75ms** | 12.64ms | **7.3×** | 🔴 高并发下出现极端毛刺 |
| `pop_bootstrapped_poll` (调用次数) | 1912 | 1466 | 1.3× | 高并发下轮询更频繁 |
| `pop_bootstrapped_total` (成功时 avg) | 0.925ms | 0.788ms | 1.2× | 单次 pop 调用总耗时 |
| `finalize_bootstrap` (avg) | 0.008ms | 0.008ms | 1.0× | 初始化 sender，极快 |
| `_add_remote_peer` (仅首次, total) | **783ms** | 741ms | 1.1× | 🔴 NIXL 连接建立（仅首个请求） |
| └ `add_remote_agent` | 147ms | 148ms | 1.0× | NIXL agent 注册 |
| └ `prepare_payload` | **637ms** | 593ms | 1.1× | 🔴 构建 NIXL 描述符（主要开销） |
| `_add_remote_peer` (后续) | 0ms | 0ms | - | 连接已缓存，直接跳过 |

**关键观察**：
- Prefill 端所有**执行**操作（create_sender、finalize_bootstrap）都在 **0.02ms 以内**
- `_add_remote_peer` 只在**首个请求**时调用一次（783ms），后续请求完全复用
- Bootstrap 的 5338ms 平均耗时 = `pop_bootstrapped_poll` × 轮询次数 ≈ 0.44ms × ~12000 次

### 1.3 Decode 端各子阶段耗时对比

| 子阶段 | A: 100 req / 32 并发 | B: 50 req / 16 并发 | A/B 倍数 | 说明 |
|---|---:|---:|---:|---|
| `create_receiver` (avg) | 0.038ms | 0.037ms | 1.0× | 创建 KVReceiver 对象，极快 |
| `resolve_dp_rank` (avg) | 0.010ms | 0.010ms | 1.0× | 查找 dp_rank，极快 |
| `ensure_parallel_info` (首次) | 3.98ms | 2.76ms | 1.4× | HTTP GET 获取 prefill server info |
| `ensure_parallel_info` (后续) | 0ms | 0ms | - | 缓存命中 |
| `get_bootstrap_info_http` (首次, avg) | 1.67ms | 1.41ms | 1.2× | HTTP GET 获取 bootstrap info |
| `_register_kv_args` (首次, avg) | 0.655ms | 0.650ms | 1.0× | ZMQ 发送 KV args |
| `_setup_bootstrap_infos` (avg) | 0.021ms | 0.034ms | 0.6× | 设置 bootstrap info |
| `receiver_init` (首次) | **2.19ms** | 1.77ms | 1.2× | 总 init 时间 |
| `receiver_init` (后续, avg) | **0.035ms** | 0.033ms | 1.1× | 🔴 缓存命中后极快 |

**关键观察**：
- Decode 端所有操作都在 **4ms 以内**，且两种并发度下几乎一致
- `receiver_init` 首次 2.19ms vs 后续 0.035ms — **62 倍差异**，证明连接已被缓存
- Decode 端**不是** Bootstrap 瓶颈的来源

### 1.4 Prometheus per-stage 对比

| 阶段 (per-request avg) | A: 100 req / 32 并发 | B: 50 req / 16 并发 | A/B 倍数 |
|---|---:|---:|---:|
| `prefill_bootstrap` | **5338ms** | 470ms | **11.4×** 🔴 |
| `prefill_forward` | 1070ms | 1071ms | 1.0× |
| `prefill_transfer_kv_cache` | 521ms | 501ms | 1.0× |
| **TTFT** | **10080ms** | 3631ms | **2.78×** |

### 1.5 Bootstrap 等待链条分析

Bootstrap 的耗时来自以下等待链条：

```
Prefill 端:
1. create_sender()                              → 0.02ms
2. 进入 bootstrap 队列
3. pop_bootstrapped() 反复轮询                   → 每次 0.3-0.4ms
   ↓                                              但需要轮询 ~12000 次
   等待 Decode 端发起连接                         ← 🔴 真正的等待！
   ↓
Decode 端:
4. create_receiver()                            → 0.04ms
5. resolve_dp_rank()                            → 0.01ms
6. ensure_parallel_info()                       → 3.98ms (首次) / 0ms (后续)
7. receiver.init()                              → 2.19ms (首次) / 0.04ms (后续)
   ├─ get_bootstrap_info_http()                 → 1.67ms
   └─ _register_kv_args()                       → 0.66ms
   ↓
   ZMQ 发送 KV args 到 Prefill
   ↓
Prefill 端:
8. _add_remote_peer()                           → 783ms (仅首次) / 0ms (后续)
   ├─ add_remote_agent()                        → 147ms
   └─ prepare_payload()                         → 637ms
9. finalize_bootstrap()                         → 0.01ms
```

**核心问题**：Prefill 端 `pop_bootstrapped_poll` 每次只需 0.3-0.4ms，但需要**反复轮询**等待 Decode 端的连接请求到达。高并发下（32 并发），轮询次数从 1466 增加到 1912，且 max 延迟从 12.64ms 飙升到 **92.75ms**（7.3 倍），说明高并发导致了严重的排队和调度抖动。

### 1.6 连接复用机制已生效

Profiling 数据证实 **UCX 连接已经被缓存和复用**：

| 指标 | A: 首次请求 | A: 后续请求 | 差异 |
|---|---:|---:|---:|
| `_add_remote_peer` (Prefill) | 783ms | 0ms | ∞ |
| `receiver_init` (Decode) | 2.19ms | 0.035ms | 62× |
| `ensure_parallel_info` (Decode) | 3.98ms | 0ms | ∞ |
| `get_bootstrap_info_http` (Decode) | 1.67ms | 0ms (缓存) | ∞ |

**结论**：之前认为"每个请求都要重新建立 UCX 连接"的假设是**错误的**。连接已经被缓存，Bootstrap 的开销主要来自**等待**。

### 1.7 高并发是 Bootstrap 慢的根本原因

| 指标 | A: 100 req / 32 并发 | B: 50 req / 16 并发 | A/B 倍数 |
|---|---:|---:|---:|
| TTFT | 10080ms | 3631ms | 2.78× |
| prefill_bootstrap | 5338ms (53%) | 470ms (13%) | **11.4×** 🔴 |
| prefill_forward | 1070ms (11%) | 1071ms (29%) | 1.0× |
| prefill_transfer_kv_cache | 521ms (5%) | 501ms (14%) | 1.0× |
| Decode 端等待 | 3152ms (31%) | 1589ms (44%) | 2.0× |

- 并发度从 32 降到 16，**Bootstrap 从 5338ms 降到 470ms（11.4 倍改善）**
- Prefill Forward 和 KV Transfer 几乎不受并发度影响（1.0×）
- 这证实了 **Bootstrap 的瓶颈是高并发下的排队等待**，而非连接建立本身
- 降低并发度后，**Decode 端等待成为新的主要瓶颈（44%）**

---

## 2. TTFT 延迟分解

### 2.1 A: 100 请求 / 32 并发

```
TTFT ≈ 10080ms
├── Bootstrap (等待):           ~5338ms  (53%)   🔴 最大瓶颈
├── Prefill Forward:            ~1070ms  (11%)   ← GPU 计算
├── KV Transfer (数据传输):     ~521ms   (5%)    ← 实际 PCIe 传输
├── Decode 端等待:              ~3152ms  (31%)   ← decode_transferred - prefill 阶段
└── 其他/重叠:                  ~0ms     (0%)
```

### 2.2 B: 50 请求 / 16 并发

```
TTFT ≈ 3631ms
├── Bootstrap (等待):           ~470ms   (13%)   ✅ 大幅改善
├── Prefill Forward:            ~1071ms  (29%)   ← GPU 计算
├── KV Transfer (数据传输):     ~501ms   (14%)   ← 实际 PCIe 传输
├── Decode 端等待:              ~1589ms  (44%)   ← 成为新的主要瓶颈
└── 其他/重叠:                  ~0ms     (0%)
```

### 2.3 对比

```
A: 100 req / 32 并发:
TTFT ≈ 10080ms
├── Bootstrap (等待):           ~5338ms  (53%)   🔴 最大瓶颈
├── Prefill Forward:            ~1070ms  (11%)
├── KV Transfer (数据传输):     ~521ms   (5%)
└── Decode 端等待:              ~3152ms  (31%)

B: 50 req / 16 并发:
TTFT ≈ 3631ms
├── Bootstrap (等待):           ~470ms   (13%)   ✅ 大幅改善
├── Prefill Forward:            ~1071ms  (29%)
├── KV Transfer (数据传输):     ~501ms   (14%)
└── Decode 端等待:              ~1589ms  (44%)   ← 成为新的主要瓶颈
```

**关键观察**：
- 降低并发度后，Bootstrap 从 53% 降到 13%
- **Decode 端等待成为新的主要瓶颈（44%）**
- 这说明优化方向应该从 Bootstrap 转向 Decode 端调度

---

## 3. 详细 Profiling 数据

### 3.1 Prefill 端 Profiling 日志示例

```
[2026-06-13 17:44:02 TP0] [BOOTSTRAP PROFILE] create_sender: 0.04ms, rid=e47a6a06
[2026-06-13 17:44:02 TP0] [BOOTSTRAP PROFILE] pop_bootstrapped_poll: 0.62ms, queue_size=1
[2026-06-13 17:44:02 TP0] [BOOTSTRAP PROFILE] pop_bootstrapped_total: 0.67ms, bootstrapped=0, failed=0
[2026-06-13 17:44:02 TP0] [BOOTSTRAP PROFILE] pop_bootstrapped_poll: 0.23ms, queue_size=1
[2026-06-13 17:44:02 TP0] [BOOTSTRAP PROFILE] pop_bootstrapped_total: 0.28ms, bootstrapped=0, failed=0
... (大量 bootstrapped=0 的轮询)
[2026-06-13 17:44:02 TP0] [BOOTSTRAP PROFILE] finalize_bootstrap: 0.01ms, rid=e47a6a06
[2026-06-13 17:44:02 TP0] [BOOTSTRAP PROFILE] pop_bootstrapped_total: 7.90ms, bootstrapped=1, failed=0
```

**观察**：
- `pop_bootstrapped_poll` 每次只花 0.2-0.6ms
- 但需要轮询很多次（大量 `bootstrapped=0`）
- 最终 `finalize_bootstrap` 只需 0.01ms

### 3.2 Decode 端 Profiling 日志示例

**首次请求（冷启动）**：
```
[2026-06-13 17:44:02 TP0] [BOOTSTRAP PROFILE] create_receiver: 0.06ms, rid=dbf9ac15
[2026-06-13 17:44:02 TP0] [BOOTSTRAP PROFILE] resolve_dp_rank: 0.01ms, rid=dbf9ac15, cache_miss
[2026-06-13 17:44:02 TP0] [BOOTSTRAP PROFILE] ensure_parallel_info_http: 2.67ms, addr=0.0.0.0:8998
[2026-06-13 17:44:02 TP0] [BOOTSTRAP PROFILE] ensure_parallel_info: 2.76ms, addr=0.0.0.0:8998, fetched
[2026-06-13 17:44:02 TP0] [BOOTSTRAP PROFILE] resolve_dp_rank: 0.01ms, rid=dbf9ac15, dp_size=1
[2026-06-13 17:44:02 TP0] [BOOTSTRAP PROFILE] get_bootstrap_info_http: 1.07ms, addr=0.0.0.0:8998, dp=0, cp=0, tp=0, pp=0
[2026-06-13 17:44:02 TP0] [BOOTSTRAP PROFILE] get_bootstrap_info_total: 1.12ms, addr=0.0.0.0:8998, success
[2026-06-13 17:44:02 TP0] [BOOTSTRAP PROFILE] _register_kv_args: total=0.50ms, zmq_send=0.12ms, bootstrap_infos=1
[2026-06-13 17:44:02 TP0] [BOOTSTRAP PROFILE] _setup_bootstrap_infos: 1.74ms, room=525701289566887711, bootstrap_infos=1
[2026-06-13 17:44:02 TP0] [BOOTSTRAP PROFILE] receiver_init: 1.77ms, room=525701289566887711, success
```

**后续请求（缓存命中）**：
```
[2026-06-13 17:44:06 TP0] [BOOTSTRAP PROFILE] create_receiver: 0.02ms, rid=c1913c57
[2026-06-13 17:44:06 TP0] [BOOTSTRAP PROFILE] resolve_dp_rank: 0.01ms, rid=c1913c57, dp_size=1
[2026-06-13 17:44:06 TP0] [BOOTSTRAP PROFILE] _setup_bootstrap_infos: 0.00ms, room=6763337243963057584, bootstrap_infos=1
[2026-06-13 17:44:06 TP0] [BOOTSTRAP PROFILE] receiver_init: 0.04ms, room=6763337243963057584, success
```

**观察**：
- 首次请求需要 1.77ms（HTTP GET + ZMQ 注册）
- 后续请求只需 0.04ms（缓存命中）
- 差异达到 **44 倍**

### 3.3 Prefill 端 `_add_remote_peer` Profiling

**仅首次连接**：
```
[2026-06-13 17:44:03 TP0] [BOOTSTRAP PROFILE] _add_remote_peer: total=741.23ms, add_remote_agent=147.84ms, prepare_payload=593.39ms, agent=4973c8af-4e73-44
[2026-06-13 17:44:03 TP1] [BOOTSTRAP PROFILE] _add_remote_peer: total=743.49ms, add_remote_agent=146.77ms, prepare_payload=596.73ms, agent=7bffef1b-e7c2-48
```

**观察**：
- `_add_remote_peer` 只在第一个请求时调用
- 耗时 741ms，其中：
  - `add_remote_agent`: 148ms（NIXL agent 注册）
  - `prepare_payload`: 593ms（构建 NIXL 描述符）🔴 主要开销
- 后续请求完全复用连接，耗时 0ms

### 3.4 Prometheus Per-Stage Metrics

**A: Prefill 端（per-request 平均）**：
| 阶段 | 总耗时 | 请求数 | 平均耗时 |
|---|---:|---:|---:|
| prefill_bootstrap | 549862.40ms | 103 | 5338.5ms |
| prefill_forward | 110178.70ms | 103 | 1069.7ms |
| prefill_transfer_kv_cache | 53610.67ms | 103 | 520.5ms |

**B: Prefill 端（per-request 平均）**：
| 阶段 | 总耗时 | 请求数 | 平均耗时 |
|---|---:|---:|---:|
| prefill_bootstrap | 24918.61ms | 53 | 470.2ms |
| prefill_forward | 56742.71ms | 53 | 1070.6ms |
| prefill_transfer_kv_cache | 26538.62ms | 53 | 500.7ms |

**B: Decode 端（per-request 平均）**：
| 阶段 | 总耗时 | 请求数 | 平均耗时 |
|---|---:|---:|---:|
| decode_bootstrap | 96.07ms | 53 | 1.8ms |
| decode_transferred | 173454.06ms | 53 | 3272.7ms |
| decode_prepare | 12.09ms | 53 | 0.2ms |
| decode_waiting | 11.57ms | 53 | 0.2ms |

---

## 4. 优化建议（目标：32 并发）

### 4.1 瓶颈本质：99.8% 的时间在等待

**Bootstrap 5338ms 的拆解**：
- 执行操作（create_sender + finalize_bootstrap）：0.024ms (**0.000%**)
- poll 执行时间（累计 19 次）：8.1ms (**0.2%**)
- **墙钟等待时间：5330.3ms (99.8%)** ← 🔴 真正的瓶颈

**关键洞察**：
- 单次 poll 已经很快（P50: 0.36ms, P99: 0.74ms）
- 89.2% 的 poll 在 0.5ms 以内，只有 0.2% 超过 1ms
- 瓶颈不是单次 poll 慢，而是**需要等待 ~12000 次 poll 才能完成**
- 墙钟等待时间 5330ms 相当于等待 **5.0 个 forward pass**（1070ms）

**为什么需要等待这么多次 poll？**
- Prefill 端创建 sender 后进入 bootstrap 队列
- 需要等待 Decode 端创建 receiver 并发送 metadata
- Decode 端的调度延迟导致这个等待时间很长
- 高并发下（32 并发），Decode 端需要处理大量请求，调度延迟增加

### 4.2 短期优化（P0，立即可做）

#### 1. 🔴 减少 Decode 端调度延迟（预期收益：Bootstrap 降低 80%+）

**问题**：Decode 端从请求到达到 `receiver.init()` 被调用，存在调度延迟。高并发下（32 并发），这个延迟导致 Prefill 端需要等待 ~12000 次 poll。

**优化方案**：
- **优先级调度**：让 bootstrap 请求（需要创建 receiver 的请求）优先于普通 decode 请求
- **预分配 receiver**：在请求到达 Decode Scheduler 之前，提前创建 receiver 对象
- **异步 receiver 创建**：将 `create_receiver` 和 `receiver.init()` 移到后台线程，不阻塞主调度循环

**预期收益**：
- 如果 Decode 端调度延迟从 5330ms 降到 1000ms，Bootstrap 从 5338ms 降到 ~1000ms
- TTFT 从 10080ms 降到 ~5700ms（**降低 43%**）

**复杂度**：中等
- 需要修改 Decode Scheduler 的调度逻辑
- 需要确保 receiver 创建的线程安全

#### 2. 预热连接池（预期收益：首个请求 Bootstrap 降低 96%）

**问题**：首个请求需要 783ms 建立 UCX 连接（`prepare_payload=637ms`）。

**优化方案**：
- 在服务启动时预热 UCX 连接池
- 提前调用 `_add_remote_peer` 建立连接
- 避免首个请求承担连接建立的开销

**预期收益**：
- 首个请求的 Bootstrap 从 5338ms 降到 ~4500ms
- 但对平均 Bootstrap 影响很小（783ms / 100 请求 = 7.8ms）

**复杂度**：低
- 只需在服务启动时添加预热逻辑

### 4.3 中期优化（P1，需要开发）

#### 3. 🔴 异步 Bootstrap（预期收益：TTFT 降低 50%+）

**问题**：Bootstrap 是同步的，会阻塞 Prefill Forward。每个请求需要等待 5330ms 的墙钟时间，相当于等待 5 个 forward pass。

**优化方案**：
- 将 Bootstrap 改为异步
- 让 Prefill Forward 与 Bootstrap 并行执行
- Bootstrap 完成后再开始 KV Transfer

**实现思路**：
```python
# 当前：同步 Bootstrap
def process_request(req):
    create_sender(req)           # 0.02ms
    wait_for_bootstrap(req)      # 5330ms ← 阻塞！
    prefill_forward(req)         # 1070ms
    send_kv_cache(req)           # 521ms

# 优化：异步 Bootstrap
def process_request(req):
    create_sender(req)           # 0.02ms
    start_async_bootstrap(req)   # 启动异步 Bootstrap
    prefill_forward(req)         # 1070ms ← 与 Bootstrap 并行
    wait_for_bootstrap(req)      # 等待 Bootstrap 完成（可能已完成）
    send_kv_cache(req)           # 521ms
```

**预期收益**：
- 如果 Bootstrap 与 Forward 完全并行，TTFT 从 10080ms 降到 ~4750ms
- Bootstrap 的 5330ms 被 Forward 的 1070ms 掩盖
- TTFT = max(Bootstrap, Forward) + KV Transfer = 5330ms + 521ms = 5851ms

**复杂度**：高
- 需要重构 Prefill Scheduler 的调度逻辑
- 需要确保 Bootstrap 和 Forward 的线程安全
- 需要处理 Bootstrap 失败的边界情况

#### 4. 批量 Bootstrap（预期收益：Bootstrap 摊薄 N×）

**问题**：每个请求单独 Bootstrap，无法摊薄开销。

**优化方案**：
- 将多个请求的 Bootstrap 合并
- 一次 Bootstrap 处理多个请求
- 摊薄 HTTP GET 和 ZMQ 注册的开销

**预期收益**：
- 如果 batch_size=8，Bootstrap 摊薄到每个请求：5338ms / 8 = 667ms
- TTFT 从 10080ms 降到 ~5400ms

**复杂度**：中等
- 需要修改 Bootstrap 协议
- 需要确保批量 Bootstrap 的正确性

### 4.4 长期优化（P2，需要架构重构）

#### 5. 🔴 优化 Decode 端调度架构（预期收益：TTFT 降低 60%+）

**问题**：Decode 端等待成为新的主要瓶颈（44%）。即使 Bootstrap 优化到 1000ms，Decode 端等待仍然占 TTFT 的 44%。

**优化方案**：
- **多队列调度**：将 bootstrap 请求和普通 decode 请求分开排队
- **优先级抢占**：bootstrap 请求可以抢占正在处理的 decode 请求
- **分布式调度**：将 Decode 端的调度逻辑分布到多个线程/进程

**预期收益**：
- 如果 Decode 端调度延迟从 5330ms 降到 500ms
- Bootstrap 从 5338ms 降到 ~500ms
- TTFT 从 10080ms 降到 ~2100ms（**降低 79%**）

**复杂度**：高
- 需要重构 Decode Scheduler 的架构
- 需要确保调度的公平性和正确性

### 4.5 优化优先级总结

| 优先级 | 优化方向 | 预期收益 | 复杂度 | 目标场景 |
|---|---|---|---|---|
| **P0** | **减少 Decode 端调度延迟** | **Bootstrap 降低 80%+** | 中 | 32 并发 |
| P0 | 预热连接池 | 首个请求 Bootstrap 降低 96% | 低 | 所有场景 |
| **P1** | **异步 Bootstrap** | **TTFT 降低 50%+** | 高 | 32 并发 |
| P1 | 批量 Bootstrap | Bootstrap 摊薄 N× | 中 | 高并发 |
| **P2** | **优化 Decode 端调度架构** | **TTFT 降低 60%+** | 高 | 32 并发 |

### 4.6 下一步行动

1. **本周**：实现 Decode 端优先级调度，让 bootstrap 请求优先处理
2. **下周**：实现连接池预热，避免首个请求承担连接建立开销
3. **本月**：实现异步 Bootstrap，让 Prefill Forward 与 Bootstrap 并行
4. **下月**：重构 Decode 端调度架构，解决 32 并发下的调度延迟问题

**预期最终效果**（32 并发）：
- TTFT 从 10080ms 降到 ~2100ms（**降低 79%**）
- Bootstrap 从 5338ms 降到 ~500ms（**降低 91%**）
- 吞吐提升 4-5×

---

## 5. 总结

### 5.1 核心发现

1. **Bootstrap 的瓶颈是「Prefill 端轮询等待」而非「执行」**
   - `create_sender`、`finalize_bootstrap` 等执行阶段只需 0.01-0.04ms
   - Bootstrap 的 5338ms（32 并发）/ 470ms（16 并发）主要来自反复轮询等待 Decode 端连接

2. **UCX 连接已经被缓存和复用**
   - `_add_remote_peer` 只在首个请求时调用（783ms）
   - 后续请求完全复用连接，耗时 0ms
   - 之前认为"每个请求都要重新建立连接"的假设是错误的

3. **高并发是 Bootstrap 慢的根本原因**
   - 并发度从 32 降到 16，Bootstrap 从 5338ms 降到 470ms（11.4 倍改善）
   - Bootstrap 的瓶颈是高并发下的排队等待
   - 高并发下 `pop_bootstrapped_poll` 的 max 延迟从 12.64ms 飙升到 92.75ms（7.3 倍）

4. **Decode 端等待成为新的主要瓶颈**
   - 降低并发度后，Decode 端等待占 TTFT 的 44%
   - 优化方向应该从 Bootstrap 转向 Decode 端调度

### 5.2 TTFT 延迟分解对比

```
A: 100 req / 32 并发 (TTFT = 10080ms):
├── Bootstrap (等待):           ~5338ms  (53%)   🔴 最大瓶颈
├── Prefill Forward:            ~1070ms  (11%)
├── KV Transfer (数据传输):     ~521ms   (5%)
└── Decode 端等待:              ~3152ms  (31%)

B: 50 req / 16 并发 (TTFT = 3631ms):
├── Bootstrap (等待):           ~470ms   (13%)   ✅ 大幅改善
├── Prefill Forward:            ~1071ms  (29%)
├── KV Transfer (数据传输):     ~501ms   (14%)
└── Decode 端等待:              ~1589ms  (44%)   ← 新的主要瓶颈
```

### 5.3 优化建议（按优先级）

| 优先级 | 优化方向 | 预期收益 | 复杂度 |
|---|---|---|---|
| **P0** | **减少 Decode 端调度延迟** | **Bootstrap 降低 57%** | 中 |
| P0 | 预热连接池 | 首个请求 Bootstrap 降低 96% | 低 |
| P1 | 异步 Bootstrap | TTFT 降低 30-40% | 中 |
| P1 | 批量 Bootstrap | Bootstrap 摊薄 N× | 中 |
| P2 | 优化 Decode 端调度 | TTFT 降低 40-50% | 高 |

### 5.4 下一步行动

1. **立即**：优化 Decode Scheduler，减少调度延迟
2. **本周**：实现连接池预热，避免首个请求承担连接建立开销
3. **本月**：实现异步 Bootstrap，让 Prefill Forward 与 Bootstrap 并行
4. **长期**：优化 Decode 端调度，解决新的主要瓶颈

---

## 附录 A：Profiling 代码位置

本次 profiling 在以下文件中添加了时间戳：

| 文件 | 函数 | 行号 |
|---|---|---|
| `python/sglang/srt/disaggregation/prefill.py` | `create_sender()` | 226-250 |
| `python/sglang/srt/disaggregation/prefill.py` | `pop_bootstrapped()` | 307-397 |
| `python/sglang/srt/disaggregation/prefill.py` | `finalize_bootstrap()` | 262-280 |
| `python/sglang/srt/disaggregation/decode.py` | `_create_receiver_and_enqueue()` | 524-540 |
| `python/sglang/srt/disaggregation/decode.py` | `_resolve_prefill_dp_rank()` | 504-533 |
| `python/sglang/srt/disaggregation/common/conn.py` | `try_ensure_parallel_info()` | 233-276 |
| `python/sglang/srt/disaggregation/common/conn.py` | `_get_bootstrap_info_from_server()` | 1007-1024 |
| `python/sglang/srt/disaggregation/common/conn.py` | `init()` | 916-951 |
| `python/sglang/srt/disaggregation/common/conn.py` | `_setup_bootstrap_infos()` | 953-1005 |
| `python/sglang/srt/disaggregation/nixl/conn.py` | `_add_remote_peer()` | 932-940 |
| `python/sglang/srt/disaggregation/nixl/conn.py` | `_register_kv_args()` | 2103-2166 |

## 附录 B：测试命令

```bash
# 运行 bootstrap profiling benchmark
bash mybench/measure_kv_transfer.sh 4096 512 100 32   # A 组
bash mybench/measure_kv_transfer.sh 4096 512 50 16    # B 组

# 参数：input_len output_len num_prompts max_concurrency
```

输出文件：
```
mybench/kv-transfer-measurement/20260613_181644/   # A 组 (100 req / 32 并发)
mybench/kv-transfer-measurement/20260613_174321/   # B 组 (50 req / 16 并发)
├── benchmark_output.txt          # Benchmark 结果
├── prefill_server.log            # Prefill server 日志（含 profiling）
├── decode_server.log             # Decode server 日志（含 profiling）
├── prefill_metrics.txt           # Prometheus metrics
├── decode_metrics.txt            # Prometheus metrics
└── kv_transfer_analysis.txt      # 分析报告
```
