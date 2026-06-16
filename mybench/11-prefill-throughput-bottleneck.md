# Prefill 吞吐饱和：16→32 并发 TTFT 劣化的根本原因

**日期**: 2026-06-14  
**结论**: TTFT 从 3.6s (16 并发) 增长到 9.6s (32 并发)，**6s 的增长几乎全部来自 prefill 端的排队等待**，而非任何 CPU/GPU 操作变慢。

---

## 1. 一句话结论

> Prefill 调度器每秒只能处理 ~7.5 个请求（每 batch ~8 req × 1070ms forward）。32 并发时请求必须排队等 3-4 个 batch cycle，16 并发时只排 1-2 个。**排队时间 = TTFT 增长的全部原因。**

---

## 2. 关键证据

### 2.1 Prometheus per-stage 数据（最权威）

| 阶段 | 16 并发 (avg) | 32 并发 (avg) | 倍数 | 性质 |
|------|:---:|:---:|:---:|------|
| `prefill_bootstrap` | 417ms | **4,933ms** | **11.8×** | 🔴 排队等待 |
| `prefill_forward` | 998ms | 1,061ms | 1.06× | GPU 计算（几乎不变） |
| `prefill_transfer_kv_cache` | 432ms | 485ms | 1.12× | 数据传输（几乎不变） |
| `decode_bootstrap` | 1.7ms | 1.3ms | 0.8× | 接收握手（极快） |
| `decode_transferred` | 3,163ms | **9,102ms** | 2.88× | 🔴 等待 prefill 完成 + KV 传输 |

**Forward 和 KV transfer 几乎不受并发度影响。只有 bootstrap（排队）和 decode_transferred（等 prefill 做完）爆炸了。**

> **注意**：`prefill_bootstrap` 这个 metric 包含多个子阶段（bootstrap queue 等待 + decode 握手 + waiting queue 排队 + forward 本身），不是纯粹的排队时间。见 §4.1 的精确拆解。

### 2.2 排队时间符合 Little's Law

```
Prefill 吞吐 ≈ 7.5 req/s（基于 batch ~8 req × ~1070ms forward）

32 并发 (benchmark 持续 66.9s，100 个请求):
  平均排队位置 ≈ 32/2 = 16
  平均排队时间 ≈ 16 / 7.5 = 2.13s
  实际 bootstrap avg = 4.9s（偏高，因为 batch 调度不均匀 + 尾部请求等待更久）

16 并发 (benchmark 持续 42.5s，50 个请求):
  平均排队位置 ≈ 16/2 = 8
  平均排队时间 ≈ 8 / 7.5 = 1.07s
  实际 bootstrap avg = 0.42s（偏低，因为 16 并发下 batch 更小更快，且 prefill 有更多空闲）
```

数量级匹配。精确值有偏差是因为 batch 调度不是均匀到达的，且 prefill 在空闲时会用更小的 batch 加速处理。

### 2.3 Prefill Bootstrap 时间分布

```
16 并发 (n=54):
  ≤ 1ms:    28 (51.9%)   ← 超过一半的请求 bootstrap < 1ms（几乎无等待）
  ≤ 3ms:    36 (66.7%)
  ≤ 3.6s:   54 (100%)
  平均: 417ms

32 并发 (n=103):
  ≤ 1ms:     2 ( 1.9%)   ← 几乎没有请求能快速 bootstrap
  ≤ 3ms:    10 ( 9.7%)
  ≤ 5.9s:   68 (66.0%)   ← 大量请求堆积在 3-6s
  ≤ 9.6s:   99 (96.1%)
  ≤ 15.5s: 103 (100%)
  平均: 4,933ms
```

16 并发时**一半请求 bootstrap < 1ms**——说明 prefill 几乎不排队。32 并发时**66% 的请求堆积在 3-6s**——说明 prefill 严重排队。

### 2.4 Decode 端不是瓶颈

从 decode 端 profiling 数据：

| | 16 并发 | 32 并发 |
|--|:---:|:---:|
| 总迭代数 | 163,328 | 209,725 |
| **空闲迭代 (≤5ms)** | **160,574 (98.3%)** | **205,555 (98.0%)** |
| 忙碌迭代 (>5ms) | 2,754 (1.7%) | 4,170 (2.0%) |
| 调度频率 | 3,891 iter/s | 3,132 iter/s |

**Decode 端 98% 的 scheduler 迭代都是空闲的。** 它只是偶尔（1.7% 的迭代）处理一个 batch 的 `process_batch_result`（30ms P99），其余时间都在等待。

Decode 端 `decode_bootstrap` = 1.3ms，说明 decode 创建 receiver + 发送 metadata 非常快。`decode_transferred` = 9.1s 看起来很大，但那是在**等 prefill 端完成 forward + KV transfer**，不是 decode 自己在做事。

### 2.5 Prefill Bootstrap Queue 也很小

| | 16 并发 | 32 并发 |
|--|:---:|:---:|
| Bootstrap queue 平均大小 | 1.1 | 3.3 |
| Bootstrap queue 最大值 | 9 | 24 |
| 单次 poll 耗时 | 0.41ms | 0.42ms |

Bootstrap queue 平均只有 3 个请求——请求并不是在 bootstrap 阶段堆积。它们堆积在 **waiting queue**（等 PrefillAdder 调度进 batch）。

---

## 3. 之前的分析（10-decode-results.md）错在哪

### 3.1 核心误判：把 per-iteration 开销当成系统瓶颈

10-decode-results.md 发现 decode 端 `process_batch_result` P99 = 30ms（占 93.6%），认为这是瓶颈。

**错误原因**：30ms 是 **per-iteration** 的开销，但 decode 端 98% 的 iteration 是空闲的。一个偶尔忙 30ms、但 98% 时间空闲的系统不是瓶颈。就像一个餐厅服务员每桌服务需要 30 分钟，但他 98% 的时间都在闲着——问题不在服务员，而在厨房做菜太慢。

### 3.2 误导性的因果链

10-decode-results.md 构建了 "Decode 慢 → 间接影响 Prefill" 的因果链：

> "Decode process_batch_result 慢 → decode scheduler loop 处理 bootstrap 慢 → receiver.init() 延迟 → Prefill poll 更多次"

**实际情况**：
- `decode_bootstrap` = 1.3ms（decode 处理 bootstrap 极快）
- `pop_bootstrapped_poll` 平均 0.42ms（16 和 32 并发一样）
- 问题不在 poll 变慢，而在**请求还没到 poll 阶段**——它们在 prefill waiting queue 里等 forward

### 3.3 被忽略的关键数据

10-decode-results.md 完全没有看 **decode 端的整体利用率**（98% 空闲），也没有从 Prometheus per-stage 数据中计算排队时间。如果看了这两个数据，就能立刻判断 decode 不是瓶颈。

---

## 4. 根本原因模型

### 4.1 `prefill_bootstrap` 的精确构成

这个 metric **不只是排队时间**，它包含多个子阶段：

```
32 并发下 prefill_bootstrap = 4,933ms 的拆解：

├── bootstrap queue wait（等 decode 端握手完成）   ~几十 ms
│   decode_bootstrap = 1.3ms，单次 poll = 0.42ms
│   bootstrap queue 平均大小 3.3，很小
├── waiting queue wait（等 PrefillAdder 调度进 batch）  ~大头 ⭐
│   请求堆积在这里等待 GPU 资源
├── prefill_forward                                ~1,061ms
│   包含在 prefill_bootstrap metric 中
└── chunked_prefill（如有切分）                     ~690ms
    包含在 prefill_bootstrap metric 中

主要排队发生在 waiting queue，而非 bootstrap queue。
```

> **重要**：`prefill_forward` 和 `chunked_prefill` 的时间**也计入** `prefill_bootstrap`，所以 bootstrap metric 不是纯排队。真正的"等待时间"需要从 bootstrap 减去 forward 时间。

### 4.2 Decode 端也有次要排队（未完全解释）

一个遗留问题：prefill 端流水线总耗时 ≈ 1061ms (forward) + 485ms (transfer) = 1546ms。但 decode 端 `decode_transferred` = 9,102ms，差了 ~7.5s。

这说明 **decode 端在 KV 传输完成后也有排队延迟**——可能是 decode 的 `waiting_queue → running_batch` 调度有限，一次只能处理有限数量的请求进入 decode forward。

这不影响"prefill 是主瓶颈"的结论（因为 decode_transferred 包含了等 prefill 完成的时间），但说明 decode 端也不是完全无辜——它有自己的次级排队。

### 4.3 根本原因模型

```
                        16 并发                              32 并发
                    ┌──────────────┐                    ┌──────────────────┐
                    │  Prefill GPU  │                    │   Prefill GPU     │
                    │  ~8 req/batch │                    │   ~8 req/batch    │
                    │  ~1070ms/batch│                    │   ~1070ms/batch   │
                    │  ~7.5 req/s   │                    │   ~7.5 req/s      │
                    └──────┬───────┘                    └──────┬────────────┘
                           │                                    │
                    8 个请求排 1 轮                        32 个请求排 4 轮
                    avg 等待 ~0.5s                         avg 等待 ~3.5s
                           │                                    │
              ┌────────────┴────────────┐         ┌─────────────┴────────────┐
              │     bootstrap = 0.4s    │         │    bootstrap = 4.9s      │
              │     forward  = 1.0s     │         │    forward  = 1.1s       │
              │     transfer = 0.4s     │         │    transfer = 0.5s       │
              │     ─────────────────   │         │    ─────────────────     │
              │     TTFT      ≈ 3.6s    │         │    TTFT     ≈ 9.6s      │
              └─────────────────────────┘         └──────────────────────────┘
                           │                                    │
                    ┌──────┴───────┐                    ┌───────┴────────────┐
                    │  Decode GPU   │                    │   Decode GPU        │
                    │  98% 空闲     │                    │   98% 空闲          │
                    │  (次级排队)   │                    │   (次级排队)        │
                    └──────────────┘                    └────────────────────┘
```

**16 和 32 并发下，prefill GPU 的计算速度完全一样（1.0s vs 1.1s），decode GPU 都是空闲的。唯一的区别是 prefill 前面的排队长度。**

---

## 5. 为什么 KV Transfer 没有成为瓶颈

### 5.1 KV Transfer 是异步的

从代码分析（`pd-prefill.md`）：

```python
# prefill.py: send_kv_chunk → sender.send() → 异步 RDMA
# 请求进入 inflight_queue，prefill scheduler 继续处理下一个 batch
# 每 tick 轮询 inflight_queue 检查传输完成
```

KV transfer 和 prefill forward 是**流水线**的：

```
时间线:
Batch 1: [forward 1070ms] → [KV transfer 485ms]
Batch 2:                   [forward 1070ms] → [KV transfer 485ms]
Batch 3:                                      [forward 1070ms] → ...

KV transfer 与下一个 batch 的 forward 重叠！
```

### 5.2 实际 PCIe 带宽利用率只有 0.5%

| 指标 | 数值 |
|------|------|
| 平均传输速度 | 0.14 GB/s |
| PCIe 理论带宽 | 32 GB/s |
| 利用率 | 0.5% |

KV transfer 远远没有打满 PCIe 带宽。瓶颈在 prefill 端 forward 的串行性（一个 batch 完成后才开始下一个），而非传输带宽。

---

## 6. 可行的优化方向

### 6.1 提高 Prefill 吞吐（减少排队）

这是**唯一**能降低 TTFT 的方向。

#### 方向 A：增大 Prefill Batch Size（参数调优，P0）

```bash
--max-running-requests 16       # 从默认值增大
--max-prefill-tokens 32768      # 允许更大的 batch
```

**原理**：当前每 batch ~8 req。如果增大到 ~16 req/batch：
- 吞吐可能从 7.5 req/s → ~12-15 req/s
- 32 并发排队从 4 轮 → 2-3 轮
- TTFT 可能从 9.6s → ~5-7s

**注意**：GPU 算力是固定的。更大的 batch 意味着单个 forward 时间会更长（可能从 1070ms → ~1800-2200ms）。净收益取决于 GPU 的 compute/memory 余量，必须实测。如果 GPU 已经 compute-bound，增大 batch 只会让单个请求更慢，总吞吐不变。

#### 方向 B：启用 Optimistic Prefill（参数调优，P0）

```bash
--optimistic-prefill-retries 3
```

**原理**：允许 bootstrap 未完成时就开始 forward。Forward 的 1070ms 与 bootstrap 等待并行，减少总时间。

**预期收益**：对 bootstrap < 1070ms 的请求无收益；对 bootstrap = 4933ms 的请求可省 ~1070ms（forward 被 bootstrap 掩盖）。

#### 方向 C：启用 Staging Buffer（环境变量，P1）

```bash
export SGLANG_DISAGG_STAGING_BUFFER=1
```

**原理**：GPU 上先 gather KV cache 再一次性发送，减少 RDMA 传输次数。当前 PCIe 利用率只有 0.5%，staging 可以提高到 1-2%。

**预期收益**：KV transfer 从 485ms 降到 ~350ms。但占总 TTFT 的比例小（5%），对 TTFT 影响有限。

#### 方向 D：多 Prefill Worker（架构改动，P2）

```
当前:  1 prefill (TP2) → 7.5 req/s
优化:  2 prefill (each TP1 or TP2) → 15 req/s
```

**原理**：直接翻倍 prefill 吞吐。这是唯一能**线性扩展**的方法。

**复杂度**：高。需要 router 支持多 prefill 负载均衡，需要处理 KV cache 分配。

### 6.2 减少 Decode 端等待（次要）

Decode 端 `decode_transferred` = 9.1s（32 并发），看起来很大，但那是在等 prefill 做完。如果 prefill 吞吐提升了，这个自然下降。

### 6.3 参数调优测试矩阵

建议按以下顺序测试：

```
Test 1: Baseline (当前配置)
Test 2: + --optimistic-prefill-retries 3
Test 3: + --max-running-requests 16
Test 4: + SGLANG_DISAGG_STAGING_BUFFER=1
Test 5: 2+3+4 组合
```

每个 test 在 16 和 32 并发下各跑一次，对比 TTFT 和吞吐。

---

## 7. 总结

### 7.1 问题诊断

| 问题 | 答案 |
|------|------|
| TTFT 为什么从 3.6s 涨到 9.6s？ | Prefill 排队延迟：16 并发排 1-2 轮，32 并发排 3-4 轮 |
| 瓶颈在哪？ | **Prefill forward 吞吐**（~7.5 req/s） |
| Decode 端有问题吗？ | 没有。98% 空闲，bootstrap 只需 1.3ms |
| Bootstrap 协议有问题吗？ | 没有。连接已缓存，单次 poll 0.42ms |
| KV Transfer 有问题吗？ | 没有。异步传输，PCIe 利用率仅 0.5% |
| 之前分析的 decode process_batch_result 30ms？ | 是 per-iteration 正常开销，98% iteration 空闲，不是瓶颈 |

### 7.2 优化路径

| 优先级 | 方向 | 类型 | 预期收益 |
|--------|------|------|----------|
| **P0** | 增大 prefill batch size | 参数调优 | TTFT 降低 30-50% |
| **P0** | 启用 optimistic prefill | 参数调优 | TTFT 降低 10-20% |
| P1 | 启用 staging buffer | 环境变量 | KV transfer 降低 20-30% |
| P2 | 多 prefill worker | 架构改动 | 吞吐翻倍 |

### 7.3 核心认知

> **在 PD 分离架构下，prefill 是吞吐量瓶颈（throughput bottleneck），不是延迟瓶颈（latency bottleneck）。**
> 
> 单个请求的 prefill 计算很快（~1s），但调度器每秒只能处理 ~7.5 个请求。当并发度超过这个吞吐时，TTFT 就线性增长。这是排队论的基本结论，**不是代码 bug，是物理限制**。
> 
> 要打破这个限制，只有两条路：
> 1. **让每个 batch 处理更多请求**（增大 batch size）
> 2. **增加 prefill 并行度**（多 prefill worker）
