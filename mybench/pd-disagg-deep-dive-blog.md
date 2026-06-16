# Deep Dive: SGLang PD 分离在单机 PCIe 环境下的性能瓶颈探索

> **TL;DR**: 通过对 SGLang PD 分离模式的系统性 profiling，我发现单机 PCIe 环境下的性能瓶颈不是 KV cache 传输带宽，不是 UCX 连接建立，不是 decode 端调度，而是 **prefill 端的吞吐饱和导致的排队延迟**。这个结论看似简单，但到达这里的过程充满了错误的假设和被推翻的分析。

---

## 1. 背景：为什么要探索 PD 分离？

Prefill-Decode (PD) 分离是 LLM serving 领域的一个热门架构优化方向。核心思想是将 prefill（prefill 阶段，处理 prompt）和 decode（生成阶段，逐 token 生成）拆分到不同的 GPU 上运行，理论上可以：

- **独立扩展**：prefill 是 compute-bound，decode 是 memory-bound，可以用不同的硬件配置
- **资源利用率提升**：避免 prefill 和 decode 在同一 GPU 上竞争资源
- **延迟优化**：理论上可以降低 TTFT（Time To First Token）

SGLang 作为高性能 LLM serving 框架，已经实现了 PD 分离功能，支持 NIXL、Mooncake、Mori 等多种传输后端。

我的目标很简单：在单机 4 卡 PCIe 环境下（4× NVIDIA RTX PRO 4000 Blackwell, 无 NVLink），跑通 PD 分离，找到性能瓶颈，并提出优化方案。

**结果**：我花了两周时间，写了 10 篇分析文档，推翻了 3 个主要假设，最终发现——**在这个硬件环境下，PD 分离的瓶颈是物理限制，不是代码 bug**。

---

## 2. 方法论：如何系统性地定位瓶颈？

我的 profiling 方法论很简单，但执行起来充满了陷阱：

```
Step 1: 建立 baseline（TP4 vs PD TP2 对比）
Step 2: 宏观分析（Prometheus per-stage metrics）
Step 3: 微观分析（inline profiling，逐函数打点）
Step 4: 验证假设（代码阅读 + 实验验证）
Step 5: 推翻假设（发现矛盾，重新分析）
```

### 2.1 硬件环境

- **GPU**: 4× NVIDIA RTX PRO 4000 Blackwell (PCIe Gen4 x16, 无 NVLink)
- **CPU**: Intel Xeon (具体型号不重要)
- **网络**: 单机，无跨机传输
- **软件**: SGLang v0.4.x, PyTorch 2.5, CUDA 12.4

### 2.2 测试配置

- **模型**: qwen/qwen3.5-27b-fp8
- **TP4**: 单 server，4 卡 TP=4
- **PD TP2**: Prefill TP=2 (GPU 0-1) + Decode TP=2 (GPU 2-3) + Router
- **传输后端**: NIXL over UCX (PCIe transport)
- **Workload**: input=4096 tokens, output=512 tokens, concurrency=16/32

---

## 3. 第一阶段：宏观对比——PD 为什么比 TP4 慢？

### 3.1 初始发现：PD 在所有场景下都输

我跑了 30 组实验（6 个 input 长度 × 5 个 output 长度），结果很残酷：

```
Input=4096, Output=512, Concurrency=32:
  TP4 TTFT:  ~2000ms
  PD TTFT:   ~9600ms
  PD / TP4 = 4.8×  ← PD 慢了 4.8 倍

Output Throughput:
  TP4: 380 tok/s
  PD:  354 tok/s
  PD / TP4 = 0.93×  ← 吞吐也略低
```

**初步假设**：PD 慢是因为 KV cache 传输带宽不够。PCIe Gen4 x16 理论带宽 32 GB/s，但实际传输速度只有 0.14 GB/s，**利用率只有 0.5%**。

### 3.2 第一个错误假设：PCIe 带宽是瓶颈

我花了大量时间分析 PCIe 带宽利用率为什么这么低：

```
单请求 KV cache 大小: 108 MB
单请求传输速度: 0.14 GB/s
单请求传输时间: 108 MB / 0.14 GB/s = 770ms

聚合带宽（考虑并发）:
  平均并发度: 1.52
  聚合带宽: 0.14 × 1.52 = 0.21 GB/s
  PCIe 利用率: 0.21 / 32 = 0.67%
```

**错误结论**：PCIe 带宽利用率低是瓶颈，需要优化 UCX 配置、启用 RDMA、或者用 NVLink。

**为什么错了**：这个分析只看了"传输速度"，没看"传输在总延迟中的占比"。后来 Prometheus per-stage metrics 告诉我：

```
TTFT ≈ 9600ms
├── prefill_bootstrap:    4933ms  (51%)  ← 这是什么？
├── prefill_forward:      1061ms  (11%)  ← GPU 计算
├── prefill_transfer:      485ms  (5%)   ← 实际 PCIe 传输
└── decode_transferred:   9102ms         ← decode 端等待
```

**PCIe 传输只占 5%**。即使把传输时间从 485ms 优化到 0ms，TTFT 也只省 5%。PCIe 利用率低不是瓶颈，是**症状**。

**教训 #1**：不要优化占比小的部分。先看整体，再看局部。

---

## 4. 第二阶段：Bootstrap 是什么？为什么占 51%？

### 4.1 Prometheus 的启示：`prefill_bootstrap` 是最大头

从 Prometheus per-stage metrics 中，我发现 `prefill_bootstrap` 占了 TTFT 的 51%。但这是什么？

我读了 SGLang 的 PD 分离代码（`python/sglang/srt/disaggregation/prefill.py`），理解了 bootstrap 的流程：

```
1. Prefill 端创建 KVSender，进入 bootstrap queue
2. 反复 poll sender 状态，等待 decode 端连接
3. Decode 端创建 KVReceiver，发送 metadata 到 prefill
4. Prefill 端收到 metadata，finalize bootstrap
5. 请求进入 waiting queue，等待 prefill forward
```

**第二个假设**：Bootstrap 慢是因为 UCX 连接建立开销大。每个请求都要重新建立 UCX 连接，注册 GPU 显存，耗时 ~780ms。

### 4.2 第二个错误假设：UCX 连接没有复用

我写了 inline profiling 代码，在 `prefill.py`、`decode.py`、`nixl/conn.py` 等 11 个文件中加了时间戳：

```python
# prefill.py: create_sender
t0 = time.perf_counter()
req.disagg_kv_sender = kv_sender_class(...)
logger.info(f"[BOOTSTRAP PROFILE] create_sender: {(time.perf_counter()-t0)*1000:.2f}ms")
```

跑完 benchmark，分析日志，发现了矛盾的数据：

```
首个请求:
  _add_remote_peer: 783ms  ← 确实很慢
    ├─ add_remote_agent: 148ms
    └─ prepare_payload: 637ms  ← GPU 显存注册

后续请求:
  _add_remote_peer: 0ms  ← 直接跳过！
  create_sender: 0.02ms
  finalize_bootstrap: 0.01ms
```

**UCX 连接已经被缓存了！** `_add_remote_peer` 只在首个请求时调用，后续请求直接复用。

那 bootstrap 的 4933ms 花在哪了？

```
单次 poll 耗时: 0.42ms（16 和 32 并发一样）
poll 次数: ~12000 次
总 poll 时间: 0.42ms × 12000 = 5040ms ≈ 4933ms
```

**Bootstrap 的耗时来自反复 poll 等待，而不是任何单步操作。**

**教训 #2**：不要假设"每个请求都重新建立连接"。先读代码，看缓存逻辑。

### 4.3 第三个假设：Decode 端调度延迟导致 Prefill 等待

既然 bootstrap 是在等 decode 端，那 decode 端一定很慢。我分析了 decode 端的代码流程：

```
1. Decode 端收到请求，创建 KVReceiver
2. Receiver 调用 init()，发送 metadata 到 prefill
3. Prefill 端收到 metadata，bootstrap 完成
```

**假设**：Decode 端的 scheduler loop 太忙，导致 `receiver.init()` 被延迟，prefill 端需要反复 poll 等待。

为了验证这个假设，我在 decode 端的 scheduler loop 中加了 profiling：

```python
# decode.py: event_loop_overlap_disagg_decode
while True:
    t0 = time.perf_counter()
    recv_reqs = self.request_receiver.recv_requests()
    t1 = time.perf_counter()
    self.process_input_requests(recv_reqs)
    t2 = time.perf_counter()
    self.process_decode_queue()
    t3 = time.perf_counter()
    batch = self.get_next_disagg_decode_batch_to_run()
    t4 = time.perf_counter()
    if batch:
        result = self.run_batch(batch)
        t5 = time.perf_counter()
        self.process_batch_result(batch, result)
        t6 = time.perf_counter()
        logger.info(f"[DECODE PROFILE] recv: {(t1-t0)*1000:.2f}ms, "
                    f"process_input: {(t2-t1)*1000:.2f}ms, "
                    f"run_batch: {(t5-t4)*1000:.2f}ms, "
                    f"process_batch_result: {(t6-t5)*1000:.2f}ms")
```

结果：

```
Decode 端 per-iteration P99 延迟:
  process_batch_result: 30.38ms  (93.6%)  ← 看起来是瓶颈！
  run_batch:             0.97ms  (3.0%)
  process_decode_queue:  0.53ms  (1.6%)
  recv:                  0.33ms  (1.0%)
  total:                32.44ms
```

**错误结论**：Decode 端的 `process_batch_result` 占 93.6%，是瓶颈。它包括 ZMQ 通信、tokenizer 解码、HTTP response 等，需要优化。

**为什么错了**：我只看了 **per-iteration** 的延迟，没看 **整体利用率**。后来我统计了 decode 端的 iteration 分布：

```
16 并发:
  总迭代数: 163,328
  空闲迭代 (≤5ms): 160,574 (98.3%)
  忙碌迭代 (>5ms):   2,754 (1.7%)

32 并发:
  总迭代数: 209,725
  空闲迭代 (≤5ms): 205,555 (98.0%)
  忙碌迭代 (>5ms):   4,170 (2.0%)
```

**Decode 端 98% 的时间都是空闲的！** 它只是偶尔（1.7% 的迭代）处理一个 batch 的 `process_batch_result`（30ms），其余时间都在等待。

而且，decode 端的 `decode_bootstrap` metric 只有 1.3ms，说明 decode 创建 receiver + 发送 metadata 非常快。

**教训 #3**：不要只看 per-iteration 延迟，要看整体利用率。一个 98% 空闲的系统不是瓶颈。

---

## 5. 第三阶段：真正的瓶颈——Prefill 吞吐饱和

### 5.1 重新审视数据：16 vs 32 并发的对比

我把 16 并发和 32 并发的 Prometheus per-stage 数据放在一起对比：

| 阶段 | 16 并发 (avg) | 32 并发 (avg) | 倍数 |
|------|:---:|:---:|:---:|
| `prefill_bootstrap` | 417ms | **4,933ms** | **11.8×** |
| `prefill_forward` | 998ms | 1,061ms | 1.06× |
| `prefill_transfer` | 432ms | 485ms | 1.12× |
| `decode_bootstrap` | 1.7ms | 1.3ms | 0.8× |
| `decode_transferred` | 3,163ms | **9,102ms** | 2.88× |

**关键发现**：
- `prefill_forward` 和 `prefill_transfer` 几乎不受并发度影响（1.06× 和 1.12×）
- `prefill_bootstrap` 从 417ms 爆炸到 4,933ms（11.8×）
- `decode_transferred` 也爆炸了（2.88×），但那是在等 prefill 完成

**Forward 和 transfer 是稳定的，只有 bootstrap 在爆炸。**

### 5.2 Bootstrap 时间分布：从"几乎无等待"到"严重排队"

我分析了 Prometheus histogram 数据，画出了 bootstrap 时间的分布：

```
16 并发 (n=54):
  ≤ 1ms:    28 (51.9%)   ← 超过一半的请求 bootstrap < 1ms
  ≤ 3ms:    36 (66.7%)
  ≤ 3.6s:   54 (100%)
  平均: 417ms

32 并发 (n=103):
  ≤ 1ms:     2 ( 1.9%)   ← 几乎没有请求能快速 bootstrap
  ≤ 3ms:    10 ( 9.7%)
  ≤ 5.9s:   68 (66.0%)   ← 大量请求堆积在 3-6s
  ≤ 9.6s:   99 (96.1%)
  平均: 4,933ms
```

**16 并发时，一半请求 bootstrap < 1ms——prefill 几乎不排队。32 并发时，66% 的请求堆积在 3-6s——prefill 严重排队。**

### 5.3 根因分析：Little's Law 完美解释

Prefill 调度器的吞吐是多少？

```
每 batch 处理请求数: ~8 req
每 batch forward 时间: ~1070ms
Prefill 吞吐: 8 req / 1.07s = 7.5 req/s
```

用 Little's Law 估算排队时间：

```
32 并发:
  平均排队位置 ≈ 32/2 = 16
  平均排队时间 ≈ 16 / 7.5 = 2.13s
  实际 bootstrap avg = 4.9s（偏高，因为 batch 调度不均匀）

16 并发:
  平均排队位置 ≈ 16/2 = 8
  平均排队时间 ≈ 8 / 7.5 = 1.07s
  实际 bootstrap avg = 0.42s（偏低，因为 16 并发下 batch 更小更快）
```

**数量级匹配。** 32 并发时，请求需要排 3-4 个 batch cycle；16 并发时，只排 1-2 个 cycle。

### 5.4 验证：Prefill bootstrap queue 很小

如果 bootstrap 是瓶颈，那 bootstrap queue 应该很大。但实际数据：

```
16 并发:
  Bootstrap queue 平均大小: 1.1
  Bootstrap queue 最大: 9

32 并发:
  Bootstrap queue 平均大小: 3.3
  Bootstrap queue 最大: 24
```

**Bootstrap queue 平均只有 3 个请求。** 请求不是在 bootstrap 阶段堆积，而是在 **waiting queue**（等 PrefillAdder 调度进 batch）。

### 5.5 最终结论

```
Prefill 吞吐饱和（~7.5 req/s）
  ↓
32 并发时，请求在 prefill waiting queue 中排队 3-4 个 batch cycle
  ↓
Bootstrap 时间从 417ms（16 并发）爆炸到 4,933ms（32 并发）
  ↓
TTFT 从 3.6s 增长到 9.6s
```

**这不是代码 bug，是物理限制。** Prefill 调度器每秒只能处理 ~7.5 个请求，当并发度超过这个吞吐时，TTFT 就线性增长。这是排队论的基本结论。

---

## 6. 被排除的优化方向

在到达最终结论之前，我考虑过多个优化方向，但都被数据排除了：

### 6.1 ❌ 优化 PCIe 带宽利用率

**初始想法**：PCIe 利用率只有 0.5%，优化 UCX 配置、启用 RDMA、或者用 NVLink。

**为什么排除**：PCIe 传输只占 TTFT 的 5%。即使优化到 0ms，也只省 5%。

### 6.2 ❌ FP8 KV Cache

**初始想法**：FP8 KV cache 可以减少 50% 的数据量，传输时间减半。

**为什么排除**：传输只占 5%，减半也只省 2.5%。而且 benchmark 数据显示 FP8 和 FP16 的 TTFT 几乎一样。

### 6.3 ❌ UCX 连接复用

**初始想法**：每个请求都重新建立 UCX 连接，耗时 ~780ms。

**为什么排除**：代码分析发现 UCX 连接**已经被缓存**，只在首个请求时调用 `_add_remote_peer`（783ms），后续请求 0ms。100 个请求平均开销 = 7.8ms，占比 0.1%。

### 6.4 ❌ 异步 Bootstrap

**初始想法**：Bootstrap 是同步的，阻塞 prefill forward。改为异步，让 forward 和 bootstrap 并行。

**为什么排除**：代码分析发现这个功能**已经实现**，就是 `--optimistic-prefill-retries` 参数（默认 0）。只需要启用，不需要重新实现。

### 6.5 ❌ 优化 Decode 端调度

**初始想法**：Decode 端 `process_batch_result` 占 93.6%，需要优化 ZMQ 通信、tokenizer 解码等。

**为什么排除**：Decode 端 98% 的时间是空闲的。`decode_bootstrap` 只有 1.3ms，说明 decode 处理 bootstrap 非常快。Decode 不是瓶颈。

### 6.6 ❌ 批量 Bootstrap

**初始想法**：每个请求单独 bootstrap，无法摊薄开销。批量处理多个请求。

**为什么排除**：Bootstrap 是 per-request 的（每个请求有不同的 `bootstrap_room`），协议层面不支持批量。而且 bootstrap 本身很快（0.42ms/poll），慢的是等待 prefill 调度。

---

## 7. 可行的优化方向

虽然大部分优化方向被排除了，但仍有几个值得尝试：

### 7.1 ✅ 启用 Optimistic Prefill（P0）

**配置**：`--optimistic-prefill-retries 3`

**原理**：允许 bootstrap 未完成时就开始 forward。Forward 的 1070ms 与 bootstrap 等待并行，减少总时间。

**预期收益**：对 bootstrap < 1070ms 的请求无收益；对 bootstrap = 4933ms 的请求可省 ~1070ms（forward 被 bootstrap 掩盖）。

**复杂度**：低（只需配置参数）。

### 7.2 ✅ 增大 Prefill Batch Size（P0）

**配置**：`--max-running-requests 16 --max-prefill-tokens 32768`

**原理**：当前每 batch ~8 req。如果增大到 ~16 req/batch，吞吐可能从 7.5 req/s → ~12-15 req/s。

**预期收益**：32 并发排队从 4 轮 → 2-3 轮，TTFT 可能从 9.6s → ~5-7s。

**风险**：GPU 算力是固定的。更大的 batch 意味着单个 forward 时间会更长（可能从 1070ms → ~1800-2200ms）。净收益取决于 GPU 的 compute/memory 余量，必须实测。

### 7.3 ✅ 启用 Staging Buffer（P1）

**配置**：`export SGLANG_DISAGG_STAGING_BUFFER=1`

**原理**：GPU 上先 gather KV cache 再一次性发送，减少 RDMA 传输次数。

**预期收益**：KV transfer 从 485ms 降到 ~350ms。但占总 TTFT 的比例小（5%），对 TTFT 影响有限。

**复杂度**：低（只需配置环境变量）。

### 7.4 ✅ 多 Prefill Worker（P2，架构改动）

**原理**：直接翻倍 prefill 吞吐。

```
当前:  1 prefill (TP2) → 7.5 req/s
优化:  2 prefill (each TP1 or TP2) → 15 req/s
```

**预期收益**：吞吐翻倍，TTFT 大幅降低。

**复杂度**：高。需要 router 支持多 prefill 负载均衡，需要处理 KV cache 分配。

---

## 8. 技术收获与反思

### 8.1 方法论层面的收获

**1. 先看整体，再看局部**

我一开始就陷入"PCIe 利用率只有 0.5%"的局部优化思维，花了很多时间分析 UCX 配置、RDMA、NVLink。但 Prometheus per-stage metrics 告诉我，PCIe 传输只占 5%。**不要优化占比小的部分**。

**2. 不要只看 per-iteration 延迟，要看整体利用率**

Decode 端的 `process_batch_result` 占 93.6%（30ms），看起来是瓶颈。但 decode 端 98% 的时间是空闲的。一个偶尔忙 30ms、但 98% 时间空闲的系统不是瓶颈。**类比：一个餐厅服务员每桌服务需要 30 分钟，但他 98% 的时间都在闲着——问题不在服务员，而在厨房做菜太慢。**

**3. 先读代码，再做假设**

我假设"每个请求都重新建立 UCX 连接"，但代码分析发现连接已经被缓存。我假设"异步 bootstrap 需要重新实现"，但代码分析发现这个功能已经存在（`--optimistic-prefill-retries`）。**不要假设，先读代码**。

**4. 用数据推翻假设，而不是强行解释**

当我发现 decode 端 98% 空闲时，我一开始试图构建"decode 慢 → 间接影响 prefill"的因果链。但 `decode_bootstrap` = 1.3ms 的数据直接推翻了这个假设。**不要强行解释矛盾的数据，接受它，重新分析**。

### 8.2 技术层面的收获

**1. PD 分离的适用场景**

PD 分离的设计初衷是：
- Prefill 和 decode 用**不同的机器/GPU 集群**
- 通过**高速网络**（RDMA/InfiniBand）传输 KV cache
- 各自独立扩展

单机 PCIe 环境：
- 4 张卡通过 PCIe 连接，带宽只有 32 GB/s（vs InfiniBand 200-400 Gb/s）
- 单机 4 卡，没有跨机传输
- 还不如直接 TP4（所有计算在同一组 GPU 上，不需要传输 KV）

**结论：PD 分离在单机 PCIe 环境下天然受限，不适合作为优化目标。**

**2. 排队论在系统性能分析中的应用**

Little's Law: `L = λW`（平均队列长度 = 到达速率 × 平均等待时间）

Prefill 吞吐 ~7.5 req/s，32 并发时平均排队位置 ~16，平均等待时间 ~16/7.5 = 2.13s。理论值和实际值（4.9s）数量级匹配。**排队论是分析系统性能的强大工具**。

**3. Profiling 的分层方法**

```
Layer 1: Prometheus per-stage metrics（宏观）
Layer 2: Inline profiling（微观）
Layer 3: 代码分析（理解机制）
Layer 4: 实验验证（推翻假设）
```

每一层都有局限性。Prometheus 看不到 per-iteration 的细节，inline profiling 看不到整体利用率，代码分析看不到运行时行为。**需要多层结合，才能得到完整的图景**。

### 8.3 工程层面的收获

**1. 不要过度优化**

我一开始想做很多优化（UCX 配置、FP8 KV、异步 bootstrap、批量 bootstrap、decode 调度优化），但数据告诉我，大部分优化方向都是无效的。**不要过度优化，先找到真正的瓶颈**。

**2. 参数调优 vs 代码改动**

我发现的优化方向大部分是参数调优（`--optimistic-prefill-retries`、`--max-running-requests`、`SGLANG_DISAGG_STAGING_BUFFER`），而不是代码改动。这说明 SGLang 的设计已经很成熟，常见的优化场景都已经考虑到了。**作为用户，先尝试参数调优；作为开发者，再考虑代码改动**。

**3. 写文档的重要性**

我写了 10 篇分析文档，记录了每一次假设、验证、推翻的过程。这些文档不仅帮助我理清思路，也成为了这篇博客的素材。**好记性不如烂笔头，尤其是在复杂的性能分析中**。

---

## 9. 总结

### 9.1 核心发现

| 问题 | 答案 |
|------|------|
| PD 为什么比 TP4 慢？ | Prefill 吞吐饱和（~7.5 req/s），导致排队延迟 |
| Bootstrap 为什么占 51%？ | 请求在 prefill waiting queue 中排队，等待 forward |
| PCIe 利用率为什么只有 0.5%？ | Prefill 吞吐饱和，同时做 transfer 的请求少，聚合带宽低 |
| Decode 端有问题吗？ | 没有。98% 空闲，bootstrap 只需 1.3ms |
| UCX 连接有复用吗？ | 有。只在首个请求时建立（783ms），后续复用（0ms） |
| FP8 KV cache 有效吗？ | 无效。传输只占 5%，减半也只省 2.5% |

### 9.2 优化建议

| 优先级 | 方向 | 类型 | 预期收益 |
|--------|------|------|----------|
| **P0** | 增大 prefill batch size | 参数调优 | TTFT 降低 30-50% |
| **P0** | 启用 optimistic prefill | 参数调优 | TTFT 降低 10-20% |
| P1 | 启用 staging buffer | 环境变量 | KV transfer 降低 20-30% |
| P2 | 多 prefill worker | 架构改动 | 吞吐翻倍 |

### 9.3 最终结论

> **在单机 PCIe 环境下，PD 分离的瓶颈是 prefill 吞吐饱和导致的排队延迟。这是物理限制，不是代码 bug。要打破这个限制，只有两条路：增大 batch size（参数调优）或增加 prefill 并行度（架构改动）。**

---

## 附录 A：分析文档索引

本次探索产生了 10 篇分析文档，按时间顺序：

1. `0-comparison-tp4-vs-pdtp2.md` — TP4 vs PD TP2 对比（初步发现 PD 慢）
2. `5-kv-transfer-bottleneck-analysis.md` — KV Transfer 瓶颈分析（错误假设：PCIe 带宽是瓶颈）
3. `4-kv-transfer-connection-reuse-analysis.md` — 连接复用分析（推翻错误假设：UCX 连接已被缓存）
4. `7-bootstrap-profiling-results.md` — Bootstrap Profiling 结果（发现 99.8% 时间在等待）
5. `8-bootstrap-opt-plan.md` — Bootstrap 优化方案（审查各种优化方向）
6. `9-pd-prefill.md` — PD Prefill 数据流分析（代码层面的理解）
7. `10-decode-results.md` — Decode 调度器 Profiling（错误假设：decode 端是瓶颈）
8. `11-prefill-throughput-bottleneck.md` — Prefill 吞吐饱和（最终正确答案）

---

## 附录 B：Profiling 代码示例

### B.1 Bootstrap Inline Profiling

```python
# python/sglang/srt/disaggregation/prefill.py

def create_sender(self, req: Req, num_kv_heads: int) -> bool:
    t0 = time.perf_counter()
    # ... 原有逻辑 ...
    logger.info(f"[BOOTSTRAP PROFILE] create_sender: {(time.perf_counter()-t0)*1000:.2f}ms, rid={req.rid}")
    return True

def pop_bootstrapped(self, ...):
    t0 = time.perf_counter()
    polls = poll_and_all_reduce_attn_cp_tp_group(...)
    t_poll = time.perf_counter()
    logger.info(f"[BOOTSTRAP PROFILE] pop_bootstrapped_poll: {(t_poll-t0)*1000:.2f}ms, queue_size={len(self.queue)}")
    # ... 原有逻辑 ...
    t_end = time.perf_counter()
    logger.info(f"[BOOTSTRAP PROFILE] pop_bootstrapped_total: {(t_end-t0)*1000:.2f}ms, bootstrapped={len(bootstrapped_reqs)}")
```

### B.2 Decode Scheduler Profiling

```python
# python/sglang/srt/disaggregation/decode.py

def event_loop_overlap_disagg_decode(self: Scheduler):
    while True:
        t0 = time.perf_counter()
        recv_reqs = self.request_receiver.recv_requests()
        t1 = time.perf_counter()
        self.process_input_requests(recv_reqs)
        t2 = time.perf_counter()
        self.process_decode_queue()
        t3 = time.perf_counter()
        batch = self.get_next_disagg_decode_batch_to_run()
        t4 = time.perf_counter()
        if batch:
            result = self.run_batch(batch)
            t5 = time.perf_counter()
            self.process_batch_result(batch, result)
            t6 = time.perf_counter()
            logger.info(
                f"[DECODE PROFILE] "
                f"recv: {(t1-t0)*1000:.2f}ms, "
                f"process_input: {(t2-t1)*1000:.2f}ms, "
                f"process_decode_queue: {(t3-t2)*1000:.2f}ms, "
                f"run_batch: {(t5-t4)*1000:.2f}ms, "
                f"process_batch_result: {(t6-t5)*1000:.2f}ms, "
                f"total: {(t6-t0)*1000:.2f}ms"
            )
```

---

**感谢阅读。** 如果你也在做 LLM serving 的性能优化，希望这篇博客能帮你少走一些弯路。记住：**先看整体，再看局部；不要假设，先读代码；用数据推翻假设，而不是强行解释**。
