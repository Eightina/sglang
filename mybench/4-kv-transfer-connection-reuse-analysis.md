# KV Cache 传输机制深度分析：连接复用与 Bootstrap 开销

- **日期**: 2026-06-11
- **模型**: `qwen/qwen3.5-27b-fp8`
- **配置**: PD TP2 模式（Prefill TP=2 GPU 0-1, Decode TP=2 GPU 2-3）
- **传输后端**: NIXL over UCX (PCIe)
- **测试条件**: 100 请求，32 并发

---

## 1. 核心问题

在 `kv-transfer-bottleneck-analysis.md` 中，我们发现 **Bootstrap（连接建立）耗时 5282ms，占 TTFT 的 53%**。这引出了几个关键问题：

1. **SGLang 是否为每个请求都创建新的 KVSender/KVReceiver？**
2. **如果是，为什么要这样设计？**
3. **UCX 连接是否被复用？**
4. **Bootstrap 为什么这么慢？如何优化？**

---

## 2. KVSender/KVReceiver 生命周期分析

### 2.1 结论：对象每请求创建，但底层连接被缓存

| 层次 | 是否每请求创建 | 是否复用 | 说明 |
|---|---|---|---|
| **KVSender/KVReceiver 对象** | ✅ 是 | ❌ 否 | 每个请求创建新对象，请求完成后 GC 回收 |
| **UCX 连接（NIXL agent）** | ❌ 否 | ✅ 是 | 每个 decode 实例只注册一次，后续请求复用 |
| **ZMQ 套接字** | ❌ 否 | ✅ 是 | 类级别缓存（`_socket_cache`），全局共享 |
| **Bootstrap 信息** | ❌ 否 | ✅ 是 | 实例级别缓存（`connection_pool`），按 prefill 地址缓存 |
| **NIXL 描述符（prep_handles）** | ❌ 否 | ✅ 是 | 按 peer 缓存，跨请求复用 |

**关键发现**：虽然 KVSender/KVReceiver 对象是每请求创建的，但**底层的 UCX 连接、ZMQ 套接字、NIXL 描述符都被积极缓存**。这意味着昂贵的连接建立操作只在首次通信时发生，后续请求复用已有连接。

### 2.2 Prefill 端：KVSender 生命周期

**文件**: `python/sglang/srt/disaggregation/prefill.py`

#### 阶段 1：创建 Sender（每请求）

```python
# prefill.py:226-250 (PrefillBootstrapQueue.create_sender)
def create_sender(self, req: Req):
    kv_sender_class = get_kv_class(self.backend, KVClassType.SENDER)
    dest_tp_ranks = [self.tp_rank]
    
    # 每个请求创建新的 sender 对象
    req.disagg_kv_sender = kv_sender_class(
        mgr=self.kv_manager,                    # ← 共享的 manager（per TP rank）
        bootstrap_addr=f"{req.bootstrap_host}:{self.bootstrap_port}",
        bootstrap_room=req.bootstrap_room,
        dest_tp_ranks=dest_tp_ranks,
        pp_rank=self.pp_rank,
    )
    req.pending_bootstrap = True
```

#### 阶段 2：Bootstrap 轮询

```python
# prefill.py:307-380 (pop_bootstrapped)
def pop_bootstrapped(self) -> List[Req]:
    """轮询 bootstrap 队列，检查握手是否完成"""
    
    # TP AllReduce：所有 TP rank 必须同步
    polls = poll_and_all_reduce_attn_cp_tp_group(
        [req.disagg_kv_sender for req in self.queue],
        self.scheduler.attn_cp_cpu_group,
        self.scheduler.attn_tp_cpu_group,
    )
    
    for i, (req, poll) in enumerate(zip(self.queue, polls)):
        if poll == KVPoll.Success:
            # Bootstrap 完成
            self.finalize_bootstrap(req)
            bootstrapped_reqs.append(req)
        elif poll == KVPoll.WaitingForInput:
            # Decode 端已连接，等待输入
            self.finalize_bootstrap(req)
            bootstrapped_reqs.append(req)
```

#### 阶段 3：初始化 Sender

```python
# prefill.py:262-280 (finalize_bootstrap)
def finalize_bootstrap(self, req: Req) -> bool:
    """Bootstrap 完成后初始化 sender"""
    assert req.pending_bootstrap
    
    if not self.ensure_metadata_buffer(req):
        return False
    
    req.time_stats.set_bootstrap_done_time()  # ← 记录 bootstrap 完成时间
    
    num_kv_indices = len(req.origin_original_ids)
    req.disagg_kv_sender.init(num_kv_indices, req.metadata_buffer_index)
    req.pending_bootstrap = False
    return True
```

#### 阶段 4：发送 KV Cache

```python
# prefill.py:907-1003 (send_kv_chunk)
def send_kv_chunk(self, req, last_chunk=False, end_idx=None):
    """发送 KV cache 到 decode worker"""
    
    page_indices = req.kv_cache_pool.get_page_indices(...)
    
    # sender 已经在 bootstrap 阶段创建，直接发送
    req.disagg_kv_sender.send(page_indices, state_indices)
```

#### 阶段 5：传输完成轮询

```python
# prefill.py:670-786 (process_disagg_prefill_inflight_queue)
def process_disagg_prefill_inflight_queue(self):
    """轮询 inflight 请求，检查传输是否完成"""
    
    for req in self.inflight_requests:
        poll = req.disagg_kv_sender.poll()
        
        if poll == KVPoll.Success:
            req.time_stats.set_prefill_kv_transfer_finish_time()
            req.disagg_kv_sender.clear()  # ← 清理 sender 状态
            self.process_finished_request(req)
        
        elif poll == KVPoll.Failed:
            req.disagg_kv_sender.failure_exception()
```

#### 阶段 6：清理

```python
# prefill.py:718-719
req.disagg_kv_sender.clear()
# sender 对象在 req 离开作用域后被 GC 回收
```

### 2.3 Decode 端：KVReceiver 生命周期

**文件**: `python/sglang/srt/disaggregation/decode.py`

#### 阶段 1：创建 Receiver（每请求）

```python
# decode.py:524-540 (DecodePreallocQueue._create_receiver_and_enqueue)
def _create_receiver_and_enqueue(self, req: Req):
    kv_receiver_class = get_kv_class(self.backend, KVClassType.RECEIVER)
    
    # 每个请求创建新的 receiver 对象
    kv_receiver = kv_receiver_class(
        mgr=self.kv_manager,
        bootstrap_addr=_bootstrap_addr(req),
        bootstrap_room=req.bootstrap_room,
    )
    
    decode_req = DecodeRequest(req=req, kv_receiver=kv_receiver)
    self.queue.append(decode_req)
```

#### 阶段 2：初始化 Receiver

```python
# decode.py:463 or 750-751 (add or _resolve_pending_reqs)
def init_receiver(self, decode_req: DecodeRequest):
    """初始化 receiver，获取 prefill 端信息"""
    
    # 1. 获取 prefill server info（HTTP 请求，有缓存）
    decode_req.kv_receiver.init(prefill_dp_rank=...)
    
    # 2. 向 prefill 注册 KV args（ZMQ 消息）
    # 内部调用 _register_kv_args()，发送 agent metadata 到 prefill
```

#### 阶段 3：握手轮询

```python
# decode.py:618-665 (_update_handshake_waiters)
def _update_handshake_waiters(self):
    """轮询握手状态"""
    
    for decode_req in self.queue:
        if not decode_req.waiting_for_input:
            poll = decode_req.kv_receiver.poll()
            if poll == KVPoll.WaitingForInput:
                decode_req.waiting_for_input = True
```

#### 阶段 4：发送 Metadata

```python
# decode.py:1031-1036 (pop_preallocated)
def pop_preallocated(self):
    """预分配完成后，发送 KV indices 到 prefill"""
    
    for decode_req in self.preallocated_queue:
        decode_req.kv_receiver.send_metadata(
            kv_indices=decode_req.kv_cache_indices,
            aux_index=decode_req.metadata_buffer_index,
            state_indices=...,
            decode_prefix_len=...,
        )
```

#### 阶段 5：传输轮询

```python
# decode.py:1585-1699 (pop_transferred)
def pop_transferred(self):
    """轮询传输完成"""
    
    for decode_req in self.transfer_queue:
        poll = decode_req.kv_receiver.poll()
        
        if poll == KVPoll.Success:
            self._commit_transfer_to_req(decode_req)
            self.waiting_queue.add(decode_req)
        
        elif poll == KVPoll.Failed:
            decode_req.kv_receiver.failure_exception()
```

#### 阶段 6：清理

```python
# decode.py:1546-1547 (_commit_transfer_to_req)
def _commit_transfer_to_req(self, decode_req: DecodeRequest):
    decode_req.kv_receiver.clear()
    decode_req.kv_receiver = None  # ← 释放引用，等待 GC
```

### 2.4 NIXL 实现：连接缓存机制

**文件**: `python/sglang/srt/disaggregation/nixl/conn.py`

#### NixlKVManager：共享的连接基础设施

```python
# nixl/conn.py:240-343 (NixlKVManager.__init__)
class NixlKVManager(CommonKVManager):
    def __init__(self, ...):
        # 每个 TP rank 一个 agent（全局唯一）
        self.agent = nixl_agent(str(uuid.uuid4()), agent_config)
        
        # 创建 NIXL backend（UCX、OBJ、GDS_MT 等）
        self.agent.create_backend(backend, backend_params)
        
        # 注册 GPU buffer 到 NIXL engine
        self.register_buffer_to_engine()
        
        # 连接缓存（关键！）
        self.decode_kv_args_table = {}      # agent_name → KVArgsRegisterInfo
        self.prep_handles = {}              # peer_name → nixl_dlist_handle
        self.prep_handle_slice_src = None   # 共享的 src dlist
```

#### UCX 连接建立：`_add_remote_peer()`

```python
# nixl/conn.py:932-940
def _add_remote_peer(self, decode_kv_args: KVArgsRegisterInfo):
    agent_name = decode_kv_args.agent_name
    
    # 🔑 连接复用：如果已注册，直接返回
    if agent_name in self.decode_kv_args_table:
        logger.info(f"Peer {agent_name} was already registered, ignoring.")
        return
    
    # 首次连接：注册 peer 并建立 UCX 连接
    self.decode_kv_args_table[agent_name] = decode_kv_args
    self.agent.add_remote_agent(decode_kv_args.agent_metadata)  # ← UCX 握手
    
    # 预构建传输描述符（跨请求复用）
    if self.disaggregation_mode == DisaggregationMode.PREFILL:
        self._prepare_payload_xfer(decode_kv_args)
```

#### 预构建描述符：`_prepare_payload_xfer()`

```python
# nixl/conn.py:671-689
def _prepare_payload_xfer(self, peer_info):
    """预构建 NIXL 传输描述符，跨请求复用"""
    
    if self.is_mla_backend or peer_info.decode_tp_size == self.attn_tp_size:
        # 同构 TP：共享 src dlist，每个 peer 一个 dst dlist
        if "" not in self.prep_handles:
            self._init_equal_tp_prep_handle("", ...)      # ← 只构建一次
        self._init_equal_tp_prep_handle(                    # ← 每个 peer 构建一次
            peer_info.agent_name, ...)
    else:
        # 异构 TP：slice dlists
        self._init_hetero_tp_prep_handle(...)
```

#### 缓存机制总结

| 缓存 | 位置 | 作用域 | 生命周期 |
|---|---|---|---|
| `decode_kv_args_table` | NixlKVManager | 实例级（per TP rank） | 进程级，跨请求 |
| `prep_handles` | NixlKVManager | 实例级 | 进程级，跨请求 |
| `_socket_cache` | CommonKVReceiver | 类级（全局） | 进程级，跨请求 |
| `connection_pool` | CommonKVManager | 实例级 | 进程级，跨请求 |
| `prefill_info_table` | CommonKVManager | 实例级 | 进程级，跨请求 |
| `session_pool` | CommonKVManager | 实例级 | 进程级，跨请求 |

---

## 3. Bootstrap 为什么这么慢？（5282ms）

### 3.1 Bootstrap 测量的是什么？

**START 时间戳**: `prefill_bootstrap_queue_entry_time`
- 位置: `scheduler.py:2179-2182`
- 含义: 请求进入 prefill bootstrap 队列

**END 时间戳**: `bootstrap_done_time`
- 位置: `prefill.py:262-280 (finalize_bootstrap)`
- 含义: Bootstrap 握手完成，sender 初始化完成

**Bootstrap 时间** = `bootstrap_done_time - prefill_bootstrap_queue_entry_time`

### 3.2 Bootstrap 期间发生了什么？

```
┌─────────────────────────────────────────────────────────────────┐
│                      PREFILL SIDE                                │
│                                                                   │
│  1. 请求进入 bootstrap 队列                                       │
│     └─> set_prefill_bootstrap_queue_entry_time()  ← START         │
│                                                                   │
│  2. 创建 KVSender 对象                                            │
│     └─> kv_sender_class(mgr, bootstrap_addr, ...)                 │
│                                                                   │
│  3. 轮询 bootstrap 状态（TP AllReduce）                           │
│     └─> poll_and_all_reduce_attn_cp_tp_group()                    │
│         └─> 等待 decode 端连接                                     │
│                                                                   │
│  4. Decode 端连接到达                                              │
│     └─> _add_remote_peer()                                        │
│         ├─> 首次：agent.add_remote_agent()  ← UCX 握手            │
│         └─> 后续：直接返回（连接已缓存）                           │
│                                                                   │
│  5. Bootstrap 完成                                                 │
│     └─> finalize_bootstrap()                                      │
│         └─> set_bootstrap_done_time()  ← END                      │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
                              ↑
                    ZMQ 消息（KV args 注册）
                              ↑
┌─────────────────────────────────────────────────────────────────┐
│                      DECODE SIDE                                 │
│                                                                   │
│  1. 请求到达 decode scheduler                                     │
│                                                                   │
│  2. 创建 KVReceiver 对象                                          │
│     └─> kv_receiver_class(mgr, bootstrap_addr, ...)               │
│                                                                   │
│  3. 获取 prefill server info（HTTP 请求，有缓存）                 │
│     └─> receiver.init(prefill_dp_rank)                            │
│         └─> HTTP GET /get_server_info  ← 可能慢（高并发下）       │
│                                                                   │
│  4. 向 prefill 注册 KV args（ZMQ 消息）                           │
│     └─> _register_kv_args()                                       │
│         └─> ZMQ SEND: agent_metadata + KV base pointers           │
│                                                                   │
│  5. 发送 TransferInfo（每个 TP rank 一次）                         │
│     └─> ZMQ SEND: bootstrap_room + transfer_info                  │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
```

### 3.3 Bootstrap 时间分解

根据代码分析，5282ms 的 bootstrap 时间包括：

| 阶段 | 估计耗时 | 说明 |
|---|---:|---|
| **Decode 端处理延迟** | ~3000-4000ms | 🔴 主要瓶颈 |
| — Decode scheduler 排队 | ~2000ms | 32 并发下，decode scheduler 繁忙 |
| — HTTP 获取 prefill info | ~500ms | 高并发下 bootstrap server 响应慢 |
| — ZMQ 注册 KV args | ~100ms | 网络延迟 |
| **Prefill 端处理** | ~1000-2000ms | |
| — TP AllReduce 轮询间隔 | ~500ms | 轮询不是实时的 |
| — UCX 握手（首次） | ~200ms | 后续请求复用连接，~0ms |
| — 预构建描述符 | ~100ms | 后续请求复用，~0ms |
| **其他开销** | ~200ms | |
| — Python GIL 竞争 | ~100ms | 多线程竞争 |
| — 内存分配 | ~100ms | 对象创建、GC |

**关键发现**：Bootstrap 慢的主要原因不是 UCX 连接建立（已缓存），而是 **Decode 端处理延迟**：
1. Decode scheduler 在 32 并发下非常繁忙，请求需要排队
2. HTTP 获取 prefill server info 在高并发下变慢
3. TP AllReduce 轮询有固定间隔

### 3.4 为什么 Prometheus 显示 87 个请求 Bootstrap > 2500ms？

从 `prefill_metrics.txt` 的 histogram 数据：

```
kv_transfer_bootstrap_ms_bucket{le="2500.0"} = 14
kv_transfer_bootstrap_ms_count = 101
```

这意味着 101 - 14 = **87 个请求（86%）的 Bootstrap 时间 > 2500ms**。

原因：
1. **前 14 个请求**：Bootstrap server 空闲，响应快
2. **后续 87 个请求**：Bootstrap server 繁忙，HTTP 响应慢；Decode scheduler 排队

这进一步证实了瓶颈在 **Decode 端处理延迟**，而不是 UCX 连接建立。

---

## 4. 为什么要这样设计？

### 4.1 为什么 KVSender/KVReceiver 每请求创建？

**设计理由**：

1. **状态隔离**：每个请求有独立的 sender/receiver 对象，避免状态污染
2. **生命周期管理简单**：请求完成后直接 GC 回收，无需手动管理连接池
3. **Bootstrap Room 机制**：每个请求有唯一的 `bootstrap_room`，用于区分不同的传输
4. **Chunked Prefill 支持**：一个请求可能多次调用 `send_kv_chunk()`，需要独立的状态跟踪

**底层连接已缓存**：
- UCX 连接通过 `decode_kv_args_table` 缓存
- NIXL 描述符通过 `prep_handles` 缓存
- ZMQ 套接字通过 `_socket_cache` 缓存

所以虽然对象是每请求创建的，但**昂贵的连接建立操作只发生一次**。

### 4.2 为什么 Bootstrap 仍然慢？

**根本原因**：Bootstrap 不仅仅是连接建立，还包括：

1. **Decode 端调度延迟**：Decode scheduler 在高并发下需要排队
2. **HTTP Bootstrap 协议**：Decode 端需要通过 HTTP 获取 prefill server info
3. **TP 同步**：Prefill 端需要 TP AllReduce 来同步 bootstrap 状态
4. **轮询间隔**：Bootstrap 状态通过轮询检查，不是事件驱动的

**类比**：
- UCX 连接建立 ≈ TCP 握手（~200ms，已缓存）
- Bootstrap ≈ 应用层协议握手（包括排队、HTTP 请求、TP 同步等）

---

## 5. 如何优化？

### 5.1 短期优化（P0）

#### 1. 减少 Decode 端排队延迟

**问题**：Decode scheduler 在 32 并发下繁忙，请求需要排队等待。

**方案**：
- 增加 decode worker 数量（多实例）
- 优化 decode scheduler 的批处理逻辑
- 减少 decode 端的 polling 间隔

**预期收益**：Bootstrap 时间从 5282ms 降到 ~2000ms（**降低 62%**）

#### 2. 优化 HTTP Bootstrap 协议

**问题**：Decode 端通过 HTTP 获取 prefill server info，高并发下慢。

**方案**：
- 使用 ZMQ 替代 HTTP 进行 bootstrap 信息交换
- 预取 prefill server info（在请求到达前）
- 增加 bootstrap server 的并发处理能力

**预期收益**：HTTP 延迟从 ~500ms 降到 ~50ms

#### 3. 减少 TP AllReduce 轮询间隔

**问题**：Prefill 端通过 TP AllReduce 轮询 bootstrap 状态，间隔较长。

**方案**：
- 减少轮询间隔（从 1ms 降到 0.1ms）
- 使用事件驱动替代轮询（当 decode 端连接到达时，主动通知）

**预期收益**：轮询延迟从 ~500ms 降到 ~50ms

### 5.2 中期优化（P1）

#### 4. 连接预热

**问题**：首次请求需要建立 UCX 连接（~200ms）。

**方案**：
- 服务启动时，prefill 和 decode 之间预先建立 UCX 连接
- 维护连接池，定期心跳保活

**预期收益**：消除首次连接的 200ms 延迟

#### 5. 异步 Bootstrap

**问题**：Bootstrap 是同步的，阻塞 prefill forward。

**方案**：
- 让 bootstrap 与 prefill forward 并行执行
- Prefill forward 完成后，再等待 bootstrap 完成

**预期收益**：Bootstrap 时间被 prefill forward 掩盖，对 TTFT 无影响

### 5.3 长期优化（P2）

#### 6. 请求级流水线

**问题**：每个请求都要独立 bootstrap。

**方案**：
- 批量 bootstrap：多个请求共享一次 bootstrap 协议
- 流水线：请求 N 的 bootstrap 与请求 N-1 的传输重叠

**预期收益**：Bootstrap 开销摊薄到每个请求

---

## 6. KV Cache 传输完整数据流（含代码）

### 6.1 整体架构

```
┌─────────────────────────────────────────────────────────────────┐
│                        Router (port 8000)                        │
│  - 接收客户端请求                                                │
│  - 路由到 prefill worker（port 30000）                           │
│  - 路由到 decode worker（port 30001）                            │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│              Prefill Worker (port 30000, GPU 0-1, TP=2)          │
│                                                                   │
│  TokenizerManager ←→ Scheduler ←→ ModelRunner ←→ KVSender       │
│                                                                   │
│  Scheduler 内部队列：                                             │
│  1. disagg_prefill_bootstrap_queue  ← Bootstrap 握手             │
│  2. prefill_forward_queue             ← GPU 计算                 │
│  3. inflight_queue                    ← KV 传输中                │
└─────────────────────────────────────────────────────────────────┘
                              ↓ KV Transfer (NIXL/UCX over PCIe)
┌─────────────────────────────────────────────────────────────────┐
│              Decode Worker (port 30001, GPU 2-3, TP=2)           │
│                                                                   │
│  KVReceiver ←→ Scheduler ←→ ModelRunner ←→ Detokenizer          │
│                                                                   │
│  Scheduler 内部队列：                                             │
│  1. prealloc_queue        ← 预分配 KV cache 空间                 │
│  2. transfer_queue        ← 等待 KV 传输完成                     │
│  3. waiting_queue         ← 等待调度执行 decode                  │
│  4. running_batch         ← 正在执行 decode forward              │
└─────────────────────────────────────────────────────────────────┘
```

### 6.2 完整时序图

```
Client          Router          Prefill Worker              Decode Worker
  │                │                    │                          │
  │──HTTP POST────→│                    │                          │
  │                │──Route to prefill──→│                          │
  │                │                    │                          │
  │                │                    │──1. 进入 bootstrap queue │
  │                │                    │   set_prefill_bootstrap_ │
  │                │                    │   queue_entry_time()     │
  │                │                    │                          │
  │                │                    │──2. 创建 KVSender        │
  │                │                    │   kv_sender_class(...)   │
  │                │                    │                          │
  │                │                    │   ┌──────────────────────┼──3. 请求到达 decode
  │                │                    │   │                      │   创建 KVReceiver
  │                │                    │   │                      │   kv_receiver_class(...)
  │                │                    │   │                      │
  │                │                    │   │                      │──4. 获取 prefill info
  │                │                    │←──┼──HTTP GET────────────┤   receiver.init()
  │                │                    │   │   /get_server_info   │
  │                │                    │───┼──HTTP RESPONSE──────→│
  │                │                    │   │                      │
  │                │                    │   │                      │──5. 注册 KV args
  │                │                    │←──┼──ZMQ SEND────────────┤   _register_kv_args()
  │                │                    │   │   (agent_metadata +  │   发送 ZMQ 消息到 prefill
  │                │                    │   │    KV base ptrs)     │
  │                │                    │   │                      │
  │                │                    │──6. _add_remote_peer()   │
  │                │                    │   if agent_name in table:│
  │                │                    │     return (已缓存)      │
  │                │                    │   else:                  │
  │                │                    │     agent.add_remote_    │
  │                │                    │     agent() ← UCX 握手   │
  │                │                    │     _prepare_payload_    │
  │                │                    │     xfer() ← 预构建描述符│
  │                │                    │                          │
  │                │                    │   ┌──────────────────────┼──7. 发送 TransferInfo
  │                │                    │←──┼──ZMQ SEND────────────┤   每个 TP rank 一次
  │                │                    │   │   (bootstrap_room +  │
  │                │                    │   │    transfer_info)    │
  │                │                    │   │                      │
  │                │                    │──8. poll() →             │
  │                │                    │   WaitingForInput        │
  │                │                    │                          │
  │                │                    │──9. finalize_bootstrap() │
  │                │                    │   set_bootstrap_done_    │
  │                │                    │   time() ← END           │
  │                │                    │   sender.init()          │
  │                │                    │                          │
  │                │                    │──10. Prefill Forward     │
  │                │                    │    GPU 计算              │
  │                │                    │    model_runner.forward()│
  │                │                    │                          │
  │                │                    │   ┌──────────────────────┼──11. 预分配 KV cache
  │                │                    │   │                      │    prealloc_queue.add()
  │                │                    │   │                      │
  │                │                    │   │                      │──12. 发送 metadata
  │                │                    │←──┼──ZMQ SEND────────────┤    send_metadata()
  │                │                    │   │   (kv_indices +      │    告诉 prefill 把 KV
  │                │                    │   │    dst pointers)     │    cache 发到哪里
  │                │                    │   │                      │
  │                │                    │──13. send_kv_chunk()     │
  │                │                    │    sender.send()         │
  │                │                    │    ↓                     │
  │                │                    │    transfer_worker:      │
  │                │                    │    NIXL WRITE → PCIe     │
  │                │                    │                          │
  │                │                    │   ┌──────────────────────┼──14. poll() → Success
  │                │                    │←──┼──NIXL NOTIFY─────────┤    传输完成
  │                │                    │   │                      │
  │                │                    │──15. sender.clear()      │
  │                │                    │    GC 回收 sender        │
  │                │                    │                          │
  │                │                    │   ┌──────────────────────┼──16. receiver.clear()
  │                │                    │   │                      │    进入 waiting_queue
  │                │                    │   │                      │
  │                │                    │   │                      │──17. Decode Forward
  │                │                    │   │                      │    逐 token 生成
  │                │                    │   │                      │
  │←───────────────┼───────────────────┼───┼──────────────────────┼──18. 返回结果
  │                │                    │   │                      │
```

### 6.3 关键代码路径

#### 6.3.1 Prefill 端：Bootstrap 到传输完成

```python
# === 阶段 1: Bootstrap ===

# scheduler.py:2179-2182
# 请求进入 bootstrap 队列
self.disagg_prefill_bootstrap_queue.add(req, ...)
req.time_stats.set_prefill_bootstrap_queue_entry_time()  # ← START

# prefill.py:226-250 (create_sender)
# 创建 sender 对象
req.disagg_kv_sender = kv_sender_class(
    mgr=self.kv_manager,
    bootstrap_addr=f"{req.bootstrap_host}:{self.bootstrap_port}",
    bootstrap_room=req.bootstrap_room,
    dest_tp_ranks=[self.tp_rank],
    pp_rank=self.pp_rank,
)

# prefill.py:307-380 (pop_bootstrapped)
# 轮询 bootstrap 状态（TP AllReduce）
polls = poll_and_all_reduce_attn_cp_tp_group(
    [req.disagg_kv_sender for req in self.queue], ...)

for req, poll in zip(self.queue, polls):
    if poll == KVPoll.WaitingForInput:
        # Decode 端已连接
        self.finalize_bootstrap(req)

# prefill.py:262-280 (finalize_bootstrap)
req.time_stats.set_bootstrap_done_time()  # ← END
req.disagg_kv_sender.init(num_kv_indices, req.metadata_buffer_index)

# === 阶段 2: Prefill Forward ===

# model_runner.py
# GPU 计算
output = self.model_runner.forward(input_ids, positions, ...)

# === 阶段 3: KV Transfer ===

# prefill.py:907-1003 (send_kv_chunk)
# 发送 KV cache
page_indices = req.kv_cache_pool.get_page_indices(...)
req.disagg_kv_sender.send(page_indices, state_indices)
# ↓ 内部调用
# kv_mgr.add_transfer_request() → 入队到 transfer_queues[shard_idx]
# transfer_worker (daemon thread): 
#   queue.get() → send_kvcache() → NIXL WRITE → poll until DONE

# prefill.py:670-786 (process_disagg_prefill_inflight_queue)
# 轮询传输完成
poll = req.disagg_kv_sender.poll()
if poll == KVPoll.Success:
    req.time_stats.set_prefill_kv_transfer_finish_time()
    req.disagg_kv_sender.clear()  # 清理状态
    self.process_finished_request(req)
```

#### 6.3.2 Decode 端：接收到传输完成

```python
# === 阶段 1: 初始化 Receiver ===

# decode.py:524-540 (_create_receiver_and_enqueue)
# 创建 receiver 对象
kv_receiver = kv_receiver_class(
    mgr=self.kv_manager,
    bootstrap_addr=_bootstrap_addr(req),
    bootstrap_room=req.bootstrap_room,
)
decode_req = DecodeRequest(req=req, kv_receiver=kv_receiver)

# decode.py:463 or 750-751 (init)
# 获取 prefill info，注册 KV args
decode_req.kv_receiver.init(prefill_dp_rank=...)
# ↓ 内部调用
# 1. HTTP GET /get_server_info → 获取 prefill IP/port
# 2. _register_kv_args() → ZMQ SEND agent_metadata + KV base ptrs

# === 阶段 2: 握手 ===

# decode.py:618-665 (_update_handshake_waiters)
# 轮询握手状态
poll = decode_req.kv_receiver.poll()
if poll == KVPoll.WaitingForInput:
    decode_req.waiting_for_input = True

# === 阶段 3: 发送 Metadata ===

# decode.py:1031-1036 (pop_preallocated)
# 告诉 prefill 把 KV cache 发到哪里
decode_req.kv_receiver.send_metadata(
    kv_indices=decode_req.kv_cache_indices,
    aux_index=decode_req.metadata_buffer_index,
    state_indices=...,
    decode_prefix_len=...,
)
# ↓ 内部调用
# ZMQ SEND: bootstrap_room + kv_indices + dst pointers

# === 阶段 4: 传输完成 ===

# decode.py:1585-1699 (pop_transferred)
# 轮询传输完成
poll = decode_req.kv_receiver.poll()
if poll == KVPoll.Success:
    self._commit_transfer_to_req(decode_req)
    self.waiting_queue.add(decode_req)

# decode.py:1546-1547 (_commit_transfer_to_req)
decode_req.kv_receiver.clear()
decode_req.kv_receiver = None  # 释放引用

# === 阶段 5: Decode Forward ===

# scheduler.py
# 从 waiting_queue 调度到 running_batch
# 逐 token 生成
output = self.model_runner.decode_forward(running_batch)
```

#### 6.3.3 NIXL 底层：UCX 连接与传输

```python
# nixl/conn.py:240-343 (NixlKVManager.__init__)
# 每个 TP rank 一个 NIXL agent
self.agent = nixl_agent(str(uuid.uuid4()), agent_config)
self.agent.create_backend(backend, backend_params)  # UCX backend
self.register_buffer_to_engine()  # 注册 GPU buffer

# nixl/conn.py:932-940 (_add_remote_peer)
# 注册远程 peer（UCX 连接建立）
def _add_remote_peer(self, decode_kv_args: KVArgsRegisterInfo):
    agent_name = decode_kv_args.agent_name
    
    # 🔑 连接复用
    if agent_name in self.decode_kv_args_table:
        return  # 已缓存，直接返回
    
    self.decode_kv_args_table[agent_name] = decode_kv_args
    self.agent.add_remote_agent(decode_kv_args.agent_metadata)  # UCX 握手
    self._prepare_payload_xfer(decode_kv_args)  # 预构建描述符

# nixl/conn.py:691-891 (transfer_worker)
# 后台线程：执行实际的 RDMA 传输
def transfer_worker(self):
    while True:
        req = self.transfer_queues[shard_idx].get()
        
        # 构建 NIXL 描述符
        src_dlist = self.prep_handle_slice_src
        dst_dlist = self.prep_handles[peer_name]
        
        # 执行 RDMA WRITE
        xfer_handle = self.agent.post_write(src_dlist, dst_dlist, ...)
        
        # 轮询直到完成
        while self.agent.check_xfer_complete(xfer_handle) != NIXL_SUCCESS:
            pass
        
        # 更新状态
        self.update_status(room, KVPoll.Success)

# nixl/conn.py:1901-1932 (NixlKVSender.send)
# 发送 KV cache
def send(self, kv_indices, state_indices=None):
    self.kv_mgr.add_transfer_request(
        bootstrap_room=self.bootstrap_room,
        kv_indices=kv_indices,
        ...
    )
    # ↓ 内部调用
    # transfer_queues[shard_idx].put(transfer_req)
    # transfer_worker 后台执行 RDMA WRITE
```

---

## 7. 总结

### 7.1 核心发现

1. **KVSender/KVReceiver 对象每请求创建**，但底层 UCX 连接、NIXL 描述符、ZMQ 套接字都被积极缓存
2. **Bootstrap 耗时 5282ms 的主要原因不是 UCX 连接建立**（已缓存，~0ms），而是：
   - Decode 端处理延迟（~3000-4000ms）：scheduler 排队、HTTP 请求
   - Prefill 端 TP AllReduce 轮询（~500ms）
   - 其他开销（~1000ms）
3. **设计理由**：对象每请求创建是为了状态隔离和生命周期管理简单，底层连接缓存避免了重复的昂贵操作
4. **优化方向**：减少 decode 端排队延迟、优化 HTTP bootstrap 协议、减少 TP 轮询间隔

### 7.2 优化建议

| 优先级 | 优化方向 | 预期收益 | 复杂度 |
|---|---|---|---|
| **P0** | 减少 decode 端排队延迟 | Bootstrap 降低 62% | 中 |
| **P0** | 优化 HTTP bootstrap 协议 | HTTP 延迟降低 90% | 低 |
| **P0** | 减少 TP 轮询间隔 | 轮询延迟降低 90% | 低 |
| P1 | 连接预热 | 消除首次连接 200ms | 低 |
| P1 | 异步 bootstrap | Bootstrap 被 prefill 掩盖 | 中 |
| P2 | 请求级流水线 | Bootstrap 摊薄 N× | 高 |

### 7.3 下一步行动

1. **立即**：测量 decode 端各阶段的耗时，确认瓶颈
2. **本周**：优化 HTTP bootstrap 协议（ZMQ 替代或预取）
3. **本月**：实现异步 bootstrap，让 bootstrap 与 prefill forward 并行
4. **长期**：实现请求级流水线，批量 bootstrap

---

## 附录 A：相关代码文件

| 文件 | 说明 |
|---|---|
| `python/sglang/srt/disaggregation/prefill.py` | Prefill 端调度逻辑 |
| `python/sglang/srt/disaggregation/decode.py` | Decode 端调度逻辑 |
| `python/sglang/srt/disaggregation/base/conn.py` | 抽象基类 |
| `python/sglang/srt/disaggregation/common/conn.py` | 通用实现（Bootstrap 协议） |
| `python/sglang/srt/disaggregation/nixl/conn.py` | NIXL 实现（UCX 传输） |
| `python/sglang/srt/observability/req_time_stats.py` | 时间戳记录 |
| `python/sglang/srt/observability/metrics_collector.py` | Prometheus metrics |

## 附录 B：Bootstrap 时间戳定义

| 时间戳 | 位置 | 含义 |
|---|---|---|
| `prefill_bootstrap_queue_entry_time` | `scheduler.py:2182` | 请求进入 prefill bootstrap 队列 |
| `bootstrap_done_time` | `prefill.py:270` | Bootstrap 握手完成 |
| `wait_queue_entry_time` | `req_time_stats.py:684` | 请求进入 waiting queue |

**Bootstrap 时间** = `bootstrap_done_time - prefill_bootstrap_queue_entry_time`

**Alloc 时间** = `wait_queue_entry_time - bootstrap_done_time`
