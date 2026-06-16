# Bootstrap 流程拆解 Profile 方案

## 目标

拆解 KV Transfer 的 Bootstrap 流程（5282ms），精确定位每个子阶段的耗时。

## 当前已知信息

- Bootstrap 总耗时：5282ms
- 测量方式：`bootstrap_done_time - prefill_bootstrap_queue_entry_time`
- 缺失：中间步骤的时间戳

## Bootstrap 流程分解

### Prefill 端

```
1. 请求进入 bootstrap 队列
   └─> set_prefill_bootstrap_queue_entry_time()  ← START

2. 创建 KVSender 对象
   └─> kv_sender_class(...)

3. 轮询 bootstrap 状态（TP AllReduce）
   └─> poll_and_all_reduce_attn_cp_tp_group()
       └─> 等待 decode 端连接

4. Decode 端连接到达
   └─> _add_remote_peer()
       ├─> 首次：agent.add_remote_agent()  ← UCX 握手
       └─> 后续：直接返回（连接已缓存）

5. Bootstrap 完成
   └─> finalize_bootstrap()
       └─> set_bootstrap_done_time()  ← END
```

### Decode 端

```
1. 请求到达 decode scheduler

2. 创建 KVReceiver 对象
   └─> kv_receiver_class(...)

3. 获取 prefill server info（HTTP 请求，有缓存）
   └─> receiver.init(prefill_dp_rank)
       └─> HTTP GET /get_server_info  ← 可能慢

4. 向 prefill 注册 KV args（ZMQ 消息）
   └─> _register_kv_args()
       └─> ZMQ SEND: agent_metadata + KV base pointers

5. 发送 TransferInfo（每个 TP rank 一次）
   └─> ZMQ SEND: bootstrap_room + transfer_info
```

## Profile 方案

### 方案 1：添加时间戳到代码（推荐）

在关键代码路径添加 `time.perf_counter()` 时间戳，记录到日志或 metrics。

**优点**：
- 精确测量每个子阶段
- 可以收集大量样本
- 可以集成到 Prometheus metrics

**缺点**：
- 需要修改核心代码
- 可能影响性能（微小）

**实现**：

#### 1.1 Prefill 端时间戳

```python
# prefill.py:226-250 (create_sender)
def create_sender(self, req: Req):
    t0 = time.perf_counter()
    
    kv_sender_class = get_kv_class(self.backend, KVClassType.SENDER)
    req.disagg_kv_sender = kv_sender_class(...)
    
    t1 = time.perf_counter()
    req.time_stats.bootstrap_create_sender_ms = (t1 - t0) * 1000
```

```python
# prefill.py:307-380 (pop_bootstrapped)
def pop_bootstrapped(self) -> List[Req]:
    t0 = time.perf_counter()
    
    polls = poll_and_all_reduce_attn_cp_tp_group(...)
    
    t1 = time.perf_counter()
    # 记录轮询耗时
    for req in self.queue:
        req.time_stats.bootstrap_poll_duration_ms = (t1 - t0) * 1000
```

```python
# nixl/conn.py:932-940 (_add_remote_peer)
def _add_remote_peer(self, decode_kv_args: KVArgsRegisterInfo):
    t0 = time.perf_counter()
    
    if agent_name in self.decode_kv_args_table:
        # 已缓存
        return
    
    # 首次连接
    self.agent.add_remote_agent(decode_kv_args.agent_metadata)
    
    t1 = time.perf_counter()
    # 记录到 manager 的统计
    self.bootstrap_add_remote_agent_ms = (t1 - t0) * 1000
    
    self._prepare_payload_xfer(decode_kv_args)
    
    t2 = time.perf_counter()
    self.bootstrap_prepare_payload_ms = (t2 - t1) * 1000
```

#### 1.2 Decode 端时间戳

```python
# decode.py:524-540 (_create_receiver_and_enqueue)
def _create_receiver_and_enqueue(self, req: Req):
    t0 = time.perf_counter()
    
    kv_receiver = kv_receiver_class(...)
    
    t1 = time.perf_counter()
    req.time_stats.bootstrap_create_receiver_ms = (t1 - t0) * 1000
```

```python
# common/conn.py:233-276 (try_ensure_parallel_info)
def try_ensure_parallel_info(self, bootstrap_addr: str) -> bool:
    if bootstrap_addr in self.prefill_info_table:
        return True  # 已缓存
    
    t0 = time.perf_counter()
    
    response = requests.get(url, timeout=5)
    
    t1 = time.perf_counter()
    self.bootstrap_http_get_ms = (t1 - t0) * 1000
```

```python
# nixl/conn.py:2084-2141 (_register_kv_args)
def _register_kv_args(self):
    t0 = time.perf_counter()
    
    for bootstrap_info in self.bootstrap_infos:
        sock.send_multipart([...])
    
    t1 = time.perf_counter()
    self.bootstrap_zmq_send_ms = (t1 - t0) * 1000
```

#### 1.3 新增时间戳字段

在 `req_time_stats.py` 添加：

```python
# Bootstrap 子阶段时间戳
bootstrap_create_sender_ms: float = 0.0      # 创建 sender 对象
bootstrap_create_receiver_ms: float = 0.0    # 创建 receiver 对象
bootstrap_http_get_ms: float = 0.0           # HTTP 获取 prefill info
bootstrap_zmq_send_ms: float = 0.0           # ZMQ 发送 KV args
bootstrap_add_remote_agent_ms: float = 0.0   # NIXL add_remote_agent
bootstrap_prepare_payload_ms: float = 0.0    # 预构建描述符
bootstrap_poll_duration_ms: float = 0.0      # 轮询耗时
```

### 方案 2：Monkey Patch（快速验证）

不修改核心代码，通过 monkey patch 添加计时。

**优点**：
- 不需要修改核心代码
- 快速验证

**缺点**：
- 只能测量函数级别的耗时
- 无法测量函数内部的子阶段

**实现**：

```python
# bootstrap_profiler.py
import time
from functools import wraps

def profile_function(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        t0 = time.perf_counter()
        result = func(*args, **kwargs)
        t1 = time.perf_counter()
        duration_ms = (t1 - t0) * 1000
        print(f"[PROFILE] {func.__name__}: {duration_ms:.2f}ms")
        return result
    return wrapper

# Monkey patch
import sglang.srt.disaggregation.nixl.conn as nixl_conn
nixl_conn.NixlKVManager._add_remote_peer = profile_function(nixl_conn.NixlKVManager._add_remote_peer)

import sglang.srt.disaggregation.common.conn as common_conn
common_conn.CommonKVManager.try_ensure_parallel_info = profile_function(common_conn.CommonKVManager.try_ensure_parallel_info)
```

### 方案 3：Logging + 分析（推荐用于生产）

在关键路径添加结构化日志，事后分析。

**优点**：
- 可以收集大量数据
- 可以离线分析
- 对性能影响小

**实现**：

```python
import logging
import time

logger = logging.getLogger(__name__)

def _add_remote_peer(self, decode_kv_args):
    t0 = time.perf_counter()
    
    # ... 原有代码 ...
    
    t1 = time.perf_counter()
    logger.info(f"[BOOTSTRAP] _add_remote_peer: {(t1-t0)*1000:.2f}ms, agent={agent_name}")
```

## 推荐方案

**阶段 1：快速验证（Monkey Patch）**
- 使用方案 2，快速测量关键函数的耗时
- 验证哪些子阶段是瓶颈

**阶段 2：精确测量（添加时间戳）**
- 使用方案 1，在关键代码路径添加时间戳
- 集成到 Prometheus metrics

**阶段 3：生产监控（Logging）**
- 使用方案 3，添加结构化日志
- 持续监控 bootstrap 性能

## 需要测量的关键函数

| 函数 | 文件 | 预期耗时 | 说明 |
|---|---|---:|---|
| `create_sender()` | prefill.py | ~1ms | 创建 sender 对象 |
| `create_receiver()` | decode.py | ~1ms | 创建 receiver 对象 |
| `try_ensure_parallel_info()` | common/conn.py | ~100-2000ms | 🔴 HTTP 获取 prefill info |
| `_register_kv_args()` | nixl/conn.py | ~10-50ms | ZMQ 发送 KV args |
| `_add_remote_peer()` | nixl/conn.py | ~0-200ms | 🔴 UCX 握手（首次） |
| `_prepare_payload_xfer()` | nixl/conn.py | ~10-100ms | 预构建描述符 |
| `poll_and_all_reduce_attn_cp_tp_group()` | prefill.py | ~100-1000ms | 🔴 TP 轮询 |

## 预期结果

通过拆解 profile，我们预期会发现：

1. **HTTP 获取 prefill info**（~1000-2000ms）
   - 高并发下 bootstrap server 响应慢
   - 可以通过预取或 ZMQ 替代优化

2. **TP 轮询间隔**（~500-1000ms）
   - 轮询不是实时的
   - 可以减少轮询间隔或使用事件驱动

3. **UCX 握手**（~0-200ms）
   - 首次连接需要握手
   - 后续连接复用，~0ms

4. **其他开销**（~1000-2000ms）
   - Decode scheduler 排队
   - Python GIL 竞争
   - 内存分配

## 下一步

1. 实现 Monkey Patch 方案，快速验证
2. 根据结果决定是否需要添加时间戳到核心代码
3. 集成到 Prometheus metrics，持续监控
