# MXFP4 PLAIN 模式问答笔记：语义、代价与 kernel 化路径

> 来源：2026-09-03 关于 MXFP4 PLAIN access 模式的问答讨论整理，按原始递进顺序组织：
> PLAIN 是什么 → 它承诺了什么 → 代价在代码哪里 → dequant 进 kernel 的好处 → quant 是否对称。
>
> 关联文档：
> - codec 逐段讲解：[qwen38_mxfp4_codec_walkthrough.md](qwen38_mxfp4_codec_walkthrough.md)
> - 验收记录：[qwen38_mxfp4_kv_acceptance_records.md](qwen38_mxfp4_kv_acceptance_records.md)
>
> 核心代码：
> - `python/sglang/srt/layers/quantization/fp4_kv_cache_quant_method.py`（access 规则声明）
> - `python/sglang/srt/mem_cache/memory_pool.py`（读取路径）
> - `python/sglang/srt/layers/attention/flashinfer_backend.py`（消费侧）
> - `python/sglang/srt/arg_groups/kv_cache_hook.py`（fail-fast）

---

## Q1: `_plain(...)` 是什么含义，有什么效果？

（起因：`fp4_kv_cache_quant_method.py` L930-933 的 access 声明）

```python
MXFP4KVCacheMethod.name: (
    _plain(_PREFILL, _MXFP4_FLASHINFER_BACKENDS, _MXFP4_SCALE, _BF16),
    _plain(_DECODE, _MXFP4_FLASHINFER_BACKENDS, _MXFP4_SCALE, _BF16),
),
```

### PLAIN 的定义与三种 access kind

枚举注释（fp4_kv_cache_quant_method.py L53-60）：

| kind | 注释原文 | 谁做转换 |
|---|---|---|
| `PLAIN` | "KV cache is already in the dtype/layout expected by the attention backend." | pool 读取时反量化，kernel 零感知 |
| `DEQUANT_WORKSPACE` | "stored quantized, then dequantized and decompressed into a temporary workspace before attention" | 先反量化进临时 workspace 再喂 kernel |
| `NATIVE_FP4` | "Attention backend directly consumes FP4 KV cache storage and scales" | kernel 直接吃 packed FP4+scale |

PLAIN 是三者中"量化最透明、kernel 改动为零"的一条。access 框架的分工声明（`KVCacheQuantMethodBase` docstring L110-115）：**quant_method（纯计算）► Pool（buffer + 批量 dequant）► Backend（view 适配）**。

### `_plain` 参数逐个解

`_plain`（L868-881）构造 `KVCacheAttentionAccess`：

| 参数 | MXFP4 传入的值 | 字段含义 |
|---|---|---|
| `_PREFILL` / `_DECODE` | 两条规则 | 这条规则管哪个阶段 |
| `_MXFP4_FLASHINFER_BACKENDS` | `frozenset({"flashinfer"})`（L859） | backend 白名单：只有 flashinfer 命中 |
| `_MXFP4_SCALE` | `"mxfp4"`（L849） | scale 语义标签；非 None 同时令 `storage_dtype=uint8`——pool 存 packed FP4 字节 |
| `_BF16` | `torch.bfloat16`（L852） | **kernel 实际消费的 KV dtype** |

两行声明连起来读：*"mxfp4 的 prefill 和 decode，都只允许 flashinfer 使用；pool 存 uint8 packed FP4（recipe mxfp4），读取时整层反量化成 BF16 交给 kernel。"*

### 四个运行时效果

1. **读路径换向**：`needs_plain_kv_dequant_read()`（L152-158：存在 `kind==PLAIN 且 storage_dtype 非 None` 的 access）为真 → `_get_key_buffer/_get_value_buffer` 走 `dequantize_kv_tensor` 整层反量化。flashinfer 拿到的张量与未量化时形状/dtype 完全一样，**kernel 零改动**。
2. **FlashInfer KV dtype 解析**：L2 改造后 `flashinfer_kv_cache_dtype` 从活跃 phase 的 `access.attention_kv_dtype` 解析——命中 PLAIN → BF16（对比 NVFP4 prefill 命中 DEQUANT_WORKSPACE → FP8）。
3. **backend 门禁（fail-fast）**：其他 backend（triton、trtllm_mha…）`resolve_attention_access` 返回 None → 启动参数校验直接拒绝。"mxfp4 只在 flashinfer + MHA + disable-cuda-graph 组合可用"的机制来源。
4. **容量预算加 scratch**：`compute_cell_size`（L827-837）注释明说——"PLAIN materializes one layer's K and V as BF16 simultaneously"，预算额外加一层瞬态 BF16 足迹。

### 为什么 L2 选 PLAIN

L2 阶段定位是**功能集成验证 + 数值 oracle 闭环**，不是性能：PLAIN 是唯一"零新 kernel、天然适配现有 prefill/decode"的接入方式，代价是每次读取全量反量化（见 Q3）。性能留给 L3 的 `NATIVE_FP4`。

### Registry 全景对比（L917-934）

| method | prefill | decode | 允许 backend |
|---|---|---|---|
| Unquantized | PLAIN | PLAIN | 任意 |
| NVFP4 | DEQUANT_WORKSPACE→FP8 | NATIVE_FP4（packed） | flashinfer / trtllm_mha |
| FP4MXBlock16 | PLAIN→BF16 | PLAIN→BF16 | triton/torch_native/flex/trtllm_mha(+fa4) |
| **MXFP4** | **PLAIN→BF16** | **PLAIN→BF16** | **仅 flashinfer** |

---

## Q2: "承诺在 torch 层做掉 quant/dequant，attention kernel 完全按既有实现"——对吗？

对，这正是 PLAIN 的契约，且 fail-fast 报错文案自己承认（"validation-only"）。三个精确化：

### 精确化 1：位置比"torch 层"更具体——在 pool 读取时

- **quant（写）**：`set_kv_buffer` → `quant_method.quantize_and_store`（写入前完成）；
- **dequant（读）**：`pool.get_kv_buffer(layer_id)` → `_get_key_buffer` → `dequantize_kv_tensor`（backend 拿到之前完成）。

backend 里的代码还是那一行 `pool.get_kv_buffer(layer_id)`，拿到的已经是 BF16——"既有实现"对 backend 是字面意义的：连调用路径都原封不动。

### 精确化 2：kernel 零改动 ≠ backend 零改动

flashinfer_backend.py 的 **host 侧胶水层**有两处新增（配置逻辑，不碰 kernel 数学）：

- `flashinfer_kv_cache_dtype` 改为从 access 规则解析（PLAIN → BF16），不再从 workspace 标志推断；
- `kv_cache_scales_valid=False`——checkpoint 自带的 FP8 descale 不参与（防原笔记 §7 的 descale 读写不对称 bug 复现）。

### 精确化 3：这份承诺自带"仅供验证"的代价标签

fail-fast 报错文案（kv_cache_hook.py L60-64）原话：

> *"mxfp4 uses a **validation-only** BF16 PLAIN read that allocates a full-layer temporary; pass --disable-cuda-graph."*

三个代价（见 Q3）全部源自"反量化被放进读取路径"这一条设计。

---

## Q3: 三个代价在代码上哪里体现？

### 代价 1：整层 BF16 瞬态物化

**产出侧**——memory_pool.py `_get_key_buffer`（L2325-2339，`_get_value_buffer` L2349-2363 对称）：

```python
def _get_key_buffer(self, layer_id):
    if (self.is_quantized_kv_cache
        and self.quant_method.needs_plain_kv_dequant_read()):
        return self.quant_method.dequantize_kv_tensor(...)   # ← 每次调用新建
    if self.store_dtype != self.dtype:
        return self.k_buffer[...].view(self.dtype)           # ← 未量化只是 view
```

对比清楚：未量化路径返回 buffer 的 **view（零拷贝）**；PLAIN 路径每次调用走
`dequantize_kv_tensor`，其内部（kvfp4_tensor.py L267-272）`torch.empty` 分配整层
unpack buffer——**没有任何缓存复用**，调多少次物化多少次。

**预算侧**——`compute_cell_size`（fp4_kv_cache_quant_method.py L827-837）：

```python
# PLAIN materializes one layer's K and V as BF16 simultaneously. Reserve
# that transient footprint ... even though it is not a persistent pool allocation.
plain_scratch_size = head_num * head_dim * 2 * 2B * kv_size   # 4 KiB/token，没乘层数
```

物理账：packed 16 KiB + scale 1 KiB + **scratch 4 KiB = ~21 KiB/token**。

### 代价 2：decode 每步全量反量化

调用链上没有"算一次、复用多次"的机制：

```
flashinfer_backend.py L1520（forward_decode）
  → pool.get_kv_buffer(layer_id)        # decode 每步、每个全注意力层各调一次
    → _get_key_buffer / _get_value_buffer   # 每次都走 dequantize 分支
```

每步每层带宽账：读 packed FP4（0.5 B/elem）+ 读 scale + **写** BF16 scratch
（2 B/elem）+ kernel **读** BF16 scratch（2 B/elem）≈ **4.5 B/elem**——比不量化
直接读 BF16（2 B/elem）还费。**量化收益只在容量上兑现（17 KiB/token），带宽上
倒贴**，这就是报错文案自称 validation-only 的实质。

顺带：prefill ragged 分支（flashinfer_backend.py L1409-1410）写了两次
`get_kv_buffer(...)[0]` 和 `[1]`——每次调用内部 K、V 都反量化，等于 4 次整层
反量化、用掉一半。prefill 低频所以无感，但是"按次物化无缓存"策略的直接产物。

### 代价 3：强制 disable-cuda-graph

kv_cache_hook.py L60-64，参数解析阶段直接 raise（原文见 Q2 精确化 3）。机制原因
就是代价 1 的 `torch.empty`：CUDA graph capture 要求重放时地址与语义固定，而读取
路径每次调用分配新临时、写完即弃。测试锁定：
`test/registered/unit/server_args/test_kv_cache_hook_mxfp4.py::TestMxfp4KvCacheHook`。

### 汇总表

| 代价 | 代码位置 | 体现形式 |
|---|---|---|
| 整层 BF16 物化 | memory_pool.py:2325-2339 + fp4_kv_cache_quant_method.py:827-837 | 每次 `torch.empty` 无缓存；容量预留 `plain_scratch_size` |
| decode 每步全量反量化 | flashinfer_backend.py:1520 → get_kv_buffer → dequant 分支 | 调用链无缓存层，带宽账倒贴；prefill ragged 双重调用 |
| 禁 CUDA graph | kv_cache_hook.py:60-64 | 解析期 raise，文案自述 "validation-only" |

三处共同根因：**反量化被放进了读取路径（torch 层、按次执行）**——它成就了零
kernel 接入，也把三个代价写进同一批代码。

---

## Q4: 如果 dequant 在 kernel 里（NATIVE_FP4），好处是什么？

根源只有一条：**decode 是 memory-bound，性能由"从显存读多少字节"决定**。
decode attention 每步读全部历史 KV，算术强度极低（每元素一次乘加），时间近似
正比于读取字节数。

### 核心账：每元素读取字节

| 路径 | 每元素读取 | 说明 |
|---|---|---|
| BF16 未量化 | 2.00 B | 精度上限基线 |
| FP8（L0 现役基线） | 1.00 B | 容量/带宽双赢，生产默认 |
| MXFP4 PLAIN | **≈ 4.53 B** | FP4 0.5B + scale + 写 scratch 2B + kernel 读 scratch 2B——倒贴 |
| MXFP4 NATIVE | **≈ 0.53 B** | FP4 0.5B + scale (1/32B)，寄存器内解包查表 |

NATIVE 相对 BF16 省约 **73%** KV 读取带宽、相对 FP8 省 **47%**——等比转化为大
batch/长上下文的 TPOT。PLAIN 不省反慢；它兑现的只有容量。原笔记 §10 的结论
"decode 必须 kernel 内 inline dequant"说的就是这笔账。

### 三个代价如何同时消失

1. **scratch 消失**：没有物化 BF16 中间产物，kernel 直接读 pool 持久 packed buffer。`compute_cell_size` 的 `plain_scratch_size` 可删，省下的 mem-fraction 多放真实 KV；每步临时分配/释放也没了。access 框架把这点声明化：`_native_fp4`（L901-914）的 **`workspace_dtype=None`——"没有临时 workspace"本身就是这个好处的签名**。
2. **CUDA graph 兼容**：NATIVE 路径无 per-call 临时张量，kernel 读地址固定的持久 buffer。现成先例：NVFP4 decode 的 trtllm_mha native FP4 kernel 直读 packed，照常跑 CUDA graph。
3. **"每步全量反量化"从瓶颈变成几条指令**：PLAIN 的反量化是十几个 torch op 的 launch 序列（每步每层一遍，含 CPU launch 开销）；NATIVE 把 unpack+查表+乘 scale 折进 attention kernel 内部，摊进访存流水，近乎免费。

### trade-off 另一面

- **必须写/改 kernel**：FlashInfer 没有 FP4 KV 输入 kernel（原笔记 §10 原因 1），三条路——Triton 原型（复用现有 paged metadata，L3 首选）、FlashInfer 扩展（动外部项目）、sgl-kernel CUDA/CUTLASS（上限最高成本最高）；
- **只在 decode 兑现**：prefill 是 compute-bound，kernel 内解包救不了它——prefill/decode 不对称原则的另一半；
- **验证成本上升**：bit-exact 契约从"torch vs torch oracle"变成"CUDA kernel vs L2 oracle"。L3 验收标准（kernel 中间解码值对 L2 codec、单层 output 对 L2、再对模型 logprob）为此准备。

**L2→L3 的分界线就是这笔交换发生的时刻**：先在 PLAIN 上把数值语义锁死（oracle
链闭环），再让 L3 kernel 只对齐已锁死的语义，不用同时调试量化数学和 kernel 正确性。

---

## Q5: quant 也应该放进 kernel 吗？代价/优化是否对称？

要放——原笔记 §11.7 L3 优先级 2 本来就有："量化写入 kernel：融合 block
amax/scale、E2M1 pack、按 loc scatter 与 scale 写入；**参考 `quant_store_kv_mxfp8`**"。
但两个优化的收益类型完全不同，**不对称**。三个根源：

### 不对称 1：数据量不同 → 瓶颈类型不同

| | dequant（读） | quant（写） |
|---|---|---|
| 每步处理量 | **全部历史 KV**（随 seq 线性增长，16 层 × T tokens） | **仅当前 token**（恒定，每层 K+V 共 4 KiB bf16 输入） |
| 瓶颈 | HBM **带宽**（读的字节数决定 TPOT） | **kernel launch / CPU 开销**（数据太小，带宽无关） |

当前 `batched_quantize` 是 **20+ 个 eager torch op**（fp32 cast、amax、log2、
floor、三个 where、div、距离、argmax、pack…），decode 每步 16 层 × K/V 两份 →
**数百次小 launch，每次只处理几 KB**，GPU 大部分时间在等发射。quant kernel 化的
收益是 **launch 融合**：数百次 → 每层 1 个。带宽上无账可算。

### 不对称 2：融合位置不同 → "进哪个 kernel"不对称

- **dequant 必须 inline 进 attention kernel**：中间产物（BF16 K/V）只被 attention
  消费一次，独立算就得写显存中转（scratch）——要消除的正是中转。没得选。
- **quant 独立成 kernel 即可**：发生在 attention **之前**（`set_kv_buffer`），输入
  qkv_proj 的 K/V、输出写 pool，不在 attention 数据路径上，无中转需消除。最优形态
  是"quant + scatter 融合成**单个写入 kernel**"——现成范本 `quant_store_kv_mxfp8`
  （memory_pool.py L3412-3420）：
  *"the layer hands us bf16 K/V and **one kernel** quantizes + scatters the fp8
  payload and the interleaved UE8M0 scales"*——MXFP4 的 L3 写入 kernel 照此做 E2M1 版。

### 不对称 3：CUDA graph 的角色也不对称

- dequant 侧：scratch 分配是 **graph 的破坏者**，NATIVE 化才修复；
- quant 侧：eager 纯 tensor 实现**本来就 graph 可捕获**（kvfp4_tensor.py L168-173
  注释明说）——kernel 化不是修复，只是减少发射次数。

"禁 cuda graph"这笔账完全记在 dequant 头上，quant 没份。

### 不对称 4：数值锁定的难度

- **dequant 纯查表乘法**：kernel 化后天然确定，对 L2 oracle 的 bit-exact 相对容易；
- **quant 要在 CUDA 里复现 fp32 语义**：amax → `log2f` → `floor` → RNE
  tie-to-even → 打包，每步都需与 torch fp32 链 bit 对齐（含 `5.0 → 4.0` 消歧等
  实现决策）。这正是 L2 先用 eager torch 锁死语义、conformance 做 CPU/CUDA byte
  parity 的原因：L3 quant kernel 是在"已锁死的契约"上做移植，不是边写 kernel 边定义语义。

### 总结：不对称的四个维度

| 维度 | dequant | quant |
|---|---|---|
| 收益类型 | 带宽（读全量历史，4.53→0.53 B/elem） | launch 融合（写当前 token，数百次→每层 1 次） |
| 形态 | 必须 inline 进 attention kernel | 独立 quant_store kernel（参考 MXFP8 先例） |
| 与 graph 的关系 | scratch 是破坏者，NATIVE 修复 | 本来就 graph-safe，kernel 化只是提速 |
| 数值锁定 | 纯查表，天然确定 | 须复现 fp32 RNE 语义，靠 L2 契约兜底 |

**两边都该进 kernel，但不是同一个理由、同一个位置、同一个收益量级。** 路线图
§11.8"只开发 decode attention kernel；**另做量化写入 kernel**"拆开的两半对应这对
不对称——dequant 决定 TPOT 的下限，quant 决定 CPU/发射开销的下限，各自独立兑现，
谁也不依赖谁。

---

## 全景收束：PLAIN 在 L0-L3 路线中的位置

```
L0  FlashInfer FP8 decode（现役基线，1.0 B/elem）
      ↓ L1：torch FP8 golden 证明 decode 数学可复制（与量化格式解耦）
L2  MXFP4 PLAIN：torch 层包办 quant/dequant，kernel 零改动
      —— 数值语义锁死（OCP codec ↔ oracle bit-exact），性能不达标（validation-only）
      ↓ L3：数值契约不变，只换执行位置
L3  NATIVE dequant（inline 进 decode kernel：带宽 0.53 B/elem、scratch 消失、graph 兼容）
    + NATIVE quant（独立 quant_store kernel：launch 融合）
```

PLAIN 的本质是**把"语义正确"与"执行高效"解耦**：先证明这套量化数学在现有 kernel
生态下端到端数值等价，再把执行位置逐块搬进 kernel，每次搬动只需对齐已锁死的数值
契约，而不是重新验证整条链。这也是 access 框架用 (phase, backend, kind) 三元组
声明规则的意图：换 kind = 换执行位置，语义随 recipe 走，不随 kernel 走。
