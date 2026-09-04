# Qwen3.8 Native MXFP4 Triton Decode Kernel 学习手册

> 面向 Triton 初学者的可扩展教程。本文把 MXFP4 KV cache 的 native Triton decode kernel 从初版（T0）到最终优化版（T3）的实现、验证和性能结论整理为循序渐进的学习路径。
>
> 本文不替代验收记录：[qwen38_mxfp4_kv_acceptance_records.md](qwen38_mxfp4_kv_acceptance_records.md) 记录“做了什么、测了什么、结论是什么”；本文解释“代码为什么这样写、优化为什么有效”。

## 目录

1. [学习目标与阅读方式](#1-学习目标与阅读方式)
2. [先给结论：性能是如何接近 FlashInfer 的](#2-先给结论性能是如何接近-flashinfer-的)
3. [术语表](#3-术语表)
4. [问题背景：为什么需要 native MXFP4 kernel](#4-问题背景为什么需要-native-mxfp4-kernel)
5. [数据格式与内存布局](#5-数据格式与内存布局)
6. [源码地图与调用链](#6-源码地图与调用链)
7. [初版 T0：per-head split-KV decode](#7-初版-t0per-head-split-kv-decode)
8. [优化 T1：grouped kernel 消除 GQA 重复读取](#8-优化-t1grouped-kernel-消除-gqa-重复读取)
9. [优化 T2：提高 split 数填满 GPU](#9-优化-t2提高-split-数填满-gpu)
10. [优化 T3：位构造反量化](#10-优化-t3位构造反量化)
11. [实验分支：为什么 `tl.dot_scaled` 默认关闭](#11-实验分支为什么-tldot_scaled-默认关闭)
12. [正确性、性能与价值边界](#12-正确性性能与价值边界)
13. [后续深入教学占位](#13-后续深入教学占位)

---

## 1. 学习目标与阅读方式

读完第一轮后，应能回答：

1. MXFP4 的 packed data 与 scale 分别是什么，为什么每 32 个元素共享一个 scale？
2. native kernel 和 L2 PLAIN 路径有什么本质差异？
3. 初版为什么在 GQA 下会重复读取 K/V？
4. grouped kernel、更多 KV splits、位构造分别解决了什么瓶颈？
5. 为什么最终态接近 FP8 FlashInfer，但在该测试场景下没有明显超过它？

### 推荐阅读方法

不要一开始逐行读千行 kernel。按下面顺序建立模型：

1. 先读本文第 2、4、5 节，建立性能目标和数据格式。
2. 打开 `mxfp4_decode_attention_fwd`，只看输入、分支和 kernel launch。
3. 回头读 T0 的 stage 1；只跟踪“一个 Triton program 做了什么”。
4. 对照 T1 的 grouped stage 1，找出“一个 Q head”变成“一组 Q heads”的地方。
5. 最后学习 split、online softmax、位构造和实验分支。

后续每一节深入教学统一采用：**目标 → 前置知识 → 输入/输出 → 代码分段 → 手算示例 → 常见误解 → 小练习** 的格式。

---

## 2. 先给结论：性能是如何接近 FlashInfer 的

这里的“接近 FlashInfer”需要准确理解。对比的不是“同一种 MXFP4 kernel 的两个实现”，而是三条不同路径：

| 路径 | KV 读取/计算方式 | 长文 TPOT（4096/128） |
|---|---|---:|
| FP8 + FlashInfer | 生产 FP8 基线 | 23.8 ms |
| MXFP4 + PLAIN + FlashInfer | 先将整层压缩 KV 物化为 BF16，再交给 FlashInfer | 152.7 ms |
| MXFP4 + native Triton | Triton 直接读取 packed FP4 + scale，并在 kernel 内反量化 | 25.9 ms |

`TPOT`（Time Per Output Token）是生成阶段每产出一个 token 的耗时。最终 native 路径与 FP8 FlashInfer 相差约 9%，并把最大 KV 容量提高到 1.52 倍。

性能收敛的路线是：

```text
L1：建立 FP8 attention 的可信参考实现
  ↓
L2：锁定 MXFP4 codec，并用 PLAIN + FlashInfer 验证数值语义
  ↓
T0：直接在 Triton 内核中读取 packed KV、核内反量化（正确但慢）
  ↓
T1：grouped kernel，消除 GQA 的重复 K/V 读取
  ↓
T2：KV splits 8 → 32，提高小 batch 长上下文下的 GPU 并行度
  ↓
T3：整数位构造 BF16，替代热路径的 exp2 与浮点乘法
  ↓
最终：MXFP4 native Triton 接近 FP8 FlashInfer
```

| 优化 | 主要瓶颈 | 长文服务级效果 |
|---|---|---:|
| T0 初版 | GQA 重复读 K/V，split 数偏小 | 47.5 ms |
| T1 grouped kernel | 消除同一 KV 被多个 Q head 重读 | 47.5 → 30.5 ms |
| T2 splits=32 | CTA 数过少，GPU 空闲 | 30.5 → 25.7 ms |
| T3 位构造解包 | `exp2` 与 fp32 乘法链在热路径中开销高 | 微基准快 7%~10%；服务级约 25.9 ms |

> 结论：不是某一条“神奇指令”带来了性能，而是依次去掉了 **冗余内存读取、并行度不足、热路径解包指令开销**。

---

## 3. 术语表

| 术语 | 含义 |
|---|---|
| Kernel | 在 GPU 上执行的函数；这里指 Triton 编译生成的 GPU 程序。 |
| Triton program | Triton 的一个工作实例，近似 CUDA 的 CTA / thread block。 |
| Grid | 一次 kernel launch 中所有 program 的三维排列。 |
| CTA | CUDA Thread Block；可粗略理解为一个会被调度到 SM 上执行的工作块。 |
| SM | Streaming Multiprocessor，GPU 中执行 CTA 的计算单元。 |
| Q/K/V | Query、Key、Value；attention 的三个输入。 |
| Decode | 自回归生成阶段，一次通常只处理每个请求的一个新 token。 |
| KV cache | 保存历史 token 的 K/V，供后续 decode attention 读取。 |
| MHA | Multi-Head Attention；每个 Q head 有自己的 K/V head。 |
| GQA | Grouped-Query Attention；多个 Q head 共用一个 KV head。 |
| MQA | Multi-Query Attention；所有 Q head 共用很少的 KV head。 |
| MXFP4 | OCP MX 规范中的 block-scaled 4-bit 浮点格式。 |
| E2M1 | 4-bit 数值格式：1 位符号、2 位指数、1 位尾数。 |
| E8M0 | 1-byte scale 格式，可理解为只存指数的 2 的幂缩放因子。 |
| Nibble | 半个 byte，即 4 bit。一个 uint8 包含两个 nibble。 |
| Dequant / 反量化 | 将压缩整数/低精度浮点恢复为计算用浮点值。 |
| PLAIN | 读取 KV 时先物化为 attention backend 所需的 BF16；便于验证但性能差。 |
| NATIVE_FP4 | attention kernel 直接读 packed FP4 和 scale，并在内部反量化。 |
| Split-KV | 将长 KV 序列切为多个片段，并行计算局部 attention，再合并。 |
| Online softmax | 顺序处理 token tile 时维护稳定的 max、sum 和输出累加器，避免先存完整 score。 |
| `tl.dot` | Triton 矩阵乘法原语，通常可使用 Tensor Core。 |
| Tensor Core | GPU 中专门加速低精度矩阵乘法的硬件。 |
| SFU | Special Function Unit；执行指数等特殊函数的硬件单元。 |
| FTZ | Flush To Zero；极小的非正规浮点数被清零的行为。 |
| Bit-exact | 逐 bit 一致，不只是数值“大致相同”。 |
| Golden / Oracle | 可信的参考实现或参考输出，用于验证新实现。 |
| Micro-benchmark | 只测一个 kernel 或小片段的性能测试。 |
| Roofline | 用计算量、内存流量与硬件峰值判断瓶颈的性能分析模型。 |

---

## 4. 问题背景：为什么需要 native MXFP4 kernel

### 4.1 L2 PLAIN 路径为什么很慢

MXFP4 物理 KV pool 保存的是：

```text
packed E2M1 bytes + 每 32 元素一个 E8M0 scale byte
```

而 FlashInfer 的既有路径需要 BF16 K/V。L2 的 `PLAIN` 做法是：每次 decode 前先把整层已缓存 K/V 反量化为新的 BF16 临时张量，再让 FlashInfer 读取它。

这条链路正确，但会产生：

```text
压缩 KV pool → 全量反量化 → 大型 BF16 临时 buffer → FlashInfer
```

它不仅重新读写大量数据，也无法安全使用 CUDA Graph 的缓冲复用，因此不是性能方案。

### 4.2 L3 native 路径改变了什么

L3 不再物化整层 BF16：

```text
packed KV pool → Triton stage 1 按 tile 读取 → 寄存器内解包/反量化 → attention
```

反量化值只在 kernel 的寄存器或局部 tile 中短暂存在。这样避免了全局 BF16 scratch buffer，也让 CUDA Graph 可以复用预分配的 split-KV 中间缓冲。

### 4.3 为什么先做 L1/L2

L3 内核同时涉及：量化格式、paged KV 寻址、attention、softmax、Triton 并行和性能优化。若没有 L2 golden，很难判断错误来自“量化数学”还是“内核实现”。

因此开发顺序是：

1. L1 锁定 FP8 attention 参考语义；
2. L2 锁定 MXFP4 codec 和端到端 attention 语义；
3. L3 只需对齐 L2 golden，专注内核性能。

---

## 5. 数据格式与内存布局

### 5.1 逻辑值到物理字节

每个连续的 32 个 head_dim 元素共享一个 E8M0 scale：

```text
logical values:  x[0], x[1], ..., x[31]
                     ↓ MXFP4 quantize
E2M1 codes:      c[0], c[1], ..., c[31]
scale:           s = 2^k
                     ↓ nibble pack
packed bytes:    [c1|c0], [c3|c2], ..., [c31|c30]
```

约定：

```text
packed_byte = low_nibble(even-index code) | high_nibble(odd-index code)
```

读取第 `d` 个元素时：

```text
byte_idx  = d // 2
block_idx = d // 32
code      = low nibble if d is even else high nibble
value     = E2M1(code) × E8M0(scale[block_idx])
```

### 5.2 本项目中最重要的张量形状

```text
q:          (batch, q_heads, head_dim)
k_packed:   (slots, kv_heads, ceil(head_dim / 2)) uint8
v_packed:   (slots, kv_heads, ceil(head_dim / 2)) uint8
k_scales:   (slots, kv_heads, ceil(head_dim / 32)) uint8
v_scales:   (slots, kv_heads, ceil(head_dim / 32)) uint8
```

`slots` 是 KV pool 的物理位置，不等同于逻辑 token 顺序。`kv_indices` 把某个请求的逻辑 token 映射到物理 slot，因而 kernel 不能假设历史 K/V 在内存中连续。

### 5.3 Qwen3.8 GQA 的直观例子

本次记录中的典型配置是：

```text
q_heads = 24
kv_heads = 4
head_dim = 256
kv_group_num = 24 / 4 = 6
```

因此一个 KV head 对应 6 个 Q heads；这正是 T1 grouped kernel 能大幅减少读取的原因。

---

## 6. 源码地图与调用链

| 文件 | 阅读目的 |
|---|---|
| `qwen38_mxfp4_kv_acceptance_records.md` | 验收、性能数据、优化证据与边界。 |
| `python/sglang/kernels/ops/attention/mxfp4_decode_attention.py` | native decode 的核心：T0、T1、T3、stage 2 和 host dispatch。 |
| `python/sglang/srt/layers/attention/triton_backend.py` | 生产 backend 如何识别 MXFP4、设置 splits、取得 raw KV 并调用 kernel。 |
| `python/sglang/kernels/ops/quantization/mxfp4_quant.py` | 融合量化 + 写入 KV pool 的 Triton kernel。 |
| `python/sglang/srt/layers/quantization/kvfp4_tensor.py` | eager codec；理解格式和作为 fused 写 kernel 的参考。 |
| `test/registered/attention/unittests/dense/test_triton_mxfp4_native_decode.py` | dequant 与 decode 的差分验证。 |
| `scripts/playground/bench_mxfp4_decode_kernel.py` | kernel 级参数 sweep 与性能验证。 |

生产 decode 调用链：

```text
TritonAttnBackend.forward_decode
  → get_raw_kv_buffer(layer_id)
  → mxfp4_decode_attention_fwd(...)
      → stage 1：按 split 解包 K/V + QKᵀ + online softmax + PV
      → stage 2：合并所有 split 的局部结果
```

---

## 7. 初版 T0：per-head split-KV decode

### 7.1 初版的目标

T0 的首要目标是正确：直接从 packed MXFP4 KV pool 做 decode attention，并对齐 L2 Torch golden。

其核心 kernel 是：

```python
_mxfp4_decode_stage1_kernel
```

逻辑 grid：

```text
(batch, q_head, split_kv_id)
```

也就是说，一个 program 处理：一个请求、一个 Q head、一个 KV 序列片段。

### 7.2 一个 program 的执行步骤

1. 由 `program_id` 得到当前 batch、Q head 和 split 编号。
2. 通过 `kv_indptr` 取得该请求的 KV 序列范围。
3. 将本 split 的 token 范围按 `BLOCK_N` 分成 tile。
4. 通过 `kv_indices` 找到每个逻辑 token 的物理 KV slot。
5. 读取 packed K/V，解出每个元素的 E2M1 code。
6. 读取每 32 元素对应的 scale，得到 fp32 K/V。
7. 计算 QKᵀ，更新 online softmax，再累计 PV。
8. 将局部 output 和 log-sum-exp 写入 stage 1 中间缓冲。

### 7.3 T0 的根本问题

在 GQA 中，同一 KV head 被多个 Q head 共用。初版却按 Q head 发 program：

```text
KV0 被 Q0、Q1、Q2、Q3、Q4、Q5 分别读取和解包
```

也就是同一压缩 K/V 被读取 6 次。长上下文时，这个冗余内存流量与解包工作成为主要问题，初版长文 TPOT 为 47.5 ms。

### 7.4 T0 逐行教学：`_mxfp4_decode_stage1_kernel`

本节逐行讲解初版 stage 1。源码：[`_mxfp4_decode_stage1_kernel`](python/sglang/kernels/ops/attention/mxfp4_decode_attention.py#L225-L406)，host 侧 launch 在 [`mxfp4_decode_attention_fwd`](python/sglang/kernels/ops/attention/mxfp4_decode_attention.py#L983-L1030)。格式沿用第 1 节约定：目标 → 前置知识 → 输入/输出 → 代码分段 → 手算示例 → 常见误解 → 小练习。

| 原清单问题 | 对应小节 |
|---|---|
| `program_id(0/1/2)` 如何映射到 batch/head/split？ | 7.4.3 |
| `tl.arange`、二维 broadcast 和 mask 的具体形状？ | 7.4.4 |
| paged KV 的三件套如何共同完成寻址？ | 7.4.6 |
| low/high nibble 解包与逻辑元素下标的对应？ | 7.4.7 |
| online softmax 为什么维护 `e_max`、`e_sum` 与 `acc`？ | 7.4.10 |
| 如何手算一个 `BLOCK_N=4` 的极小示例？ | 7.4.12 |

#### 7.4.1 目标与前置知识

读完本节，应能：

1. 不看源码复述一个 program 从 `program_id` 到写出 LSE 的完整流程；
2. 对任意 `(token, d)` 写出 packed byte 与 scale byte 的寻址公式；
3. 用一个 `BLOCK_N=4` 的小例子手算 online softmax 的每一步。

前置知识：第 5 节（packed 布局与 nibble 约定）；softmax 定义；Triton 最小概念——一次 launch 有一个 grid，grid 中每个 program 执行同一份代码，靠 `program_id` 认领自己的数据切片。

#### 7.4.2 输入、输出与编译期常量

运行期参数：

| 参数 | 形状/类型 | 含义 |
|---|---|---|
| `Q` | (batch, q_heads, head_dim) | decode 的 query |
| `K_Packed` / `V_Packed` | (slots, kv_heads, packed_dim) uint8 | packed E2M1 |
| `K_Scales` / `V_Scales` | (slots, kv_heads, num_blocks) uint8 | E8M0 scale byte |
| `sm_scale` | 标量 | 1/sqrt(head_dim) |
| `kv_indptr` | (batch+1,) int | CSR 行偏移 |
| `kv_indices` | (total_len,) int | 逻辑 token → 物理 slot |
| `Att_Out`（host 名 `attn_logits`） | (batch, q_heads, max_kv_splits, Lv) fp32 | stage 1 局部输出 scratch |
| `Att_Lse`（host 名 `attn_lse`） | (batch, q_heads, max_kv_splits) fp32 | stage 1 局部 LSE scratch |
| `num_kv_splits` | (batch,) int | 每个请求实际使用的 split 数 |

编译期常量（`tl.constexpr`，host 按取值生成专用机器码）：

| 常量 | Qwen3.8 典型值 | 含义 |
|---|---|---|
| `kv_group_num` | 6 | 每个 KV head 对应的 Q head 数 |
| `BLOCK_DMODEL` / `BLOCK_DV` | 256 / 256 | K/V 维 tile 宽 = next_pow2(head_dim/v_head_dim) |
| `PACKED_K_DIM` / `PACKED_V_DIM` | 128 | 每 token 的 packed 字节数 = ceil(dim/2) |
| `NUM_K_BLOCKS` / `NUM_V_BLOCKS` | 8 | block-32 的个数 = dim/32 |
| `BLOCK_N` | 64 | 每次 tile 迭代处理的 token 数 |
| `MIN_BLOCK_KV` | 32 | 每个 split 的最小 token 数 |
| `Lk` / `Lv` | 256 | 真实 head_dim / v_head_dim |
| `PAGE_SIZE` | 1 或 16 | 分页大小 |

其余 `stride_*` 形参是 host 传入的行距整数，kernel 只做乘加；正因为走 stride 而非假设连续，kernel 对非连续 buffer 也正确。注意 stage 1 **不写最终输出 `o`**：它只填 split-KV scratch，最终 `o` 由 stage 2 写出。scratch 由 host 用 `torch.empty` 分配——这一点在 7.4.5 会变成一个正确性要点。

#### 7.4.3 Grid 映射：一个 program 是谁

```python
cur_batch = tl.program_id(0).to(tl.int64)
cur_head = tl.program_id(1)
split_kv_id = tl.program_id(2)

cur_kv_head = cur_head // kv_group_num
```

- host 侧 `grid = (batch, head_num, max_kv_splits)`。以 batch=2、24 个 Q head、max_kv_splits=8 为例：program `(1, 7, 3)` 负责"第 2 个请求、第 8 个 Q head、第 4 个 KV 片段"。
- `cur_kv_head = cur_head // kv_group_num` 是 GQA 折叠：Q0..Q5 → KV0，Q6..Q11 → KV1，依此类推。同一 KV head 的 6 个 Q-head program 会各自独立读取并解包同一份 packed K/V——这正是 7.3 说的冗余来源。
- `.to(tl.int64)`：镜像 stock kernel 的扁平偏移防溢出保护。`Att_Out` 的扁平偏移量级为 batch × heads × splits × Lv，极端配置下会超出 int32。
- 阅读提示：T0 初版对 GQA 也走这个 grid；今天的 host 加了 `kv_group_num == 1` 的门，GQA 全部改走 T1 grouped kernel（第 8 节）。所以这个 kernel 在最终代码里仍然活着，但只服务 MHA——看 kernel 形状时要和"host 现在派给谁"分开。

#### 7.4.4 tile 下标、broadcast 与 mask

```python
offs_d = tl.arange(0, BLOCK_DMODEL)     # [256]：head_dim 方向的元素下标
offs_dv = tl.arange(0, BLOCK_DV)
mask_d = offs_d < Lk
mask_dv = offs_dv < Lv
byte_idx_k = offs_d // 2                # 元素 d 存在哪个 packed byte
blk_idx_k = offs_d // _MXFP4_BLOCK      # 元素 d 用哪个 scale block
```

| 变量 | 形状 | 值域 | 用途 |
|---|---|---|---|
| `offs_d` | [256] | 0..255 | head_dim 方向元素下标 |
| `mask_d` | [256] | head_dim=256 时全 True | 屏蔽 next_pow2 补出的 padding 维 |
| `byte_idx_k` | [256] | 0..127，每值出现 2 次 | packed byte 列偏移 |
| `blk_idx_k` | [256] | 0..7，每值重复 32 次 | scale byte 列偏移 |

- `tl.arange(0, N)` 产生 [0..N-1] 的一维 int32 张量；Triton 中后续所有运算都是"整块"的 SIMD 语义，没有逐元素循环。
- 为什么需要 mask：`BLOCK_DMODEL = next_pow2(Lk)`。head_dim=256 已是 2 的幂，mask 全 True；但 head_dim=192 时 BLOCK=256，多出的 64 条 lane 必须被屏蔽，否则越界读写、污染 QKᵀ。
- 二维 tile 的产生：后面 `k_row[:, None] + byte_idx_k[None, :]` 把 [BLOCK_N] 的行基址与 [BLOCK_DMODEL] 的列偏移按广播规则组合成 [BLOCK_N, BLOCK_DMODEL] 地址矩阵：

```text
              byte_idx_k[0]   byte_idx_k[1]   ...  byte_idx_k[127]
k_row[0]   →  addr(tok0,d0)  addr(tok0,d1)        addr(tok0,d127)
k_row[1]   →  addr(tok1,d0)  ...
...
k_row[63]  →  addr(tok63,d0) ...                 addr(tok63,d127)
```

- 双层保险：padding lane 上 load 的 `other=0` 使 code=0、magnitude=0，随后 `k = tl.where(mask_d[None, :], k, 0.0)` 再清一次；Q 侧同样 `other=0.0`。三层任意一层失效都不会污染分数。

#### 7.4.5 请求的 KV 范围与 split 切分

```python
cur_batch_kv_start_idx = tl.load(kv_indptr + cur_batch)
cur_batch_seq_len = tl.load(kv_indptr + cur_batch + 1) - cur_batch_kv_start_idx
kv_splits = tl.load(num_kv_splits + cur_batch)

kv_len_per_split = (
    tl.cdiv(tl.cdiv(cur_batch_seq_len, kv_splits), MIN_BLOCK_KV) * MIN_BLOCK_KV
)
split_kv_start = kv_len_per_split * split_kv_id
split_kv_end = tl.minimum(split_kv_start + kv_len_per_split, cur_batch_seq_len)
```

- `kv_indptr` 是 CSR 行偏移：`[0, 5, 8, 15]` 表示三个请求长度 5/3/7。每个 program 只 load 自己 batch 的两个元素。
- `tl.cdiv(a, b)` = ceil(a/b)。例：seq_len=1000、kv_splits=8 → cdiv(1000,8)=125 → cdiv(125,32)=4 → 每 split 128 个 token；split 7 的区间是 [896, min(1024, 1000)) = [896, 1000)，被 clamp。
- 对齐的副作用：seq_len=64、kv_splits=8 时每 split = 32，split 0/1 覆盖 [0,32)/[32,64)，split 2..7 start≥64 → **空区间**。即实际工作的 split 可以少于申请值。
- 区分为空的方式：grid 第三维按 `max_kv_splits` 铺满，而每个 batch 实际只应工作 `kv_splits` 个；对 `split_kv_id ≥ kv_splits` 的 program，`start = len × id ≥ len × kv_splits ≥ seq_len` 自然落空，由 `if split_kv_end > split_kv_start:` 整段跳过。
- 正确性要点：空 split 不写 scratch，其槽位保持 `torch.empty` 的**未初始化垃圾**。stage 2 必须用完全相同的切分公式（逐字一致，含 `MIN_BLOCK_KV`）重算区间并跳过垃圾槽位——两处公式若有一处改动不同步，垃圾值就会混进结果。
- 为什么对齐 32：保证每个干活的 split 至少 32 个 token，避免产生只有几个 token 的碎 split（纯增 stage 2 合并成本而无并行收益）。

#### 7.4.6 paged KV 寻址：kv_indptr → kv_indices → 行基址

三步：逻辑 token 下标 → 物理 slot → 内存行基址。

```python
for start_n in range(split_kv_start, split_kv_end, BLOCK_N):
    offs_n = start_n + tl.arange(0, BLOCK_N)   # 本 tile 的逻辑 token 下标
    tok_mask = offs_n < split_kv_end
    kv_loc = tl.load(
        kv_indices + cur_batch_kv_start_idx + offs_n,
        mask=tok_mask,
        other=0,
    )
```

- `offs_n` 是**逻辑**位置（请求序列内第几个 token），不是地址。`kv_indices` 是全 batch 共享的平铺 gather 表，用 `cur_batch_kv_start_idx + offs_n` 定位到本请求的段。
- 小例子：batch=3、seq_lens=[5,3,7]、kv_indptr=[0,5,8,15]：

```text
kv_indices = [10, 11, 12, 13, 14,  99,  3, 200,  50, 51, 52, 53, 54, 55, 56]
              └──── req 0 ───┘   └ req 1 ┘   └──────── req 2 ────────┘
```

  batch 1 的 program 读到 kv_loc=[99, 3, 200]：该请求 3 个 token 的 K/V 散落在三个 slot。radix cache 分配下物理布局通常乱序，kernel 不能假设连续。
- mask 语义：被 mask 的 lane **不发起内存访问**，`other=0` 只是填充值；无效 lane 得到 kv_loc=0，派生地址指向 slot 0，但地址计算本身安全，且这些 lane 的分数随后被置 -inf，值从不参与结果。

slot → 行基址：

```python
if PAGE_SIZE == 1:
    k_row = kv_loc * stride_kp_bs + cur_kv_head * stride_kp_h
    v_row = kv_loc * stride_vp_bs + cur_kv_head * stride_vp_h
else:
    page_id = kv_loc // PAGE_SIZE
    tok_in_p = kv_loc % PAGE_SIZE
    k_row = (
        page_id * stride_kp_page + tok_in_p * stride_kp_tok
        + cur_kv_head * stride_kp_h
    )
    v_row = (同构)
ks_row = kv_loc * stride_ks_bs + cur_kv_head * stride_ks_h   # scale 始终 slot 制
vs_row = kv_loc * stride_vs_bs + cur_kv_head * stride_vs_h
```

- 四个 stride 来自 host 的 [`_extract_kv_strides`](python/sglang/kernels/ops/attention/decode_attention.py#L197-L210)：slot_stride = buf.stride(0)，head_stride = buf.stride(1)，page_stride = slot_stride × page_size，tok_stride = slot_stride。
- 关键事实（kernel 注释原话）：对当前 `(slots, kv_heads, packed_dim)` 平铺 pool，page/tok 公式是**恒等式**：

```text
page_id·(slot_stride·P) + tok_in_p·slot_stride
= ((kv_loc // P)·P + kv_loc % P)·slot_stride
= kv_loc·slot_stride
```

  保留分支是为了与 stock decode kernel 结构逐字对齐，将来换真正的分页 pool 布局时 kernel 不必改。`PAGE_SIZE` 是 constexpr，==1 时编译器直接删除 else 分支，无运行时开销。测试对此的验证方式：`page_size=16` 的 case 直接复用平铺 buffer，考察的正是"公式退化为恒等式"这一事实。
- scale buffer 恒为 `(slots, kv_heads, num_blocks)` 平铺布局，所以始终走 slot 制寻址，没有 page 分支。

#### 7.4.7 K tile 解包：nibble → E2M1 → E8M0 → fp32

```python
kbyte = tl.load(
    K_Packed + k_row[:, None] + byte_idx_k[None, :],
    mask=tok_mask[:, None] & (byte_idx_k[None, :] < PACKED_K_DIM),
    other=0,
)
is_odd_k = (offs_d % 2) == 1
kcode = tl.where(is_odd_k[None, :], (kbyte >> 4) & 0x0F, kbyte & 0x0F)
```

- tile 形状 [BLOCK_N=64, BLOCK_DMODEL=256] uint8，第 (n, d) 格 = token n 的元素 d 所在 byte。`byte_idx_k < PACKED_K_DIM` 这半个 mask 在 head_dim 非 2 幂时才生效（如 head_dim=96 → BLOCK=128、PACKED=48，后 32 列无 byte）。
- 解包约定（第 5 节）：**相邻配对**——偶数下标元素在 low nibble，奇数下标在 high nibble：

```text
byte 0x92 = [9][2]
  low  nibble 2 → 元素 d（偶数）的 E2M1 code
  high nibble 9 → 元素 d+1（奇数）的 E2M1 code
```

E2M1 code → fp32 数值：

```python
kmant = (kcode & 0x1).to(tl.float32)    # bit 0：尾数 m
kexp = (kcode >> 1) & 0x3               # bit 1..2：指数 e
knormal = tl.exp2(kexp.to(tl.float32) - 1.0) * (1.0 + 0.5 * kmant)
ksub = 0.5 * kmant                      # e == 0 的 subnormal
kmag = tl.where(kexp == 0, ksub, knormal)
k = tl.where((kcode & 0x8) != 0, -kmag, kmag)   # bit 3：符号
k = tl.where(mask_d[None, :], k, 0.0)
```

code（0..7）到数值的完整表（8..15 为对应相反数，含 -0）：

| code | 位 s e e m | 数值 | code | 位 | 数值 |
|---|---|---|---|---|---|
| 0 | 0000 | 0 | 4 | 0100 | 2.0 |
| 1 | 0001 | 0.5 | 5 | 0101 | 3.0 |
| 2 | 0010 | 1.0 | 6 | 0110 | 4.0 |
| 3 | 0011 | 1.5 | 7 | 0111 | 6.0 |

表达集合就这 16 个值：没有 0.25、5、8。e≥1 的公式是 2^(e-1)·(1+0.5m)；e=0 时 0.5m（subnormal：0 或 0.5）。

scale gather 与相乘：

```python
ksbytes = tl.load(
    K_Scales + ks_row[:, None] + blk_idx_k[None, :],
    mask=tok_mask[:, None] & (blk_idx_k[None, :] < NUM_K_BLOCKS),
    other=0,
)
ksf = _e8m0_to_f32(ksbytes)
k = k * ksf
```

- scale tile 形状 [BLOCK_N, 8]；列 d 处因 `blk_idx_k` 的重复而每个 scale 覆盖 32 个元素。
- [`_e8m0_to_f32`](python/sglang/kernels/ops/attention/mxfp4_decode_attention.py#L43-L57) 为什么不用 `tl.exp2(b - 127)`：`ex2` 硬件近似会 FTZ，把 byte 0（2^-127）静默清零；bit 构造 `byte << 23`（byte 0 特判 subnormal 模式 0x00400000，byte 255 → quiet NaN）逐 bit 精确。这是 T0 中已修过的真实 bug；第 10 节（T3）再把同样的思想推到热路径。
- 性能备注：这条"移位/掩码/exp2/两次乘"的元素级指令链每元素每 tile 都要走一遍，且 fp32 K tile 形状 [64, 256] 的寄存器压力很大。T0 先建立"正确"；T1 用 `tl.dot` + BF16 tile 重构计算，T3 再把解包本身换成整数位构造。

#### 7.4.8 QKᵀ 与 -inf mask

```python
qk = tl.sum(q[None, :] * k, 1)      # [1,256]×[64,256] → 沿 dim1 求和 → [64]
qk *= sm_scale
qk = tl.where(tok_mask, qk, float("-inf"))
```

- T0 没有 `tl.dot`：每个 program 的 Q 只有一行，广播乘 + 沿 head_dim 求和等价于向量-矩阵积。这也是它用不上 Tensor Core 的原因（T1 引入二维 Q tile 后才由 `tl.dot` 接管）。
- `-inf` 的作用：exp(-inf)=0，让无效 token 在归一化与 PV 累加中都**精确**贡献 0，而不是近似的极小量。
- `sm_scale` 在求和后、取指数前乘，与 softmax(q·kᵀ/sqrt(d)) 的数学顺序一致。

#### 7.4.9 V tile 解包与 PV 累加

V 的解包与 K 完全同构（`offs_dv` / `byte_idx_v` / `blk_idx_v` / `vsf`），代码顺序即数据依赖顺序：K 解包必须先于 QKᵀ，V 解包只被 PV 需要，所以排在 QKᵀ 之后。PV 在 online softmax 更新内完成：

```python
acc += tl.sum(p[:, None] * v, 0)    # [64,1]×[64,256] → 沿 token 维求和 → [256]
```

`p[:, None]` 把每 token 权重升为列，与 V tile 逐行相乘后沿 token 维求和——加权平均的"部分和"，[BLOCK_DV] 一维向量，与 7.4.2 中 `acc` 的形状一致。

#### 7.4.10 online softmax：为什么维护 e_max / e_sum / acc

标准 softmax 的输出是 o = Σ_t e^{s_t}·v_t / Σ_t e^{s_t}，分子分母都需要全序列 score。流式处理 tile 时看不到后面的 max，所以维护"截至当前的部分量"，并用新 max 修正旧量：

```python
n_e_max = tl.maximum(tl.max(qk, 0), e_max)   # 历史 max ∪ 本 tile max
re_scale = tl.exp(e_max - n_e_max)           # 旧量按旧 max 存的，需缩回
p = tl.exp(qk - n_e_max)                     # 本 tile 按新 max
acc *= re_scale                              # 修正旧 PV 部分和
acc += tl.sum(p[:, None] * v, 0)
e_sum = e_sum * re_scale + tl.sum(p, 0)      # 修正旧 exp 部分
e_max = n_e_max
```

循环不变量（每轮迭代结束时成立）：

```text
acc   = Σ_{已见 t} e^{s_t - e_max} · v_t
e_sum = Σ_{已见 t} e^{s_t - e_max}
```

于是 `acc / e_sum` 恰为该 split 的真 softmax 输出（分子分母的 e^{-e_max} 因子约掉）；`e_max + log(e_sum)` 是本 split 的 log-sum-exp，stage 2 靠它做跨 split 的 max 重缩放（第 9 节展开）。

数值要点：

- `qk - n_e_max ≤ 0` 恒成立 → p ∈ (0, 1]，永不 overflow；
- 首轮 `e_max = -inf`：`re_scale = exp(-inf - 有限) = 0`，`acc × 0`、`e_sum × 0` 无害。NaN 需要 `-inf - (-inf)`，而循环内每个 tile 至少一个有效 token（进入循环体的条件就是 `start_n < split_kv_end`），n_e_max 必有限；
- split 结束时冻结的 `(acc, e_sum, e_max)` 正是 stage 2 的合并输入——online softmax 与 split-KV 是同一套数学在两个粒度上的复用。

#### 7.4.11 写出局部 output 与 LSE：`// Lv` 技巧

```python
offs_mid_o = (
    cur_batch * stride_mid_ob + cur_head * stride_mid_oh
    + split_kv_id * stride_mid_os + offs_dv
)
tl.store(Att_Out + offs_mid_o, acc / e_sum, mask=mask_dv)

offs_mid_o_1 = (
    cur_batch * stride_mid_ob + cur_head * stride_mid_oh
    + split_kv_id * stride_mid_os
) // Lv
tl.store(Att_Lse + offs_mid_o_1, e_max + tl.log(e_sum))
```

- `Att_Out` 是 (batch, head, max_kv_splits, Lv) 的 fp32 scratch，store 的 `mask=mask_dv` 防止 BLOCK_DV > Lv 时越界。
- `// Lv` 技巧：前三项是在"带 Lv 尾维"的扁平坐标系里算出的偏移，整除 Lv 恰好折掉尾维：

```text
attn_logits (b, h, s, Lv) 连续时 stride = (h·S·Lv, S·Lv, Lv)
(b·h·S·Lv + h·S·Lv + s·Lv) // Lv = b·h·S + h·S + s
而 attn_lse (b, h, S) 的扁平下标正是 b·h·S + h·S + s
```

  两份 scratch 形状差一个尾维，用一次整除就能共用同一组 stride 形参，host 不必再传三个 LSE stride。backend 对此有专门注释：`attn_logits.shape[-1]` 必须精确等于该层 `v_head_dim`，否则这个技巧失效（`triton_backend.py` 选 buffer 处）。
- 写出的是**局部**结果：`acc / e_sum` 是"只看本 split token"的 softmax 输出，LSE 也只是本 split 的 log-sum-exp；跨 split 合并是 stage 2 的职责。
- 空 split 不写：其槽位保持垃圾值，由 stage 2 的相同公式跳过（7.4.5）。

#### 7.4.12 手算示例：BLOCK_N=4 完整走一遍

设定（全部缩到能口算的规模；BLOCK_N 人为取 4，生产默认 64）：

```text
batch=1, q_heads=1, kv_heads=1（MHA，正是 host 会路由给本 kernel 的情形）
head_dim = 4 → BLOCK_DMODEL=4, PACKED_K_DIM=2, NUM_K_BLOCKS=1
seq_len=6, kv_splits=1 → kv_len_per_split = cdiv(cdiv(6,1),32)·32 = 32
  split 0 区间 = [0, min(32,6)) = [0,6)，tile 循环 range(0,6,4)：两轮
kv_indices = [2, 5, 0, 7, 3, 1]   # 逻辑 token 的物理 slot（故意乱序）
sm_scale = 1.0
```

K/V 数据（全部 E2M1×2^0 可精确表示；scale byte 全为 127 → 2^0 = 1）：

| 逻辑 t | slot | K | V |
|---|---|---|---|
| 0 | 2 | [1.0, -0.5, 2.0, 0.0] | [1, 0, 0, 0] |
| 1 | 5 | [0.0, 1.0, -1.5, 0.5] | [0, 1, 0, 0] |
| 2 | 0 | [-2.0, 0.0, 1.0, 0.5] | [0, 0, 1, 0] |
| 3 | 7 | [0.5, 0.5, 0.5, 0.5] | [0, 0, 0, 1] |
| 4 | 3 | [1.0, 1.0, 1.0, 1.0] | [1, 1, 0, 0] |
| 5 | 1 | [0.0, 0.0, 2.0, -2.0] | [0, 0, 1, 1] |

对应 packed 字节与 Q（验证解包规则用；low nibble = 偶数下标元素）：

```text
slot 2 K: byte0 = 0x92, byte1 = 0x04
  0x92: low=2→1.0, high=9→-0.5； 0x04: low=4→2.0, high=0→0.0
slot 5 K: byte0 = 0x20, byte1 = 0x1B
  0x1B: low=11→-1.5, high=1→0.5
Q = [1.0, 2.0, 0.5, -1.0]
```

解包验证 slot 5 的 d2：byte1=0x1B，d2 偶 → low nibble 11 = 0b1011 → s=1, e=(11>>1)&3=1, m=1 → 2^0·(1+0.5)=1.5 → 取负 = -1.5，与上表一致。

第一轮 tile（token 0..3；gather kv_loc=[2,5,0,7] 后解出上表 K）：

```text
QKᵀ:  qk[0] = 1·1 + 2·(-0.5) + 0.5·2 + (-1)·0   = 1.00
      qk[1] = 1·0 + 2·1    + 0.5·(-1.5) + (-1)·0.5 = 0.75
      qk[2] = 1·(-2) + 2·0 + 0.5·1 + (-1)·0.5  = -2.00
      qk[3] = 1·0.5 + 2·0.5 + 0.5·0.5 + (-1)·0.5 = 1.25

online softmax（初态 e_max=-inf, e_sum=0, acc=[0,0,0,0]）:
  n_e_max  = max(1.25, -inf) = 1.25
  re_scale = exp(-inf - 1.25) = 0
  p        = exp([1.00, 0.75, -2.00, 1.25] - 1.25)
           = [0.7788, 0.6065, 0.0388, 1.0000]
  acc      = 0·0 + p·V = [0.7788, 0.6065, 0.0388, 1.0000]   # V 是 one-hot，逐位即 p
  e_sum    = 0 + 2.4241 = 2.4241
  e_max    = 1.25
```

第二轮 tile（token 4..7；d6、d7 被 tok_mask 置 -inf）：

```text
qk = [2.50, 3.00, -inf, -inf]
  token4(slot3): 1+2+0.5-1 = 2.5；token5(slot1): 0+0+0.5·2+(-1)(-2) = 3.0
n_e_max  = max(3.0, 1.25)  = 3.0
re_scale = exp(1.25 - 3.0) = 0.1738
p        = exp([2.5, 3.0, -inf, -inf] - 3.0) = [0.6065, 1.0000, 0, 0]
acc      = [0.7788, 0.6065, 0.0388, 1.0]·0.1738
           + 0.6065·[1,1,0,0] + 1·[0,0,1,1]
         = [0.7419, 0.7119, 1.0067, 1.1738]
e_sum    = 2.4241·0.1738 + 1.6065 = 2.0278
e_max    = 3.0
```

写出：

```text
局部 output = acc / e_sum = [0.3659, 0.3511, 0.4965, 0.5790]
局部 LSE    = 3.0 + ln(2.0278) = 3.7069
```

交叉验证（把 6 个 token 当整体做标准 softmax，应逐位一致）：

```text
scores     = [1.0, 0.75, -2.0, 1.25, 2.5, 3.0]
exp(s-3.0) = [0.1353, 0.1054, 0.0067, 0.1738, 0.6065, 1.0]，Σ = 2.0278
output     = Σ w_t·V_t = [0.3659, 0.3511, 0.4965, 0.5790]  ✓
LSE        = ln(2.0278·e^3) = 3.7069                        ✓
```

这个例子同时验证三件事：乱序 slot 的 gather 正确、-inf mask 精确贡献 0、re_scale 重缩放后的部分和等于全局 softmax。

#### 7.4.13 常见误解

| 误解 | 事实 |
|---|---|
| `offs_n` 是物理地址 | 是逻辑 token 下标；物理 slot 由 `kv_indices` gather 得到 |
| masked load 的 `other=0` 会去读 slot 0 | masked lane 不发起内存访问，`other` 只是填充值；下游再用 -inf 把它剔除 |
| 写出的 `acc / e_sum` 是最终 attention 输出 | 只是本 split 的局部结果，必须经 stage 2 合并 |
| 空 split 的 scratch 槽位是 0 | 是 `torch.empty` 的未初始化垃圾；stage 2 靠逐字相同的切分公式跳过 |
| `mask_d` 在 head_dim=256 时也起作用 | head_dim 为 2 的幂时全 True；它服务 head_dim=192 之类的非 2 幂形状 |
| nibble 是"高 nibble 存前一半元素" | 相邻配对：low = 偶数下标，high = 奇数下标（与 NVFP4 硬件约定对齐） |
| 首轮 `acc × exp(-inf - x)` 会得 NaN | 得 0；NaN 需要 `-inf - (-inf)`，而每个 tile 至少一个有效 token |
| T0 kernel 已被删除 | 还在：最终 host 只把 MHA（kv_group_num==1）路由给它，无 GQA 冗余，且保留精确 fp32 数学 |

#### 7.4.14 小练习

1. 取 seq_len=40、kv_splits=2：`kv_len_per_split` 是多少？两个 split 的 token 范围各是什么？（注意 `MIN_BLOCK_KV` 对齐的副作用。）再手算两个局部 output 与 LSE，用第 9 节的 stage 2 合并公式验证合并结果等于全局 softmax。
2. 令 head_dim=6（则 BLOCK_DMODEL=8、PACKED_K_DIM=3、NUM_K_BLOCKS=1），写出元素 d=5 的 `byte_idx_k`、`blk_idx_k`、所在 nibble，以及该 nibble 所在的 packed byte 下标。哪些 lane 会被哪层 mask 掉？
3. 只用 7.4.7 的表，手算 packed 序列 `0x1B, 0xC4`（scale byte 128）解出的 4 个 K 值。
4. 对照源码：找出 `_mxfp4_decode_stage1_kernel` 与 stock [`_fwd_kernel_stage1`](python/sglang/kernels/ops/attention/decode_attention.py#L220-L404) 的 5 处结构差异，并判断每处来自"MXFP4 数据格式"还是"功能裁剪"（logit cap、score_mod、xai temperature 等在 T0 中不存在）。

> 读完 7.4，即可进入第 8 节 T1：同一份数据如何从"每 Q head 一份 program"变成"每 KV head 一份 program"，以及 `tl.dot` 在哪里接管了 7.4.8 的广播乘-求和。

---

## 8. 优化 T1：grouped kernel 消除 GQA 重复读取

### 8.1 改变 program 的“工作所有权”

优化后的 kernel：

```python
_mxfp4_grouped_decode_stage1_kernel
```

不再让一个 program 只服务一个 Q head；它让一个 program 持有一个 KV head，并同时服务该 head 对应的一组 Q heads。

对于 24 Q / 4 KV：

```text
T0：每个 KV head 对应 6 个独立 Q-head program，K/V 解包 6 次
T1：每个 KV head 对应 1 个 grouped program，K/V 解包 1 次
```

### 8.2 矩阵化计算

T1 中：

```text
Q: [group_heads, head_dim]
K: [token_tile, head_dim]
V: [token_tile, value_dim]
```

计算变为：

```text
QKᵀ: [group_heads, head_dim] × [head_dim, token_tile]
PV:  [group_heads, token_tile] × [token_tile, value_dim]
```

`tl.dot` 把这些计算变为矩阵乘法，从而使用 Tensor Core。K/V 只需要解包一次，随后多个 Q heads 共享。

### 8.3 精度变化为何可接受

T0 主要使用 fp32 元素级运算；T1 将解包后的 K/V 以 BF16 tile 交给 `tl.dot`。MXFP4 的 E2M1 值乘以 2 的幂可被 BF16 精确表示，因此反量化本身不引入额外舍入；可观察的差异主要来自 Tensor Core 的累加顺序及最终 BF16 输出舍入。

验证基准使用 L2 golden，grouped 路径的误差仍处于冻结阈值和服务级硬上限内。

### 8.4 【待深入展开】T1 逐行教学

- [ ] `BLOCK_H`、`VALID_BLOCK_H`、`mask_h` 分别解决什么问题？
- [ ] 为什么 Q 是二维 tile 而 K/V 仍只加载一份？
- [ ] `tl.dot(q, tl.trans(k))` 的输入/输出形状如何推导？
- [ ] 读一次 K/V 如何服务 6 个 Q heads？
- [ ] grouped kernel 为什么降低流量却会降低 CTA 数？
- [ ] 怎样从误差指标区分“反量化错误”和“累加顺序差异”？

---

## 9. 优化 T2：提高 split 数填满 GPU

### 9.1 T1 带来的新瓶颈

grouped kernel 将同一 KV head 的多个 Q-head program 合并，减少了冗余工作；但也减少了可并行的 CTA 数。

在 batch=1、4 个 KV head、splits=8 时：

```text
grid = 1 × 4 × 8 = 32 CTA
```

目标 GPU 有约 170 个 SM，32 个 CTA 不足以让多数 SM 同时工作。此时 kernel 是 **latency-bound**：不是 DRAM 总带宽已耗尽，而是没有足够的独立工作来掩盖单个 program 的等待时间。

### 9.2 解决：最少 32 个 KV splits

将最小 `max_kv_splits` 从默认 8 提升至 32：

```text
grid = 1 × 4 × 32 = 128 CTA
```

更多 split 增加了 stage 1 的并行度；stage 2 则负责将更多局部结果合并。生产 backend 对 MXFP4 native decode 设置最小值为 32，同时保留显式更大配置的优先级。

### 9.3 效果与权衡

- 长文服务 TPOT：30.5 → 25.7 ms。
- 4096 序列的 kernel 级时间：155.5 → 66.7 µs（包含 T1 + T2 的组合效果）。
- split 不能无限增加：会增大 stage 1 中间 buffer、写入次数与 stage 2 合并成本。

### 9.4 【待深入展开】T2 逐行教学

- [ ] CTA、SM、occupancy、latency-bound、bandwidth-bound 的区别。
- [ ] `kv_len_per_split` 如何计算，为什么对齐到 `MIN_BLOCK_KV`？
- [ ] 为什么每个 split 必须单独保存 output 与 LSE？
- [ ] stage 2 如何在数值上正确合并多个局部 softmax？
- [ ] 怎样设计 sweep，找出最优 split 数而不是盲目增大？

---

## 10. 优化 T3：位构造反量化

### 10.1 初版反量化的热点

初版对每个 E2M1 code 计算浮点 magnitude，并执行：

```text
E2M1 magnitude × E8M0 scale
```

这包含 `exp2` 和 fp32 乘法链。`exp2` 使用 SFU，且在极小 scale（如 2^-127）上会遇到 FTZ 风险。

### 10.2 关键观察

E8M0 scale 永远是 2 的幂；E2M1 只有有限的指数与 1-bit 尾数。两者相乘的结果可精确表示为 BF16 的 bit pattern。

因此不需要真正执行：

```text
exp2 + 浮点乘法
```

而是可以：

```text
读取 E2M1 code / E8M0 byte
→ 整数移位、掩码、条件选择
→ 拼出 BF16 的 sign / exponent / mantissa bits
→ bitcast 成 BF16
```

### 10.3 两个 helper 的分工

- `_e8m0_to_f32`：精确构造 E8M0 scale 的 fp32 表示，处理 byte 0 的 subnormal 与 byte 255 的 NaN。
- `_e2m1_scale_to_bf16`：在 grouped kernel 的热路径中直接构造 `E2M1 × E8M0` 的 BF16 值。

### 10.4 效果

- 4096 序列：68.9 → 63.8 µs，约快 7%。
- 8192 序列：98.4 → 88.1 µs，约快 10%。
- 服务级改善约 0.1 ms，已接近测量噪声。

服务收益比 kernel 微基准小，是因为 attention kernel 只占整步 TPOT 的很小部分。

### 10.5 【待深入展开】T3 逐行教学

- [ ] E2M1 四个 bit 如何拆成 sign、exponent、mantissa？
- [ ] 为什么 E2M1 × 2 的幂能无损落到 BF16？
- [ ] `bitcast` 与普通 `.to(tl.bfloat16)` 的差别是什么？
- [ ] normal、subnormal、zero、Inf、NaN 的优先级怎样处理？
- [ ] `tl.exp2(-127)` 的 FTZ 问题如何复现和验证？
- [ ] 为什么 dequant micro-kernel 仍可保留易读的浮点实现？

---

## 11. 实验分支：为什么 `tl.dot_scaled` 默认关闭

`tl.dot_scaled` 可将 packed FP4 和 scale 直接输入支持 block-scaled FP4 MMA 的硬件路径。单独测 QKᵀ 时，它比手动解包快约 34%。

但完整 decode loop 中它反而慢约 2.2 倍，默认关闭。原因：

1. QKᵀ 的右侧 tile 很小，硬件固定开销难以摊薄；
2. V 的 scale 沿输出维，而 `dot_scaled` 需要的 scale 语义沿归约维，PV 不能复用该路径；
3. 全链路仍有 V 解包、softmax、PV、stage 2，局部 QKᵀ 变快不足以弥补其它成本。

> 经验：micro-benchmark 表明“一个部件可能快”，端到端 benchmark 才能说明“系统真的快”。

### 【待深入展开】`tl.dot_scaled` 实验课

- [ ] FP4 MMA 与普通 BF16 MMA 的输入格式差异。
- [ ] “scale 沿归约维”和“scale 沿输出维”分别是什么意思？
- [ ] 如何解读 QK-only probe 与完整 decode benchmark 的相反结论？
- [ ] 哪些未来模型形状可能让该分支重新值得启用？

---

## 12. 正确性、性能与价值边界

### 12.1 正确性不是只看最终输出

L3 验证分层进行：

1. dequant 微内核对独立 OCP oracle 元素精确；
2. standalone native decode 对 L2 Torch golden 差分；
3. 真实 backend 集成差分；
4. 融合写 kernel 对 eager codec 与 oracle bit-exact；
5. CUDA Graph capture/replay 字节确定性；
6. 真实模型 layer dump 回放。

这套顺序的价值是：出现问题时能明确知道错误在编码、反量化、attention、集成，还是 CUDA Graph 复用，而不是只看到一个最终误差数字。

### 12.2 为什么 MXFP4 没有显著超过 FP8

该 workload 中，每一步 decode 的主要 DRAM 流量是模型权重，约 21.5 GB；KV cache 仅约：

| 项目 | FP8 KV | MXFP4 KV |
|---|---:|---:|
| KV 读取 | 128 MiB | 68 MiB |
| 估算时间 | 71 µs | 38 µs |
| 理论节省 | — | 约 33 µs |

因此，MXFP4 省下的 KV 带宽只占 23.8 ms TPOT 的极小部分；同时还需要支付解包与 stage 2 的计算成本。

当前价值定位是：

```text
单卡、小 batch、该 hybrid 模型：
  容量提升 1.52× + decode 速度接近 FP8

大 batch × 长上下文、KV 占带宽比例更高：
  才更可能兑现明显吞吐优势
```

### 12.3 需要持续区分的三种性能数字

| 指标 | 能回答的问题 | 不能单独证明什么 |
|---|---|---|
| kernel micro-bench | 某个 kernel/参数是否变快 | 整个服务是否变快 |
| 每层 kernel 时间线 | attention 在一层中占多少 | 权重/GDN/launch 的端到端成本 |
| TPOT / 吞吐 | 用户实际感受到的服务性能 | 具体哪条指令带来收益 |

---

## 13. 后续深入教学占位

以下章节预留为后续分段扩展的教学单元。扩写时应保持“小而完整”：一次只讲一个概念，附最小形状示例和对应源码区间。

### 单元 A：Triton 最小入门

- [ ] `@triton.jit`、`tl.constexpr`、grid、program、warp 的关系。
- [ ] `tl.arange`、broadcast、mask 与 pointer arithmetic。
- [ ] 从一个向量加法 kernel 过渡到本项目的 nibble unpack。

### 单元 B：MXFP4 codec 手算课

- [ ] E2M1 code 到数值的完整表。
- [ ] E8M0 byte 到 scale 的完整规则。
- [ ] block-32、amax、RNE（Round to Nearest, Even）和特殊值。
- [ ] 两个 nibble 打包/解包的手算例子。

### 单元 C：paged KV 寻址课

- [ ] `slot`、`page`、`token-in-page` 的区别。
- [ ] `kv_indptr` 与 `kv_indices` 的小型例子。
- [ ] `PAGE_SIZE == 1` 与分页路径的地址公式。

### 单元 D：T0 stage 1 逐行课

> 主体内容已由 7.4 承接（7.4.3–7.4.11 逐行、7.4.12 手算、7.4.13 误解清单），此处保留作复习索引。

- [ ] 一个 program 的所有变量和张量形状。
- [ ] K/V 解包与 scale gather。
- [ ] QKᵀ、mask、online softmax、PV。
- [ ] 局部 output/LSE 的写出。

### 单元 E：split-KV 与 stage 2 数学课

- [ ] 为什么普通 softmax 不能直接分块平均。
- [ ] 局部 max、局部 sum、局部 output 的合并公式。
- [ ] stage 2 逐行推导。

### 单元 F：T1 grouped kernel 逐行课

- [ ] GQA 的冗余读取如何从 grid 设计中产生。
- [ ] `BLOCK_H` 和 `mask_h`。
- [ ] `tl.dot` 的矩阵形状与 Tensor Core 路径。
- [ ] 精度阈值的由来与解释。

### 单元 G：T2 GPU 并行度课

- [ ] CTA/SM/occupancy 的直观模型。
- [ ] 为什么 grouped 后需要更多 split。
- [ ] 如何设计 split、`BLOCK_N`、`num_warps` 的参数 sweep。

### 单元 H：T3 位构造课

- [ ] FP32/BF16 位布局复习。
- [ ] E2M1 × E8M0 → BF16 的推导。
- [ ] subnormal、FTZ、NaN/Inf 的边界处理。
- [ ] `bitcast` 代码逐行解释。

### 单元 I：融合写 kernel 课

- [ ] `amax → scale → RNE → nibble pack → scatter store` 流水线。
- [ ] 为什么 slot 0 必须保留。
- [ ] eager fallback 的适用范围。
- [ ] CUDA Graph 下避免 host sync 的原则。

### 单元 J：验证与性能工程课

- [ ] oracle、golden、差分测试、冻结阈值。
- [ ] micro-bench、服务 benchmark、nsys 时间线的分工。
- [ ] roofline 带宽账本。
- [ ] 如何判断一个优化值得合入。

---

## 参考入口

- 验收记录：[`qwen38_mxfp4_kv_acceptance_records.md`](qwen38_mxfp4_kv_acceptance_records.md)
- native decode：[`python/sglang/kernels/ops/attention/mxfp4_decode_attention.py`](python/sglang/kernels/ops/attention/mxfp4_decode_attention.py)
- Triton backend 集成：[`python/sglang/srt/layers/attention/triton_backend.py`](python/sglang/srt/layers/attention/triton_backend.py)
- 融合写 kernel：[`python/sglang/kernels/ops/quantization/mxfp4_quant.py`](python/sglang/kernels/ops/quantization/mxfp4_quant.py)
- eager MXFP4 codec：[`python/sglang/srt/layers/quantization/kvfp4_tensor.py`](python/sglang/srt/layers/quantization/kvfp4_tensor.py)
- native decode 测试：[`test/registered/attention/unittests/dense/test_triton_mxfp4_native_decode.py`](test/registered/attention/unittests/dense/test_triton_mxfp4_native_decode.py)
- kernel benchmark：[`scripts/playground/bench_mxfp4_decode_kernel.py`](scripts/playground/bench_mxfp4_decode_kernel.py)
