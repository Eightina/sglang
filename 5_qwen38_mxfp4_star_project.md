# 求职项目梳理（STAR）：为 Qwen3.8-27B 实现 MXFP4 KV Cache 与 Triton native decode kernel

> 本文是把本仓库这批工作整理成求职可讲述形态的项目档案，按情境（Situation）/
> 工作（Action）/结果（Result）组织。**所有数字与结论均出自本仓库既有记录**，
> 不做新增推断；细节出处见下表。
>
> | 主题 | 详细文档 |
> |---|---|
> | 基线 bug 诊断与修复 | [0b_qwen38_nvfp4_kv_descale_bug.md](0b_qwen38_nvfp4_kv_descale_bug.md) |
> | 模型架构调研与 L0-L3 路线图 | [0a_qwen38_attention_survey_notes.md](0a_qwen38_attention_survey_notes.md) §8-§11 |
> | L1/L2/L3 验收记录（测试、阈值、性能、精度） | [1_qwen38_mxfp4_kv_acceptance_records.md](1_qwen38_mxfp4_kv_acceptance_records.md) |
> | PLAIN 模式语义与代价、kernel 化必要性推导 | [2_qwen38_mxfp4_plain_mode_notes.md](2_qwen38_mxfp4_plain_mode_notes.md) |
> | codec 逐段实现说明 | [3a_qwen38_mxfp4_codec_walkthrough.md](3a_qwen38_mxfp4_codec_walkthrough.md) |
> | layout 选型与后端配对决策、性能调研 | [3b_qwen38_mxfp4_layout.md](3b_qwen38_mxfp4_layout.md) |
> | Triton kernel 实现与优化教学 | [4_qwen38_mxfp4_triton_kernel_tutorial.md](4_qwen38_mxfp4_triton_kernel_tutorial.md) |
>
> 相关 commit（当前分支序）：`0e646da70b`（KV scale 修复）、`b960f9e207`（L1 FP8 Torch
> golden）、`79e52ad0bd`（L2 MXFP4 codec + PLAIN 接入）、`f0a95c27c1` + `36bffd5861`
> （L3 Triton native kernel 与优化）。验收记录中登记的 L2 哈希 `2f22cc9d79` 为改基前值。

**一句话定位**：在 SGLang 中为混合注意力模型端到端落地 OCP MXFP4 KV cache
（codec → 框架接入 → Triton native decode / 融合写 kernel），在单卡 32GB 上把可用
上下文容量提升 1.52×，decode 速度与 FP8 生产基线持平（差距 ~9%），并用 roofline
账与 PPL/HumanEval 量化出该方案真实的价值边界与精度代价。

---

## S — 情境（Situation）

### 目标与约束

| 维度 | 事实 |
|---|---|
| 硬件 | 单卡 RTX 5090 32GB（SM120），无多卡 |
| 软件 | SGLang 0.5.16-dev、torch 2.13.0+cu130、flashinfer-python 0.6.18、triton 3.7.1 |
| 模型 | Qwen3.8-27B-NVFP4（compressed-tensors 混合量化：attention/部分 MLP W8A8，其余 MLP NVFP4 W4A4）；64 层中仅 **16 层全注意力**（GQA 24 Q 头 / 4 KV 头、head_dim 256），其余 48 层为 Gated DeltaNet 线性注意力 |
| 生产基线 | `--kv-cache-dtype fp8_e4m3` + FlashInfer decode；KV 32 KiB/token；mem-fraction 0.80 下可用上下文 55,188 tokens |
| 业务诉求 | **固定显存下要更长上下文**——KV cache 是唯一随上下文线性增长、且可压缩的显存项 |

### 为什么不是"改一个 dtype"就能完成

1. 框架内 `mxfp4` recipe 当时只是 `resolve_kv_cache_quant` 里的一个"保留"报错，无实现；
2. 生态无可用 FP4 KV attention kernel：FlashInfer 只接受 fp16/bf16/fp8（FP4 需先反量化到 FP8 workspace）；TRTLLM 的 native FP4 decode kernel 在 SM120 不可用 → **要拿到带宽与容量收益必须自研 kernel**；
3. 三个变量会同时引入误差：attention 实现差异（现成 CUDA kernel → 自研）、KV 数值格式差异（FP8 → MXFP4）、kernel 正确性。若一起引入，端到端输出不对时**无法归因**；
4. **起点基线本身是坏的**：`--kv-cache-dtype bf16` 下模型输出渐进式崩坏（前 10 余个 token 正确，随后漂移为无意义算式与重复循环），必须先证明基线可信，才谈得上"对齐基线"。

---

## A — 工作（Action）

### A0. 方法论：分层参考链 + 差分实验（贯穿全程）

不允许一步跨越多个变量。把开发拆成"每层只换一件事"，每层对照对象固定：

```
L0  FlashInfer FP8 decode（现役生产基线）
     ↓ 只换 attention 实现，KV 数值语义不变
L1  Torch FP8 decode golden     —— 对 L0 验证「decode 数学 + Radix/page-table + scale 语义」被正确复制
     ↓ 只换 KV codec，decode 数学不变
L2  Torch MXFP4 + PLAIN 接入     —— 对 L1 隔离出「纯量化格式损失」
     ↓ 只换执行位置，数值契约不变
L3  Triton native MXFP4 kernel   —— 对 L2 golden 验证 kernel 正确性
```

三条配套纪律：

- **双层数值口径**。单测内"冻结阈值"= 20-seed 表征 worst × 1.25，只对同一 fixture
  分布负责，避免用宽阈值掩盖 scale/layout 错误；跨数据（合成 → 真实模型）泛化用
  "服务级硬上限"（rel_l2 ≤ 2e-2、cos ≥ 0.999、norm_ratio ∈ [0.98, 1.02]）。
- **oracle 不自证**。MXFP4 测试 oracle 按 OCP MX v1.0 规范独立重写，不 import 生产
  实现；两侧 packed data / scale 字节必须 bit-exact。
- **先用契约测试锁死生产语义，再做差分**。为此补上了仓库此前完全缺失的 FP8 KV
  量化写入语义测试（原有测试只覆盖 `set_kv_buffer` 的路由/分发），锁定四条语义：
  | # | 语义 | 测试影响 |
  |---|---|---|
  | A1 | slot 0 保留（CUDA-graph padding slot，写 kernel `reserved_skip_index=0`） | fixture 的 loc 一律从 1 开始 |
  | A2 | `set_kv_buffer` 内部 `div_(scale)` 原地修改调用方张量 | 必须写入前快照，否则 reference 被二次除 scale 产生 NaN |
  | A3 | **scale 张量形态决定除法精度**：`bf16.div_(0-dim f32 CUDA tensor)` 会把 scale 降到 bf16（实测 0.78% 字节在舍入边界翻转）；生产用的 0-dim f32 CPU Parameter 才走全精度标量路径 | fixture 必须镜像生产形态 |
  | A4 | FP8 pool `store_dtype=uint8`，读侧才 `view(fp8)` | bit 级对比直接比 uint8 字节 |

  这组契约是差分测试"两侧输入相同"的前提：若写入语义有出入，差分测到的是写入
  方式差异而非 attention 差异。

### A1. 先修基线：KV descale 的读写不对称（框架 bug）

**诊断链**（每步都是排除法）：

1. **单变量隔离**：raw / chat 两种模式都崩 → 排除 chat template；temperature=0 复跑
   逐字相同 → 确定性数值错误；`--disable-radix-cache`、`--disable-cuda-graph`、
   `--disable-overlap-schedule`、换 attention backend、换 GDN decode backend、换 FP4
   gemm backend（输出 bit 级相同）全部无效 → 排除 prefix 复用、graph replay、调度
   重叠、单一后端、gemm 实现。
2. **算子级验证**：NVFP4 W4A4 单层（复现完整调用链 vs bf16 精确参考）
   rel_err 9.6%、cos 0.9954，在双重量化理论范围内；FP8 W8A8 单层 rel_err 2.6%；
   checkpoint 反量化自洽（NVFP4 反量化 std 1.02e-2 与同结构 bf16 MTP 层 1.26e-2
   同量级）；RoPE inv_freq 与 HF 逐值一致 → 排除量化算子与 checkpoint。
3. **独立引擎 golden + 逐层 dump**：用 HF transformers 跑同一 checkpoint 作参考，
   SGLang 用 `--debug-tensor-dump-output-folder` dump 逐 module 输出。层 0-2（线性
   注意力）正常，**层 3（第一个全注意力层）cos 骤降到 0.397**；层内下钻发现
   `qkv_proj` 输出与 HF 逐位一致，而 attention 输出 norm 2.23 vs HF 87.3（小约 41 倍）
   → 嫌疑锁定 attention kernel。
4. **层内逐步二分**（`locate_fa_bug.py`）：融合 RMSNorm ✓（rel 0.0017）、partial RoPE
   ✓（0.0019）、attention 输出 ×0.025 ✗。
5. **反解实锤**：attention 输出 = P·V，V 已知可逆 → 从输出反解出实际使用的权重
   矩阵 P，发现 **P 的行和恒等于 0.0245**（softmax 行和必为 1）= checkpoint 的
   `v_scale`。机制至此完全清楚：attention 输出 = 正确结果 × v_scale。

**根因**：checkpoint 声明了 `kv_cache_scheme`（static per-tensor FP8 KV scale），
框架据此为每个 attention 层创建 `k_scale/v_scale` 参数。**写入路径有 dtype 豁免**
（bf16 pool 实际存原始值），**读取路径无任何条件**，descale 直接传给 kernel
（flashinfer 3 处 + triton 4 处）→ 净效应是输出被整体乘了 v_scale。因为
`--kv-cache-dtype auto` 会依 checkpoint 解析为 FP8 KV（写除、读乘，对称），
这个 bug 被"官方路径"完整掩盖，只有"显式 bf16 + checkpoint 自带 KV scale"
的组合才触发。

**修复原则**：*descale 是否合法，取决于 KV cache 的实际存储 dtype，而不是
checkpoint 是否附带 scale*。引入 `kv_cache_scales_valid`（仅 fp8_e4m3/e5m2 为真），
读写两侧同条件化。

**验证**：贪心续写从"第 3 token 分歧"变为与 HF 完全一致；top-1 logprob 与 HF 的
差距 0.963 → 0.107；FP8 KV 路径行为不变（回归安全）。这条修复同时成为后续
MXFP4 的免疫机制——mxfp4 下 `kv_cache_scales_valid=False`，checkpoint 的 FP8
descale 不会误入新路径。

**副产品认知**：区分两个独立的量化域——linear 层的 W8A8/W4A4 是**计算格式**
（激活量化只在 gemm 入口瞬时存在，输出回 bf16，因为后续 k_norm / RoPE / gate
需要高精度）；KV cache FP8/MXFP4 是**存储格式**（持久存在，读写必须对称）。
两者 scale 体系独立，`--kv-cache-dtype` 只应影响后者。

### A2. 格式设计：layout 选型 + quant/dequant 设计决策

**layout 选型（结论：保持 flat `(slots, head_num, 8)`，不跟随 MXFP8 的 interleaved 布局）**

| 证据 | 内容 |
|---|---|
| kernel 契约 | Triton decode kernel 只要求 buffer 为 3-D `[max_slots, head, dim]` 并提取通用 stride，地址数学 dtype 无关 → packed uint8 buffer 直接适配，无需新布局 |
| interleaved 的适用前提不成立 | MXFP8 的 5-D interleaved 布局是 FA4 `BlockScaledBasicChunk` 的**硬件 page-atom 要求**；我的 kernel 走 `kv_indices` 逐 token gather，无该约束，interleave 只会让写 kernel 与 PLAIN 反量化复杂化，零收益 |
| 流量占比 | scale 仅占 KV 读流量 1/17（8 B vs 128 B/token/head），且同一 KV 组的多个 program 读同一份 scale 字节，L2 天然广播命中 |
| 零初始化安全 | uint8 0 = 2^-127（合法极小值），不是 0xFF=NaN；未写入 slot 也不会被 page table 引用 |

**codec 设计决策（严格实现 OCP MX v1.0 §6.3）**

| 决策 | 内容与理由 |
|---|---|
| 分块轴 | 每 KV head 沿 head_dim 独立 block-32、**不跨 head**（256/32 = 8 块/head，无 partial block）；跨 head 共享指数会把不同 head 的值域混进一个 scale |
| scale 公式 | `scale = 2^floor(log2(amax)) / 4` ⇒ `scaled_amax ∈ [4, 8)`；E8M0 以原始 byte 存储，读侧 `view(torch.float8_e8m0fnu)` |
| fp32 中间量 | bf16 → fp32 无损，除以 2 的幂精确、log2 边界稳定 → **整条量化链只保留规范定义的那一次 RNE 舍入**。这是生产实现与独立 oracle 能在 CPU/CUDA 双平台 bit-exact 的前提（对比同文件的工程变体 `FP4MXBlock16`：bf16 链 + ceil 近似 + half-up 计数法，无 oracle 契约） |
| RNE ties-to-even | 显式算到 8 个码点的距离 + 奇偶消歧（而非 bounds 计数法的 round-half-up）；规范空白点 `5.0` tie 于 {4.0, 6.0}（两者尾数位同为 0）的实现决策定为向下取 4.0，由 conformance 测试锁死 |
| 特殊值语义 | 全零块 scale = 2^-127；含 NaN → scale = 0xFF 且块内数据码全部清零（NaN 完全由 scale byte 表达）；含 ±Inf → scale = 2^127 且元素符号饱和 |
| 打包约定 | uint8 手工 nibble pack，**低 4 位 = 偶数 index 元素**；选 uint8 而非 `torch.float4_e2m1fn_x2`，因为切片/位移/查表等纯 tensor 操作、CUDA graph 捕获、CPU/CUDA 逐字节对比对 uint8 最直接，且与 NVFP4 路径输出 dtype 一致 |
| eager 而非 `@torch.compile` | 实测 torch inductor 对 `slice + pad-to-even + nibble-pack` 图形**跳过最后一个输出字节的写入**，输出含陈旧缓冲区内容；同进程内确定、跨进程不同，曾伪装成"滞后一次调用"。结论上升为通用规则：**bit-exact 场景，编译输出必须与 eager oracle 逐字节对比后才可信** |

**存储账**：每 head 128 B data + 8 B scale → 每 token 每 tensor 544 B，K+V × 16 层
= **17 KiB/token**（FP8 为 32 KiB，BF16 为 64 KiB）。

### A3. 框架接入（PLAIN 模式）：先锁语义，并把代价量化清楚

利用框架的 `(phase, backend, kind)` access 三元组声明能力，先用 `PLAIN`
（pool 读取时整层反量化成 BF16、attention kernel 零改动）打通端到端。这一步的
定位是**功能集成 + 数值 oracle 闭环，不是性能**，且代价被显式量化并写进 fail-fast
文案（自述 validation-only）：

| 代价 | 机制 | 量化 |
|---|---|---|
| 带宽倒贴 | 反量化被放进读取路径，每次调用 `torch.empty` 物化整层 BF16，无缓存复用 | 读 FP4 0.5 B + 写 scratch 2 B + kernel 读 scratch 2 B ≈ **4.53 B/elem**，比不量化（2.00）还贵 → TPOT 152.7ms |
| 禁 CUDA graph | 每次调用分配新临时张量，graph replay 要求地址/语义固定 | 参数解析阶段直接拒绝非 `--disable-cuda-graph` 组合 |
| 容量预算新口径 | PLAIN 同时物化一层 K 和 V | packed 16 KiB + scale 1 KiB + scratch 4 KiB ≈ **21 KiB/token**；`compute_cell_size` 与 pool 配置两处公式一致并由测试锁定 |

同期一个容易误判的发现：extend/prefill 差分曾呈现 ~0.1 的相对误差，根因是
**FlashInfer ragged prefill 的当前 chunk 使用未量化的 raw bf16 K/V，仅 prefix 从
cache 读取**；oracle 若只用 cache 来源，会把量化误差本身计成"实现差异" → extend
oracle 必须混合两种 KV 来源。

**PLAIN 的真正作用是把"语义正确"与"执行高效"解耦**：先证明这套量化数学在现有
kernel 生态下端到端数值等价（decode 差分冻结阈值 rel_l2 3.1e-3、真实模型回放
8/8 通过），再让 L3 只负责对齐已锁死的数值契约，而不是边写 kernel 边定义语义。
换 access kind = 换执行位置，语义随 recipe 走、不随 kernel 走。

同时补齐工程化所需的框架改动：`--kv-cache-dtype` 增加 mxfp4 选项；FlashInfer 的
KV dtype 不再从 workspace 标志推断、改为从活跃 phase 的 access 规则解析；参数解析
阶段的 fail-fast（非 flashinfer 后端 / 未 disable-cuda-graph / MLA 直接拒绝）；
容量估算按 recipe 区分；slot move 时 data 与 scale 同时移动。

### A4. kernel：必要性判断、模式选择、实现与三轮优化

**范围判断：只做 decode attention kernel + 一个独立的量化写入 kernel，不做 prefill kernel。**
理由是可算的账，而非偏好：

- decode 是 memory-bound，时间近似正比于读取字节数。每元素读取：BF16 2.00 B、
  FP8 1.00 B、MXFP4 PLAIN 4.53 B、**MXFP4 native ≈ 0.53 B**（FP4 0.5 B + scale 1/32 B，
  寄存器内解包）→ 相对 BF16 省 73%、相对 FP8 省 47%。**dequant 必须 inline 进
  attention kernel**：中间产物只被 attention 消费一次，独立算就得写显存中转，而
  中转正是要消除的东西；同时 native 路径无 per-call 临时张量 → CUDA graph 兼容。
- prefill 是 compute-bound，一次反量化进 workspace 可摊销 → 复用现有 FlashInfer。
- **quant 与 dequant 的优化不对称**（四个维度）：

  | 维度 | dequant（读） | quant（写） |
  |---|---|---|
  | 每步处理量 | 全部历史 KV（随 seq 线性增长） | 仅当前 token（恒定，每层 K+V 约 4 KiB） |
  | 瓶颈与收益 | HBM 带宽（4.53 → 0.53 B/elem） | kernel launch / CPU 开销（eager 实现是 20+ 个小 op × 16 层 × K/V = 数百次小 launch），收益是 launch 融合，带宽上无账可算 |
  | 形态 | 必须 inline 进 attention kernel | 独立 quant+scatter 写 kernel（数百次 → 每层 1 次） |
  | 与 CUDA graph | scratch 分配是 graph 的破坏者，native 化才修复 | eager 纯 tensor 本来就可捕获，kernel 化只是提速 |
  | 数值锁定难度 | 纯查表乘法，天然确定 | 需在 kernel 内复现 fp32 的 amax → log2 → floor → RNE 链，靠 L2 已锁死的契约兜底 |

**实现**（`kernels/ops/attention/mxfp4_decode_attention.py` 约 1.1k 行、
`kernels/ops/quantization/mxfp4_quant.py` 248 行）：two-stage split-KV decode
（per-head stage1 用于 MHA、grouped stage1 用于 GQA/MQA、stage2 合并 splits）、
复用 stock backend 的 stride 提取工具、E8M0 与 E2M1 均用精确位构造、融合写 kernel
完成 block amax → E8M0 → E2M1 saturating RNE → scatter data + scale（保留 slot 0、
无 host sync）；eager codec 保留为回退与参考实现。

**三轮优化（每步：现象 → 根因 → 改动 → 双层证据）**，初版（per-head grid + splits=8）
长文 TPOT 47.5ms：

| 步骤 | 根因 | 改动 | kernel 级证据 | 服务级证据 |
|---|---|---|---|---|
| T1 grouped kernel | GQA 24Q/4KV 下，同一 KV head 的 packed K/V 被 6 个 q head 的 program 各读一遍（**读放大 6×**）；短文被 L2 掩盖、长文暴露 | 每 program 服务一个 KV head 的整个 query group，K/V 解包一次共享，改用 `tl.dot`（bf16 tile；dequant 值 = E2M1×2^k 在 bf16 精确可表示，与手动解包逐位一致） | 与服务级同批验证 | 长文 47.5 → 30.5ms；短文 26.7 → 24.6ms |
| T2 splits 8→32 | bs=1 长文 grid 仅 kv_heads × splits = **32 CTA vs 170 SM**，kernel 是 **latency-bound 而非带宽-bound**（sweep：seq512 与 splits 无关、4096 强相关） | mxfp4 分支把 `max_kv_splits` floor 提到 32（显式 server arg 优先） | 4096 seq：155.5 → 66.7µs（T1+T2 组合） | 长文 30.5 → 25.7ms |
| T3 位构造反量化 | 解包热路径含 SFU `exp2` + fp32 乘法链（归因实验测出约 97µs/层的解包代价） | 纯 int32 移位/或直接拼出 bf16 位模式（含 normal/subnormal 桥、±0/Inf/NaN 优先级） | 4096 seq 68.9 → 63.8µs（−7%）；8192 seq 98.4 → 88.1µs（−10%） | 收益约 0.1ms，已在测量噪声内 |

参数 sweep 结论：BLOCK_N=128 比 64 慢 2~5 倍（tile 过大）；num_warps=8 一致优于 4；
seq=512 存在约 48µs 的 eager launch 固定下限（CUDA graph 下更小）。

顺带解决的数值陷阱：`tl.exp2` 是近似指令且 FTZ，会把 2^-127 的 E8M0 scale 清零
→ E8M0 一律走精确位构造；手写 E2M1 值表最初按 2 位尾数实现（E2M1 实为 1 位尾数），
被 A 步元素级差分立即捕获。

**SM120 新特性评估并否决**：triton 3.7.1 的 `tl.dot_scaled`（FP4 MMA）在 sm_120
编译通过，且 e2m1/e8m0 打包约定与池布局完全一致；隔离的 QK-only probe 比手动解包
**快 34%**（14.5 vs 21.9µs）。但在完整 decode loop 内**慢 2.2 倍**（153.7 vs 68.9µs）：
rhs 列数小、与 PV 手动解包/softmax 混排、每迭代小 tile，FP4 MMA 固定开销吞掉收益；
且 PV 的 scale 沿**输出维**而非归约维，dot_scaled 语义无法表达 → 保留为默认关闭的
实验开关，附 probe 脚本留证。

**CUDA graph 解禁**：native 路径无 per-call 临时分配 → 放开 `(flashinfer prefill,
triton decode)` 组合的 graph 限制，PLAIN decode 继续强制禁用。

### A5. 验证与评测基础设施

总量 51 文件 / 约 11.7k 行新增：生产代码约 1.9k 行、测试约 3.3k 行、工具与评测脚本
约 1.2k 行、设计与验收文档约 2.7k 行。验证分五类：

1. **codec conformance**：全部 E2M1 码、RNE 中点、饱和、scale 指数边界、全零块、
   NaN/±Inf、head 隔离、partial block；生产实现 vs 独立 oracle 的 packed/scale
   bytes 在 CPU/CUDA 双平台 bit-exact。
2. **双层差分 + 20-seed 表征冻结**：
   | 层级 | 路径 | 20-seed worst | 冻结阈值 |
   |---|---|---|---|
   | standalone（fp32 输出） | MHA per-head kernel | rel_l2 2.61e-7 | 3.5e-7 |
   | standalone（fp32 输出） | GQA/MQA grouped kernel | rel_l2 1.45e-3 | 1.85e-3 |
   | integration（bf16 输出，真实 pool + 真实 backend 全链路） | 两路径合并 | rel_l2 2.59e-3 | 3.3e-3 |

   残差来源已归因：p → bf16（PV dot 前）+ tensor core 累加顺序 + attention 输出的
   bf16 舍入，与 stock triton kernel / FlashInfer 同类语义，与 L2 PLAIN 冻结值
   （3.1e-3）同量级。
3. **写路径三方 bit-exact**：融合写 kernel vs eager codec vs OCP oracle（随机 bf16/fp16
   多种 dim、特殊值、RNE tie 中点、slot 0 保留、门控回退条件）。
4. **CUDA graph 确定性**：kernel 级 capture → 两次 replay 字节一致且与 eager 一致；
   服务级 capture 成功、decode batch 走 graph、greedy 重复输出一致。
5. **服务级与精度评测**：真实模型 layer-3 dump 离线回放（脚本重建
   qkv_proj → RMSNorm → partial RoPE → cache 写入链路）；kernel micro-bench + nsys
   时间线 + roofline 账；teacher-forcing PPL（32,760 tokens）、HumanEval 60 题×5 样本、
   AIME25 8 题。

评测过程中修掉两个基础设施 bug 并建立纪律：仓库共享的 `simple_eval_humaneval.find_code`
未剥离思维链模型的 `<think>...</think>` 块，函数签名定位落在推理文本里，导致多数
completion 提取失败（**首跑得到 0.233 / 0.017 的假分数**），修复为先取最后一个
`</think>` 之后的文本；`filelock ≥ 3.32` 的 fork 审计会杀死 human_eval 的多进程沙箱
→ 评测驱动改用 spawn。另外 `run_eval` 报告文件名固定会互相覆盖，一次原始产物因此
丢失（该配置分数由覆盖前的离线重放恢复，并与另一口径交叉验证一致）→ 此后所有评测
产物按 `<配置>_<口径>_<时间戳>` 归档。

---

## R — 结果（Result）

### 1. 交付与功能

`--kv-cache-dtype mxfp4 --prefill-attention-backend flashinfer
--decode-attention-backend triton` 在 **CUDA graph 开启**下端到端可用；框架侧
fail-fast 在参数解析阶段强制合法组合（后端配对、MLA、speculative、lean/score_mod/
DCP/SWA 等分支）。

### 2. 量化指标（RTX 5090，mem-fraction 0.80，bs≈1）

| 指标 | A = FP8 FlashInfer（基线） | B = MXFP4 PLAIN | C = MXFP4 Triton native（最终态） |
|---|---|---|---|
| KV 占用 | 32 KiB/token | 17 KiB/token（+ 4 KiB 瞬态） | **17 KiB/token** |
| max_total_num_tokens | 55,188 | 84,096 | **84,096（容量 1.52×）** |
| TPOT 短文（512/128） | 23.7ms | 152.7ms | **24.5ms** |
| TPOT 长文（4096/128） | 23.8ms | 152.7ms | **25.9ms（差距 ~9%）** |
| bench 总吞吐（长文） | 36.6 tok/s | 6.3 tok/s | **40.7 tok/s** |
| TTFT 长文（无 prefix 命中） | 652ms | 7050ms | 986ms |
| CUDA graph | ✅ | ❌（强制禁用） | ✅ |

kernel 级：stage1（4096 seq）**155 → 34.6µs（4.5×）**，stage1 + stage2 每层 40.8µs；
PLAIN → native 服务级 **约 5.9× 提速**。

### 3. 正确性

codec 与融合写 kernel 对独立 OCP oracle bit-exact；dequant 微内核元素精确（两个
记录在案的例外均在生产域外，且其中一个是 oracle 的 `torch.exp2` 在 subnormal 上
带 1 ULP 误差、kernel 位构造反而更准）；decode 差分三档冻结阈值全部达标；真实模型
回放 8/8 通过（rel_l2 3.1e-3 ~ 6.4e-3、cos ≥ 0.999979）；回归 246 passed / 103
subtests，未破坏 L1/L2。

### 4. 精度代价（如实结论，也是本项目最硬的一条）

| 评测 | A = FP8 | C = MXFP4 native |
|---|---|---|
| Teacher-forcing PPL（32,760 tokens） | 4.081 | **16.862（4.1×）** |
| HumanEval 60 题 × 5 样本 pass@1 | 0.883 | **0.700（−18.3pp，约 5σ）** |
| AIME25 8 题 greedy pass@1 | 0.875 | 0.875（n=8 无鉴别力：若真实存在差距，观察到一致结果的概率仍约 19%） |

**归因已闭合**：

- kernel 实现由整条差分链排除（codec bit-exact + dequant 元素精确 + 三档冻结阈值 +
  端到端回放）；
- 标准框架内的精度自由度也已实验排除：scale round mode（OCP 合法自由度）在
  uniform/normal/spiky 各 40 万 block 上实测 floor 10.9~16.9% vs ceil 11.4~26.6%
  相对 RMS → **现有 floor 实现在 KV 类分布下已是最优**（ceil 把值域搬进 E2M1 步长
  更粗的中段）；
- 剩余损失 = **E2M1 格式固有**（1 位尾数，元素相对 RMS 约 11~17%，为 FP8 E4M3 的
  3~4 倍），在保留标准 MXFP4 存储格式的前提下不可消除。

PPL 是最直接的证据（不经采样/提取/任务执行环节，逐位置 NLL 直接度量量化 KV 下的
预测质量，跨启动确定），它与"贪心生成首个 token 即分叉"的早期记录、HumanEval 的
−18.3pp 三者方向完全自洽。

### 5. 吞吐收益有限的归因（roofline 账）

decode 每 step 的 DRAM 读取构成（bs=1，seq=4096）：

| 项 | FP8 KV | MXFP4 KV | 说明 |
|---|---|---|---|
| 权重（64 层 + embed/lm_head） | 21.5 GB ≈ 12.0ms @1.79TB/s | 21.5 GB | **不随 KV dtype 变化，占约 97%** |
| KV cache 读取 | 128 MiB（71µs） | 68 MiB（38µs） | 仅 16 层全注意力读 KV |
| **KV 差异** | — | — | **33µs ≈ TPOT 的 0.14%** |

四条结论：

1. **decode memory-bound 的主体是权重，不是 KV**。这不是 kernel 实现问题，而是
   workload 结构：该模型只有 16 层全注意力（其余 48 层是状态固定的线性注意力），
   且单卡 32GB 限制了 batch × seq 的规模。
2. **KV 带宽收益 ∝ batch × seq_len**。要让 KV 占 TPOT 超 20%，需
   `bs × seq > 约 30 万 token`（如 16×19k、32×9k），此时理论 TPOT 为 FP8 21.0ms vs
   MXFP4 16.7ms（20% 优势）；但 32GB 单卡不可达（线性注意力状态池把并发 cap 在 8~16）。
3. **实测归因**：同 bs=1 长文下 FP8+flashinfer 23.80ms / FP8+triton 24.15ms /
   mxfp4+triton 25.67ms（位构造前）→ backend 实现差异仅 0.35ms，mxfp4 的 1.55ms
   （约 97µs/层）是**解包的计算代价**：读的字节更少（省 38µs/层带宽）但解包更贵
   （费约 97µs/层 ALU 与延迟），净亏；T3 位构造后大幅收敛。
4. **优化已收敛**：nsys 实测 16 层 attention kernel 合计约 653µs = 最终 TPOT
   （25.9ms）的 **约 2.5%** → persistent kernel / stage 融合 / TMA 预取等的端到端
   上限 <1%。tensor core 也不是瓶颈：attention 仅约 1.6 GFLOP/step ≈ 8µs @209 TFLOPS，
   瓶颈在喂给 tensor core 之前的解包 ALU/延迟（T3 已收敛）与占用率（T2 已收敛）。

### 6. 最终价值定位（工程判断，而非"做完了"）

在单卡 32GB + 该混合架构（16 层全注意力 + 48 层线性注意力）+ bs ≤ 8 的约束下，
MXFP4 KV 的可兑现价值是 **容量 1.52×（同显存多装 52% 上下文）+ 与 FP8 同级的
decode 速度（差距约 9%）**；吞吐优势场景（大 batch × 长上下文，KV 带宽占比 >20%）
需要更大显存的部署形态。代价是**对精度敏感任务的可测损失**（PPL 4.1×、HumanEval
−18pp），部署决策应按任务权衡。若要降低精度代价，可行方向均超出标准 MXFP4 存储
格式：K/V 混合精度（V 对误差更敏感，可保留 fp8）、低秩/稀疏高精度残差补偿，或
接受代价用于容量优先场景。若要继续压 TPOT，方向在权重路径与线性注意力层
（占约 97%），而非 attention kernel。

### 7. 附带的可复用产出

- KV descale 读写不对称的修复（同一"无条件使用 descale"模式仍存在于 fa3/flashmla
  等后端，具备上游价值）；
- 仓库首个 FP8 / MXFP4 KV 写入语义契约测试（此前只有路由/分发类测试）；
- HumanEval 评分对思维链模型的提取修复 + spawn 评测驱动 + 离线重评分工具；
- `tl.dot_scaled` 在 SM120 上的实测结论与 probe 脚本；
- kernel micro-bench、位构造全域精确性 probe、PPL 评测脚本。

---

## 附：面试叙事建议

**开场 30 秒**：目标是显存受限下的长上下文；结论是容量 1.52×、decode 与 FP8 生产
基线持平、但 MXFP4 的 KV 精度代价被量化为 PPL 4.1×。这个项目最有价值的产出不是
kernel 本身，而是**一套能把"格式固有代价 / 实现 bug / workload 结构"三者干净分开
的验证方法**。

**准备三个故事（对应三类考察点）**

| 考察点 | 讲哪段 | 钩子 |
|---|---|---|
| 系统调试能力 | KV descale bug | "从 attention 输出反解 P·V，发现 softmax 行和恒为 0.0245" —— 排除法 + 独立引擎 golden + 逐层 dump + 层内二分 + 反解，比穷举运行时开关快得多 |
| 数值与规范严谨性 | codec + 分层 oracle 链 | fp32 中间量只保留一次规范 RNE；oracle 独立重写不自证；编译器误编译被逐字节对比抓住 |
| 性能工程与判断力 | T1/T2/T3 + roofline | 读放大 6×、32 CTA vs 170 SM、位构造替 SFU 指令；以及**主动喊停**：attention 只占 TPOT 2.5%，继续优化端到端上限 <1% |

**被问"结果不是纯赢，怎么看"**（这是加分项）：技术目标全部达成（容量 1.52× +
速度持平 + CUDA graph 兼容）；同时用 PPL/HumanEval 与 roofline 账给出了这个方案
**在什么部署形态下才划算**的量化边界，并把"kernel 实现问题"与"格式固有代价"彻底
分离（差分链 + scale round mode 实验）。

**要避免的表述**：不要说"生产部署"或"已合并上游"（当前是单卡研究性实现 + 受限组合
fail-fast）；不要只报容量 1.52× 而不主动提精度代价——主动量化过代价的人，和没量化
过的人，在追问下差距很大。
