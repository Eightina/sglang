# Qwen3.8-27B MXFP4 KV：容量 / LUT 优化 / 端到端 profiling 汇总

> 全部数据来自单卡 RTX 5090（31.4 GiB 可用），同一台机器、同一份代码基线，
> 逐项标注了启动参数。凡"优化前/后"均指 PR #38724 的 Triton decode kernel
> 是否启用 E2M1×E8M0 查表（LUT）。
>
> 相关代码：`python/sglang/kernels/ops/attention/decode_attention_mxfp4_sm120.py`、
> `python/sglang/srt/layers/quantization/fp4_kv_cache_quant_method.py`、
> `python/sglang/srt/layers/quantization/kvfp4_tensor.py`

---

## 0. TL;DR

1. **LUT 优化成立**：把 `E2M1 code × E8M0 scale → bf16/fp32` 的 ~20 条整数构造指令换成
   4096 项查表 gather，kernel 输出与改前**逐 bit 相同**，寄存器从 255(+16 spill) 降到 227(+0 spill)。
   kernel 级 32k 快 **42%**，端到端 150k 上下文 TPOT 快 **39%**。
2. **容量与 PR 完全对齐**：同 static 下 MXFP4/FP8 KV token 比 **1.524×**（PR 也是 1.524×）；
   单 token 原始 KV 是 17.1 KiB vs 32.8 KiB（1.91×），比值被"公共项"摊薄到 1.52。
3. **上下文上限不是 `--context-length` 决定的**，而是 static 预算里的三个隐藏消费者：
   mamba cache 自动膨胀、prefill CUDA graph 捕获、**按整条序列解包的 PLAIN dequant 缓冲**。
   本机实测：0.85 + mamba8 + 关 prefill graph → **158,548 token（150k 请求可服务）**；
   0.88 时池子能算到 **204,669 token**，但被 dequant 峰值 OOM 挡住（这就是"210k 记录"对不上的原因）。
4. **FP4 的并发价值已实测**：同 static、32k 上下文，FP8 只放得下 1 路（吞吐被钉死 23.8 tok/s），
   MXFP4 放得下 2 路并真实并批 → **29.6 tok/s（+24%）**。
5. 剩余差距不在反量化：长上下文 MXFP4 仍比 FP8 慢 10~18%，瓶颈是 Triton native decode
   相对 FlashInfer FP8 decode 的效率；prefill 侧 MXFP4 的 TTFT 比 FP8 高 12~52%，瓶颈是
   PLAIN 逐层整段解包。

---

## 1. 环境与基线

| 项 | 值 |
|---|---|
| GPU | RTX 5090，SM120，31.4 GiB（`memory.total` 32,607 MiB）|
| 软件 | sglang 0.5.23.dev5（editable，仓库 `/sgl-workspace/sglang`）、torch 2.13.0+cu130、triton 3.7.1 |
| 模型 | `unsloth/Qwen3.8-27B-NVFP4`，compressed-tensors 混合量化，22 GiB（`model.safetensors` 21.0 + MTP 0.8）|
| 结构 | `qwen3_5`，64 层 = 16 full attention（GQA 24Q/4KV，head_dim 256）+ 48 GDN 线性注意力 |
| 代码基线 | PR #38724 分支 `eightina/qwen3.8-kv-mxfp4` @ `b2ada378e1` |

服务端三套配置（除注明外，三组对比全部使用完全相同的 flag）：

```bash
# FP8 baseline
--kv-cache-dtype fp8_e4m3 --attention-backend flashinfer
# MXFP4（生产组合）
--kv-cache-dtype mxfp4 --prefill-attention-backend flashinfer --decode-attention-backend triton
# 公共
--mem-fraction-static <MEM> --max-mamba-cache-size <MAMBA> --context-length <CTX>
[--disable-prefill-cuda-graph]
```

---

## 2. 容量与显存账本

### 2.1 单 token KV 与容量

| | FP8 | MXFP4 | 比值 |
|---|---|---|---|
| KV dtype | `float8_e4m3fn` | `float4_e2m1fn_x2` | — |
| 单 token KV（实测）| **32.7 KiB** | **17.1 KiB** | 1.91× |
| KV token @ static 0.80 / ctx64k | 53,858（1.64 GB）| 82,070（1.34 GB）| **1.524×** |
| KV token @ static 0.85 / ctx200k | 104,047（3.18 GB）| 158,548（2.58 GB）| **1.524×** |
| KV token @ static 0.85 / mamba20 / ctx32k | 48,567 | 74,007 | 1.524× |

> PR 记录 53,503 / 81,529（1.524×），本机 0.80 复现为 54,498 / 83,045（ctx32k）与
> 53,858 / 82,070（ctx64k），**比值完全一致**，绝对值高 1.9%（上游版本/计账差异）。

**为什么 1.91× 的压缩只换来 1.52× 容量**：static 预算里 KV 只是其中一项，
权重（21.48 GB）+ mamba states + prefill dequant 预留是所有 dtype 共享的固定项。

### 2.2 static 扫参（MXFP4，看真正的上限）

| mem-fraction | mamba | prefill graph | KV tokens | KV 占用 | 结果 / 失败点 |
|---|---|---|---|---|---|
| 0.80 | auto(8) | on | 82,070 | 1.34 GB | ✓（余量 4.20 GB）|
| 0.84 | 8 | on | 143,175 | 2.32 GB | ✗ **prefill CUDA graph 捕获**：50 个桶要 4.89 GB，只剩 24 MiB |
| 0.85 | auto(**13**) | on | 122,169 | 1.98 GB | ✗ mamba 自动涨到 13 slot（1.97 GB）抢走 KV；warmup OOM |
| **0.85** | **8** | **off** | **158,548** | **2.58 GB** | ✓ 余量 4.50 GB，**150k 请求实测通过** |
| 0.86 | 8 | off | 173,922 | 2.82 GB | ✗ warmup 里 PLAIN 解包要 680 MB，仅剩 348 MB |
| 0.87 | 8 | off | 189,296 | 3.06 GB | ✗ 同上 |
| 0.88 | 8 | off | **204,669** | 3.32 GB | ✗ 同上（**这就是 210k 记录对应的池子**）|

### 2.3 三个"账本外"的消费者

1. **mamba cache 会按剩余显存自动膨胀**：`--max-mamba-cache-size` 默认 `None`，
   0.85 时自己从 8 涨到 13 slot（1.29 → 1.97 GB），直接吃掉 0.7 GB KV 预算。
   单 slot ≈ 0.147 GB（`ssm_state`）+ 0.003 GB（`conv_state`），**每并发请求占 5 slot**。
2. **prefill CUDA graph 捕获要 ~4.9 GB**（`breakable` 后端、50 个 bucket）。
   它只影响 TTFT，不影响 TPOT，可安全关闭换取 KV 预算。
3. **PLAIN prefill 的 dequant 按"整条序列"分配**（`kvfp4_tensor.batched_dequantize`
   里 `float_vals = torch.where(...)`），fp32 K+V ≈ **8 KB/token**：
   170k token 时单次尝试 680–800 MB；这是 0.86~0.88 起不来的唯一原因，
   也对应笔记里"预留 2G dequant buffer"那一项（当前 build 未预留）。

**结论**：想上 190k–205k，必须先把 PLAIN dequant 改成按 chunk 解包（或输出 bf16），
参数层面已经没有空间。

---

## 3. LUT 优化（E2M1×E8M0 → 查表）

### 3.1 实现

- 表：`e2m1_scale_lut(device, dtype)`，索引 `(scale_byte << 4) | e2m1_code`，4096 项；
  bf16 8 KiB / fp32 16 KiB，**由原 `_e2m1_scale_to_bf16` 自己生成**（1 个 program 的 Triton
  kernel 枚举 16×256），因此表值与改前逐 bit 相同（含 E8M0 NaN 0xFF、2⁻¹²⁷ 次正规 scale、
  带符号零、inf 编码）。
- 调用点：per-head(MHA) kernel 用 fp32 表、grouped(GQA) kernel 用 bf16 表；
  每个元素由"~20 条整数构造指令"变成"1 次 L1 gather"。一个 block 的 16 个 code 落在
  同一 32 B sector 内，gather 命中率高。
- 预热：在 `MXFP4KVCacheMethod.create_buffers()` 里把两种 dtype 都建好，
  避免首次 decode 落在 CUDA graph 捕获里被烘进图。

### 3.2 正确性

| 检查 | 结果 |
|---|---|
| 原 kernel 单测（MHA/GQA/MQA/CUDA-graph 重放，4 项）| 全过，`rel_l2` 与改前**逐位一致**（MHA 1.220e-07、GQA 1.293e-03…）|
| 新增 LUT 全组合测试（4096 项 vs 生产 codec）| byte 1..254 **bit-exact**；byte 0xFF 双方皆 NaN；byte 0 仅钉住带符号零行 |
| fp32 表 == bf16 表加宽 | `torch.equal` 通过（NaN 行按 pattern 比较）|

> 顺带发现一个**既存隐患**（本次未改）：`_e2m1_scale_to_bf16` 在 bf16 次正规域丢掉了
> E2M1 的 mantissa 位，`code=3 / scale_byte=0` 给出 1.0×2⁻¹²⁷ 而非 1.5×2⁻¹²⁷。
> 该组合不可达（amax=0 时量化器输出全零 code），因此对可达域无影响。

### 3.3 kernel 级性能（bs=1、24Q/4KV、head_dim 256、splits=32、warps=8）

| seq | 优化前 eager | 优化后 eager | 优化前 **graph** | 优化后 **graph** | graph 变化 |
|---|---|---|---|---|---|
| 1k | 0.045 ms | 0.046 ms | 0.021 ms | **0.014 ms** | −33% |
| 4k | 0.043 ms | 0.045 ms | 0.031 ms | **0.021 ms** | −32% |
| 16k | 0.096 ms | 0.057 ms | 0.098 ms | **0.059 ms** | −40% |
| 32k | 0.186 ms | 0.106 ms | 0.187 ms | **0.108 ms** | −42% |

寄存器 / 共享内存（同一编译配置）：**255 regs + 16 spills → 227 regs + 0 spills**，smem 均 43,008 B。

> 4k 以下 eager 口径两者都 ≈45 µs，是 launch 开销主导；graph 口径才看得到真实差异。
> 这一点很重要：只看 eager 会得出"4k 无收益"的错误结论。

---

## 4. 端到端延迟（单请求，128 输出，冷 prefill，3 次中位）

### 场景 A：mem 0.80 / ctx 65536 / prefill graph **on** / mamba auto(8)

| 上下文 | 指标 | FP8 | MXFP4 优化前 | MXFP4 优化后 |
|---|---|---|---|---|
| 4k | TTFT | 423.8 | 530.8 | 535.7 ms |
| 4k | TPOT | 23.47 | 24.59 | **23.62 ms** |
| 30k | TTFT | 4218.9 | 4899.6 | 4893.3 ms |
| 30k | TPOT | 24.13 | 29.89 | **25.33 ms** |
| 60k | TTFT | ✗ 不可服务 | 11620.7 | 11613.7 ms |
| 60k | TPOT | ✗ | 36.32 | **27.40 ms** |

FP8 在此容量下 60k 直接 400：`Input length (60000 tokens) exceeds the maximum allowed length (53852)`。

### 场景 B：mem 0.85 / ctx 200000 / prefill graph **off** / mamba 8（KV 158,548）

| 上下文 | TTFT (F8 / 前 / 后) | TPOT (F8 / 前 / 后) | TPOT 后 vs 前 | TPOT 后 vs F8 |
|---|---|---|---|---|
| 4k | 420.0 / 644.6 / 639.2 ms | 23.47 / 24.57 / **23.52** | −4.3% | +0.2%（打平）|
| 30k | 4183.3 / 5738.3 / 5694.1 | 24.11 / 29.90 / **25.30** | −15.4% | +4.9% |
| 60k | 10847.3 / 13233.5 / 13252.2 | 24.75 / 36.32 / **27.37** | −24.6% | +10.6% |
| 100k | 23680.5 / 26543.7 / 26618.3 | 25.57 / 44.56 / **30.09** | −32.5% | +17.7% |
| 150k | ✗（容量 104k）| 55.26 / **33.51** ms | **−39.4%** | — |

decode 吞吐（1/TPOT）：

| ctx | FP8 | 优化前 | 优化后 |
|---|---|---|---|
| 4k | 42.60 | 40.69 | **42.52 tok/s** |
| 30k | 41.48 | 33.44 | **39.52** |
| 60k | 40.41 | 27.53 | **36.54** |
| 100k | 39.11 | 22.44 | **33.23** |
| 150k | — | 18.10 | **29.84** |

### 4.1 两个必须知道的副作用

1. **关掉 prefill graph 后 MXFP4 的 TTFT 涨 ~19%（4k: 536→639 ms），FP8 不受影响（424→420）**。
   原因：PLAIN 反量化在 Python 侧是一长串 eager torch 算子，平时被 prefill graph 掩盖。
   → 把 dequant 做成单个融合 kernel，可同时省下 ~1 GB 临时缓冲并吃掉这 19%。
2. **端到端收益 ≈ kernel 隔离收益 × 3.5**（128k 上下文时 4.5 ms vs 1.26 ms 理论值）。
   原因就是那 16 处寄存器溢出：微基准里 KV 只有 2–16 MB、L2 全热，spill 被缓存掩盖；
   生产里每 token 要过 64 层权重，L2 被冲干净，溢出访存直接落在关键路径上。

---

## 5. 端到端吞吐

### 5.1 串行（concurrency 1）

| 场景 | 负载 | FP8 | 优化前 | 优化后 |
|---|---|---|---|---|
| A（0.80/64k）| 4k in × 512 out × 8 | 40.85 | 38.59 | **40.04 tok/s** |
| A（0.80/64k）| 30k in × 128 out × 4 | 17.23 | 14.52 | **15.49 tok/s** |
| B（0.85/200k）| 4k in × 512 out × 8 | 40.95 | 38.16 | **39.81 tok/s** |
| B（0.85/200k）| 100k in × 128 out × 2 | 4.69 | 3.95 | **4.17 tok/s** |

### 5.2 并发（FP4 的容量红利）

配置：mem 0.85 + mamba 20（bs 上限 4）+ ctx 32768 + 关 prefill graph。
容量差异：**FP8 48,567 vs MXFP4 74,007** token。

**32k in × 256 out × 4 请求**

| 配置 | 并发 | e2e 吞吐 | TTFT p50 / p90 | TPOT p50 |
|---|---|---|---|---|
| FP8 | 1 | 23.81 tok/s | 4.54 / 4.56 s | 24.16 ms |
| FP8 | 2 | **23.81 tok/s（零提升）** | 9.92 / **15.27 s** | 24.19 ms |
| MXFP4(LUT) | 1 | 21.93 tok/s | 5.12 / 5.13 s | 25.47 ms |
| MXFP4(LUT) | 2 | **29.62 tok/s（+35%）** | 7.95 / 10.20 s | 36.30 ms |

→ FP8 的 bs=2 是"假并发"：吞吐不变、TTFT 翻倍，第二路只是排队等 KV。
MXFP4 真正并批，吞吐高出 FP8 天花板 **24%**。注意 bs=1 时 FP8 反而快 8%。

**16k in × 512 out × 8 请求（仅 MXFP4）**

| 并发 | e2e 吞吐 | TTFT p50 / p90 | TPOT p50 |
|---|---|---|---|
| 1 | 34.67 tok/s | 2.28 / 2.30 s | 24.39 ms |
| 2 | **58.10 tok/s（+68%）** | 3.71 / 4.58 s | 27.16 ms |
| 4 | 57.27 tok/s（≈bs2）| 6.00 / **23.20 s** | 29.25 ms |

→ bs 不是越大越好：4×16k 的 prefill 吃掉墙钟，尾部 TTFT 到 23 s，**拐点在 bs=2**。

### 5.3 并发的真实成本

- 每请求 5 个 mamba slot，单 slot ≈ 0.147 GB → bs=2 要 1.47 GB，bs=4 要 2.95 GB。
  **长上下文高并发时，mamba 比 KV 更贵**。
- 32k×bs4 = 128k KV token（2.19 GB）+ mamba 2.95 GB → static ≈ 0.87，
  当前被整段 PLAIN dequant（~1 GB）挡住。

---

## 6. 结论清单

1. LUT 把 dequant 从 ALU 计算变成 L1 gather，**bit-exact**，无精度代价。
2. 收益随上下文线性放大：kernel 32k −42%；端到端 TPOT 4k −4.3% → 150k −39.4%。
3. 4k 上 LUT 让 MXFP4 与 FP8 **TPOT 打平**（23.52 vs 23.47，+0.2%），
   PR 里"decode −6.7% vs FP8"的代价在这个长度基本消掉。
4. FP8 的硬边界是容量：60k 输入在同 static 下直接 400；MXFP4 可服务 158k。
5. FP4 的并发红利已量化：32k 上下文下 FP8 天花板 23.8 tok/s，MXFP4 bs=2 达 29.6（+24%）。
6. 剩余差距在新维度上：① prefill/TTFT（PLAIN 逐层整段解包，+12~52%）；
   ② 长上下文 decode（Triton native vs FlashInfer FP8，仍差 10~18%）。
7. `mem-fraction-static` 不是唯一约束：mamba 自动膨胀、prefill graph 捕获、
   按序列分配的 dequant 缓冲三者叠加，才是"到不了 210k"的真正原因。

---

## 7. 复现指南

```bash
# 场景 A（PR 原始口径：4k/30k 对比、容量 1.52×）
MEM=0.80 MAMBA=8 CTX=65536 /tmp/launch_qwen38_mxfp4.sh
# 场景 B（最大上下文 158k）
MEM=0.85 MAMBA=8 CTX=200000 EXTRA="--disable-prefill-cuda-graph" \
  /tmp/launch_qwen38_mxfp4.sh
# FP8 baseline（同 flag，端口 30002）
MEM=0.85 MAMBA=8 CTX=200000 EXTRA="--disable-prefill-cuda-graph" \
  /tmp/launch_qwen38_fp8.sh
# 并发（bs 上限 4）
MEM=0.85 MAMBA=20 CTX=32768 EXTRA="--disable-prefill-cuda-graph" \
  /tmp/launch_qwen38_mxfp4.sh
```

基准脚本（仓库根，未提交）：

| 脚本 | 用途 |
|---|---|
| `.bench_mxfp4_decode_kernel.py` | kernel 级；eager 与 CUDA-graph 双口径，多 seq 长度 |
| `.bench_qwen38_kv.py` | 单请求 TTFT/TPOT/decode；支持 `--input-len`、`--runs`、冷 prefill 注入 |
| `.bench_qwen38_serving.py` | closed-loop 吞吐；`--concurrency` 按 wave 对齐释放，输出 p50/p90 |

关键日志：`/tmp/lut_final.log`（当前服务）、`/tmp/base_085.log`、`/tmp/fp8_085.log`、
`/tmp/lut_m08{4,6,7,8}.log`（static 扫参）、`/tmp/tp32_*.log`、`/tmp/fp8_tp32_*.log`。

---

## 8. 待办（按收益排序）

1. **PLAIN dequant 改 chunk 化 / 融合 kernel**：省 ~1 GB 峰值 → 0.87/0.88 可起，
   可用上下文 190k–205k；同时吃掉关 prefill graph 带来的 19% TTFT 亏空。
2. **decode kernel 继续优化**：nibble interleave（每字节现在被读 2 次，可减半 packed 访存）、
   spill 清零后试 `num_warps=4` / `BLOCK_N=128`；目标追平 FlashInfer FP8 的长上下文 decode。
3. **bs=4 @32k 验证**：static 0.87 + mamba 20 + ctx 32k，MXFP4 预期 45–50 tok/s，
   FP8 同 flag 下最多 bs=3。
4. 修 `_e2m1_scale_to_bf16` 次正规域丢 mantissa 的既存问题（不可达域，优先级低）。
