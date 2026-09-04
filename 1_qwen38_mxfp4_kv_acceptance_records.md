# Qwen3.8 MXFP4 KV cache：L1/L2 测试与验收记录

> 本文档由 [qwen38_attention_survey_notes.md](qwen38_attention_survey_notes.md) §11 开发路线图中
> 随代码开发追加的"L1/L2 验收记录"拆出并重构（2026-09-01）。
> 路线图本身（L0-L3 参考链设计、FP8/MXFP4 语义、开发序列）仍保留在原笔记 §11；
> 本文只回答四个问题：**交付了哪些代码、跑了哪些测试、数值结论是什么、对应哪个 commit。**

## 0. 阶段与 commit 索引

| 阶段 | 内容 | 原笔记位置 | commit |
|---|---|---|---|
| L1 验收 | FP8 Torch decode golden（路线图 §11.3 实现、§11.4 A/B/C 验收） | 原 §11.4 "L1 验收记录" | `b960f9e207` "L0: fp8 torch radix attention done" |
| L2 验收 | MXFP4 codec + PLAIN 接入现有框架（路线图 §11.5/§11.6） | 原 §11.9 "L2 验收记录" | `2f22cc9d79` "mxfp4 kv cache with torch kernel done" |
| L3 验收 | native triton decode kernel + 融合写 kernel + CUDA graph（路线图 §11.7/§11.8） | 本文 §3（新增） | `f0a95c27c1` "triton native mxfp4 decoding attention kernel" + 位构造/grouped/splits32 优化（同系列后续提交） |

两阶段共用环境：RTX 5090 32GB（SM120）、torch 2.13.0+cu130、flashinfer-python 0.6.18；
模型 Qwen3.8-27B-NVFP4；L0 生产基线 `--kv-cache-dtype fp8_e4m3`（FlashInfer decode）。

数值判定采用两层口径（两阶段一致）：

- **冻结阈值**：单测内断言，只对同一 fixture 分布负责，取 20-seed 表征 worst × 1.25；
- **服务级硬上限**：跨数据（合成 → 真实模型）泛化判定，rel_l2 ≤ 2e-2、cos ≥ 0.999、
  norm_ratio ∈ [0.98, 1.02]。

---

## 1. L1 验收：FP8 Torch decode golden（2026-08-31 完成）

**判定标准**（对应路线图 §11.4）：三步全部通过才承认"Torch FP8 decode 与
FlashInfer FP8 decode 数值等价"，才允许进入 L2——

- **A** codec 契约单测：锁定生产 KV 写入语义；
- **B** 单层 decode 差分：Torch FP8 decode reference vs FlashInfer FP8 decode；
- **C** 模型端到端：真实模型 dump 回放。

### 1.1 交付物（全部含于 commit `b960f9e207`）

| 文件 | 变更 | 关键 symbol / 内容 |
|---|---|---|
| `python/sglang/test/kits/attention_unittest/attention_methods/fp8_decode_attention.py` | 新增 | `fp8_cache_quantize_reference`（FP8 QDQ：clone → `div_(scale)` → cast fp8）；`gather_kv_sequences`（按 `req_to_token` gather 物理页，不假设 cache loc 连续）；`torch_fp8_radix_decode_reference`（decode 主 oracle：BMM 边界 scale 放置——K descale 折入 QK^T、V descale 折入 PV，fp32 数学）；`torch_fp8_decode_dequant_first_reference`（先反量化的交叉验证变体）；`decode_output_metrics`（rel_l2 / cos / norm_ratio / max_abs，device 无关） |
| `python/sglang/test/kits/attention_unittest/attention_methods/dense_attention.py` | 修改（向后兼容） | fixture 新增 `kv_cache_dtype / k_scale / v_scale / seed` 参数；FP8 pool `store_dtype=uint8`；scale 形态镜像生产 `kv_cache.py`：0-dim f32 **CPU** `nn.Parameter` + `k_scale_float`（原因见 A3） |
| `test/registered/attention/unittests/dense/test_flashinfer_fp8_decode.py` | 新增 | A 步 = `TestFp8KvWriteContract`；B 步 = `TestFlashInferFp8DecodeGolden` |
| `compare_fp8_decode.py`（仓库根目录） | 新增 | C 步真实 dump 离线回放脚本 |

### 1.2 A — 生产 KV 写入语义契约：`TestFp8KvWriteContract`，6/6 通过

**定位（被测对象与验证目标）**：本组测试的被测对象是**生产代码**
`MHATokenToKVPool.set_kv_buffer`（`memory_pool.py`），不是 Torch golden 本身。
它验证的等式是：生产 FP8 写入 == golden 的 QDQ 函数
`fp8_cache_quantize_reference`（bit-exact）。该 QDQ 函数的实现（clone →
`div_(scale)` → cast fp8）就是 `set_kv_buffer` 写入语义的复制品，本组测试
即证明这个复制品成立。这是 B 步差分"两侧输入相同"的前提：B 步中 golden
消费的 cache 就是生产 pool 的字节（`pool.get_key_buffer`），若写入语义有
出入，差分测到的将是写入方式差异而非 attention 差异（B 步内嵌的"当前
token slot 字节 == QDQ reference"断言即本组契约的简化版）。
在此改动之前，仓库仅有 `set_kv_buffer` 路由/分发类测试（SWA 路由、MLA
分支、loc fast path），无任何覆盖 FP8 量化写入语义的测试；L2 的
`test_mxfp4_pool_cuda.py::test_set_kv_buffer_bit_exact_and_slot0_reserved`
沿用同一模式。fixture 的 loc 从 1 开始、写入前快照等约定均由本组语义决定。

锁定的四项生产写入语义（每项有对应用例）：

| # | 语义 | 说明与测试影响 |
|---|---|---|
| A1 | slot 0 保留 | 生产写入 JIT kernel `reserved_skip_index=0` 跳过 slot 0（CUDA-graph padding slot）；测试 loc 一律从 1 开始 |
| A2 | `set_kv_buffer` 原地修改调用方张量 | 内部 `div_(scale)` 为 in place；测试必须在写入前快照 cache_k/cache_v，否则 reference 被二次除 scale，产生 NaN/溢出 |
| A3 | scale 张量形态决定除法精度 | `bf16.div_(0-dim f32 CUDA tensor)` 会把 scale cast 到 bf16（实测 0.78% 字节在舍入边界翻转）；0-dim f32 **CPU** tensor（生产 `kv_cache.py` 的形态）与 1-dim f32 CUDA tensor 才走全精度标量路径；fixture 必须镜像生产的 CPU 0-dim Parameter |
| A4 | FP8 pool `store_dtype=uint8` | bit 级对比直接比 uint8 字节；读侧 `_get_key_buffer` 才 `.view(fp8)` |

### 1.3 B — 单层 decode 差分：`TestFlashInferFp8DecodeGolden`，10/10 通过

覆盖矩阵：MHA/GQA/MQA × head_dim 64/128/256 × page size 1/16/32/64 ×
4 种物理 loc 布局（contiguous / shuffled / non-monotonic 等）× scale=1 与
真实 checkpoint scale；cache 来源两种：直接 codec 填充、FlashInfer prefill 产出。
对比顺序：先 cache bytes / dequant K/V，再 attention output。

结果：rel_l2 ≈ 2.2e-3 ~ 2.8e-3，cos ≥ 0.999996，norm_ratio ≈ 1.0000 ± 2e-4。

20-seed 表征与冻结阈值（开关 `SGLANG_FP8_DECODE_CHARACTERIZE=1`）：

| 指标 | 20-seed worst | 冻结阈值（代码常量，`test_flashinfer_fp8_decode.py`） |
|---|---|---|
| rel_l2 | 3.08e-3 | `FROZEN_REL_L2 = 3.9e-3` |
| cos | 0.99999523 | `FROZEN_COSINE = 0.9999940` |
| norm_ratio 偏离 | 9.4e-4 | `FROZEN_NORM_RATIO = (0.9988, 1.0012)` |

回归：`test_torch_native.py` + 既有 `test_flashinfer.py` 共 54 subtests 全通过
（fixture 扩展未破坏旧测试）。

### 1.4 C — 模型端到端（离线回放，替代服务内 decode 切换）

采集：`--debug-tensor-dump-output-folder --debug-tensor-dump-layers 3` 启动
Qwen3.8-27B-NVFP4 + `--kv-cache-dtype fp8_e4m3`，单请求（61 prompt tokens）
greedy 生成 8 步，dump layer 3 的 qkv_proj / attn 输出。

回放（`compare_fp8_decode.py`）重建链路：dump qkv_proj → gemma RMSNorm →
partial RoPE（64 维 / 32 对 / theta=1e7，fp32）→ bf16 → 按生产 pool 代码写
FP8 cache → `torch_fp8_radix_decode_reference` vs dump `model.layers.3.attn`
（gate 前 6144 维）。

结果 8/8 decode 步 PASS：rel_l2 3.3e-3 ~ 7.2e-3、cos 0.999974 ~ 0.999995、
norm_ratio 0.9980 ~ 1.0003。真实数据略高于合成冻结阈值属预期（序列更长、
KV 分布非均匀），按服务级硬上限判定通过。

### 1.5 结论

A/B/C 全部通过：decode attention 与 Radix / page-table / FP8 scale 语义已正确
复制，进入 L2（MXFP4 codec）。

---

## 2. L2 验收：MXFP4 codec + PLAIN 接入（2026-08-31 完成）

**判定标准**：

1. MXFP4 生产 codec 与独立 OCP oracle 在 CPU/CUDA 上 bit-exact；
2. `MXFP4KVCacheMethod`（PLAIN→BF16 access）接入框架后，FlashInfer
   prefill/decode 与 Torch MXFP4 golden 数值等价；
3. 真实模型端到端回放通过；
4. 回归不破坏 L1。

### 2.1 交付物（全部含于 commit `2f22cc9d79`）

生产代码：

| 文件 | 变更 | 内容 |
|---|---|---|
| `python/sglang/srt/layers/quantization/kvfp4_tensor.py` | +146 | `MXFP4KVQuantizeUtil`：OCP MX v1.0 §6.3 严格实现——block-32、每 KV head 沿 head_dim 独立分块（不跨 head）、E2M1 数据、E8M0 scale；scale = 2^floor(log2(amax)) / 4；E2M1 转换 saturating RNE；全零块 scale=2^-127；含 NaN 块 scale=0xFF（整块 NaN）；±Inf 块 scale=2^127 且元素符号饱和。scale 以原始 byte 存储，`scale_buffer_view_dtype()` 暴露 `float8_e8m0fnu`。保持 eager 而非 `@torch.compile`（原因见发现 F1） |
| `python/sglang/srt/layers/quantization/fp4_kv_cache_quant_method.py` | +154 | `MXFP4KVCacheMethod(name="mxfp4", SCALE_BLOCK_SIZE=32)`；access 规则仅 flashinfer prefill/decode PLAIN→BF16；`resolve_kv_cache_quant("mxfp4")` 解除保留报错 |
| `python/sglang/srt/layers/attention/flashinfer_backend.py` | +29 | `flashinfer_kv_cache_dtype` 不再从 workspace 标志推断，改从活跃 phase 的 access.attention_kv_dtype 解析（PLAIN 量化读=BF16、DEQUANT_WORKSPACE=FP8），多 dtype 混用报错；MXFP4 下 `kv_cache_scales_valid=False`，checkpoint FP8 scales 不再参与（防 §7 descale bug 复现） |
| `python/sglang/srt/server_args.py`、`python/sglang/srt/mem_cache/kv_cache_dtype.py` | +7 / +2 | `--kv-cache-dtype` choices 加入 mxfp4；`configure_kv_cache_dtype` 接入 |
| `python/sglang/srt/arg_groups/kv_cache_hook.py` | +28 | fail-fast：mxfp4 + 非 flashinfer backend / 未 disable-cuda-graph / MLA，在参数解析阶段直接拒绝 |
| `python/sglang/srt/mem_cache/memory_pool.py`、`python/sglang/srt/model_executor/pool_configurator.py`、`python/sglang/srt/mem_cache/kv_cache_configurator.py` | +5 / +43 / +4 | `_create_quantized_buffers` 强制 v_head_dim==head_dim（当前 quantized API 只传单一 head_dim）；容量估算按 recipe 区分（见发现 F2） |

测试与 oracle：

| 文件 | 变更 | 内容 |
|---|---|---|
| `python/sglang/test/kits/attention_unittest/attention_methods/mxfp4_decode_attention.py` | 新增 | 独立 oracle（`mxfp4_quantize_reference` 等），不调用生产实现，避免自证；共享 decode 数学抽为 `fp8_decode_attention.torch_radix_decode_from_effective_kv`（GQA + fp32 QK/softmax/PV），FP8/MXFP4 wrapper 共用；L1 函数名/返回值不变 |
| `python/sglang/test/kits/attention_unittest/attention_methods/fp8_decode_attention.py` | +76 | 抽出共享 decode 核心（见上行） |
| `test/registered/unit/layers/quantization/test_mxfp4_kv_codec.py` | 新增 | codec conformance：CPU 门禁 + CUDA byte parity |
| `test/registered/unit/layers/quantization/test_fp4_kv_cache_quant_method.py` | +97 | method 契约 / 容量 |
| `test/registered/attention/unittests/dense/test_mxfp4_pool_cuda.py` | 新增 | `TestMxfp4PoolWriteReadMove`：真实 pool 写入 / PLAIN 读 / slot move（data 与 scale 同时移动） |
| `test/registered/attention/unittests/dense/test_flashinfer_mxfp4_plain.py` | 新增 | `TestFlashInferMxfp4PlainDecode`（decode 差分 10 例）、`TestFlashInferMxfp4PlainPrefill`（3 例）、`TestMxfp4PlainCharacterization`（20-seed 表征） |
| `test/registered/unit/server_args/test_kv_cache_hook_mxfp4.py` | 新增 | `TestMxfp4KvCacheHook`：fail-fast 校验 |
| `compare_mxfp4_decode.py`（仓库根目录） | 新增 | 真实 dump 回放：复用 FP8 脚本的 dump 解析 / norm / RoPE 重建，OCP oracle 编码 + 共享 decode 核心；`--fp8-dump-dir` 提供 MXFP4 vs FP8 质量对比 |

### 2.2 codec conformance（`test_mxfp4_kv_codec.py`）

覆盖：全部 E2M1 code、RNE 中点、饱和、scale 指数边界、全零块、NaN/±Inf、
head 隔离（scale 块不跨 head）、partial block（不足 32 元素的 padding）。
生产实现与 oracle 的 packed/scale bytes 在 CPU/CUDA 均 bit-exact。

### 2.3 差分测试（`test_flashinfer_mxfp4_plain.py`）

decode 10 例 + prefill 3 例通过。20-seed 表征（开关
`SGLANG_MXFP4_PLAIN_CHARACTERIZE=1`）与冻结阈值（worst × 1.25，
worst = mqa_hd64 rel_l2 2.46e-3）：

| 指标 | 20-seed worst | 冻结阈值（代码常量，`test_flashinfer_mxfp4_plain.py`） |
|---|---|---|
| rel_l2 | 2.46e-3 | `MXFP4_PLAIN_FROZEN_REL_L2 = 3.1e-3` |
| cos | — | `MXFP4_PLAIN_FROZEN_COSINE = 0.9999950` |
| norm_ratio | — | `MXFP4_PLAIN_FROZEN_NORM_RATIO = (0.99950, 1.00050)` |

与 L1 FP8 阈值同量级：PLAIN 读物化整层 BF16 后两侧输入相同，残差只来自
kernel 累加顺序。服务级硬上限与 L1 相同。

### 2.4 真实 Qwen 端到端（`compare_mxfp4_decode.py`，layer 3，prompt 63 tokens + 8 decode 步）

等价性 8/8 PASS：rel_l2 3.9e-3 ~ 7.7e-3、cos 0.999971 ~ 0.999992、
norm_ratio 0.9993 ~ 1.0003 —— FlashInfer MXFP4 PLAIN 与 OCP oracle 数值等价。

### 2.5 关键发现（现象 → 根因 → 处置）

| # | 现象 | 根因 | 处置 |
|---|---|---|---|
| F1 | `@torch.compile` 的 `slice + pad-to-even + nibble-pack` 图形跳过最后一个输出字节的写入，输出含陈旧缓冲区内容；同进程内确定、跨进程不同，曾伪装成"滞后一次调用" | torch 2.11 inductor 误编译 | 生产 codec 保持 eager（纯 tensor 仍可 CUDA graph 捕获），L3 性能阶段再评估。**通用规则：bit-exact golden 场景，编译输出必须与 eager oracle 逐字节对比后才可信** |
| F2 | 容量预算需要额外口径 | MXFP4 物理 17 KiB/token（16 层 packed 16 KiB + E8M0 scale 1 KiB）；PLAIN 读同时物化一整层 BF16 K/V | 预算按 ~21 KiB/token 预留；`method.compute_cell_size` 与 pool_configurator 两处公式一致，测试锁定。实测 Qwen server 84096 tokens（FP8 为 82479；mem-fraction 不同不可直接比） |
| F3 | extend/prefill 差分曾呈现 ~0.1 rel_l2 | FlashInfer ragged prefill 的当前 chunk 用未量化 raw bf16 K/V，仅 prefix 从 cache 读取；oracle 只用 cache 来源时，会把 MXFP4 量化误差本身计成"差异" | extend oracle 必须混合两种 KV 来源（raw 当前 chunk + 反量化 prefix） |
| F4 | fail-fast 拦截生效 | — | mxfp4 + 非 flashinfer / 未 disable-cuda-graph / MLA 均在参数解析阶段被拒；server 启动实测命中 cuda-graph 拦截 |

### 2.6 codec 质量表征（信息性，非验收阈值）

MXFP4 vs FP8 同请求（`--fp8-dump-dir`）：单层 attention 输出 rel_l2 0.41 ~ 0.75、
cos 0.75 ~ 0.92、norm_ratio 0.86 ~ 1.10；贪心生成首个 token 即分叉
（MXFP4 "The..." vs FP8 "User..."）。这是 block-32 E2M1 相对 E4M3 的真实精度
代价（KV 噪声改变 softmax 集中模式），L3 kernel 对齐时以此为背景；
`mxfp4` 启动日志保留 accuracy-drop 警告。

### 2.7 回归

L1 FP8 golden + dense 全量 + server_args 全量：246 passed / 103 subtests。

### 2.8 结论

L2 验收完成：`--kv-cache-dtype mxfp4` 在受限组合（flashinfer + MHA +
disable-cuda-graph）下端到端可用，等价性达标；进入 L3 前的 oracle 链
（OCP codec → Torch MXFP4 decode → FlashInfer PLAIN）已闭环。

---

## 3. 复跑指南

```bash
# L1: A/B 单测（需 CUDA + flashinfer）
python -m pytest test/registered/attention/unittests/dense/test_flashinfer_fp8_decode.py -q
# L2: 差分 + 表征
python -m pytest test/registered/attention/unittests/dense/test_flashinfer_mxfp4_plain.py -q

# 20-seed 表征（较慢）
SGLANG_FP8_DECODE_CHARACTERIZE=1 \
  python -m pytest test/registered/attention/unittests/dense/test_flashinfer_fp8_decode.py -q
SGLANG_MXFP4_PLAIN_CHARACTERIZE=1 \
  python -m pytest test/registered/attention/unittests/dense/test_flashinfer_mxfp4_plain.py -q

# L2: codec / method 契约 / pool / fail-fast
python -m pytest test/registered/unit/layers/quantization/test_mxfp4_kv_codec.py -q
python -m pytest test/registered/unit/layers/quantization/test_fp4_kv_cache_quant_method.py -q
python -m pytest test/registered/attention/unittests/dense/test_mxfp4_pool_cuda.py -q
python -m pytest test/registered/unit/server_args/test_kv_cache_hook_mxfp4.py -q
```

真实模型离线回放（先按 §1.4 / §2.4 的方式用
`--debug-tensor-dump-output-folder --debug-tensor-dump-layers 3` 采集 dump，
FP8 请求与 MXFP4 请求需各自采集一份）：

```bash
python3 compare_fp8_decode.py --dump-dir /tmp/sgl_fp8_dump --layers 3 --max-steps 8
python3 compare_mxfp4_decode.py --dump-dir /tmp/sgl_mxfp4_req --layers 3 --max-steps 8 \
  --fp8-dump-dir /tmp/sgl_fp8_dump   # 可选：附 MXFP4 vs FP8 质量表征
```

mxfp4 真实服务启动的受限组合（fail-fast 会拒绝其它组合）：

```
python -m sglang.launch_server --model-path <Qwen3.8-27B-NVFP4> \
  --kv-cache-dtype mxfp4 --attention-backend flashinfer \
  --disable-cuda-graph --mem-fraction-static 0.80 --context-length 32768
```

## 4. 与原笔记的章节映射

| 原笔记内容 | 现位置 |
|---|---|
| 原 §11.4 末尾 "L1 验收记录（2026-08-31 完成）" | 本文 §1 |
| 原 §11.9 "L2 验收记录（2026-08-31 完成）" | 本文 §2 |
| 路线图设计部分（§11.1-§11.8 中未被验收记录覆盖的内容） | 保留在原笔记 §11，未移动 |

---

## 3. L3 验收：native triton decode + 融合写 kernel + CUDA graph（2026-09-04）

**判定标准**（对应路线图 §11.7/§11.8）：

1. triton dequant 微内核 vs 独立 OCP oracle 元素精确；
2. native decode kernel vs L2 Torch golden 差分（双层：standalone kernel 级 +
   真实 backend 集成级），20-seed 表征后冻结阈值；
3. 服务级 dump 回放（真实模型 layer 3）对 OCP oracle 达服务级硬上限；
4. 融合写 kernel 与 eager codec / OCP oracle bit-exact（含 slot 0 保留契约）；
5. CUDA graph capture/replay 字节确定性；
6. 回归不破坏 L1/L2。

### 3.1 交付物

kernel 与集成（commit `f0a95c27c1` + 后续优化提交）：

| 文件 | 内容 |
|---|---|
| `python/sglang/kernels/ops/attention/mxfp4_decode_attention.py` | dequant 微内核（`mxfp4_dequant_fwd`）、per-head stage1（MHA，fp32 元素级）、grouped stage1（GQA/MQA，位构造解包 + `tl.dot`）、dot_scaled 实验变体（默认关）、stage2 split 合并、host 分发。复用 stock `_extract_kv_strides`；E8M0 scale 用精确位构造 `_e8m0_to_f32`（避开 approximate-ex2 的 FTZ）；E2M1×E8M0→bf16 用纯整数位构造 `_e2m1_scale_to_bf16` |
| `python/sglang/srt/layers/quantization/fp4_kv_cache_quant_method.py` | mxfp4 access registry：prefill PLAIN→flashinfer（L2 不变）、decode PLAIN→flashinfer（验证路径）、decode NATIVE→triton（L3 新增） |
| `python/sglang/srt/layers/attention/triton_backend.py` | forward_decode 检测 mxfp4 quantized pool → `get_raw_kv_buffer` + 新 kernel；mxfp4 时 `max_kv_splits` 提升 floor 至 32；lean/score_mod/DCP/SWA/logit-cap fail-fast |
| `python/sglang/srt/arg_groups/kv_cache_hook.py` | mxfp4 允许 `(flashinfer, flashinfer)` 与 `(flashinfer, triton)`；PLAIN decode 保持 `--disable-cuda-graph`（物化不 graph-safe），triton native 解禁 |
| `python/sglang/srt/layers/quantization/fp4_kv_cache_quant_method.py`（写侧） | `MXFP4KVCacheMethod.quantize_and_store` 切换融合 kernel，eager codec 保留为回退与参考 |
| `python/sglang/kernels/ops/quantization/mxfp4_quant.py` | 融合 quant+store 写 kernel：block amax → E8M0 → E2M1 saturating RNE → scatter data+scale，slot 0 保留，无 host sync |

测试与工具：

| 文件 | 内容 |
|---|---|
| `test/registered/attention/unittests/dense/test_triton_mxfp4_native_decode.py` | A 步 dequant 精确性 + B 步 decode 差分 15 例 + 20-seed 表征 + CUDA graph capture/replay |
| `test/registered/attention/unittests/dense/test_triton_mxfp4_native_integration.py` | 集成差分 10 例（真实 pool + TritonAttnBackend 全链路，含写契约断言）+ 表征 |
| `test/registered/kernels/ops/quantization/test_mxfp4_quant_store.py` | 融合写 kernel bit-exact 契约（特殊值/tie/部分块/slot0/门控） |
| `scripts/playground/bench_mxfp4_decode_kernel.py` | kernel 级 micro-bench（shape/参数 sweep + roofline 口径） |
| `scripts/playground/probe_dot_scaled.py` | SM120 `tl.dot_scaled`（FP4 MMA）编译/数值/性能探测 |
| `scripts/playground/probe_bitconstruct_dequant.py` | 位构造解包全域（16 code × 256 scale byte）精确性探测 |
| `scripts/playground/run_humaneval_spawn.py` | 精度评测驱动（spawn 绕开 filelock fork 审计） |

### 3.2 A 步：dequant 微内核 vs OCP oracle

全 E2M1 code、E8M0 指数边界（byte 1..254）、全零块、NaN scale 块、partial block、
奇数 logical dim、bf16/fp32 输出：全部元素精确。两个已记录的例外（均非生产域）：

- scale byte 0（2^-127，fp32 subnormal）与非零 code 组合：oracle 的 `torch.exp2`
  subnormal 结果带 1 ULP 误差，kernel 位构造反而精确；生产仅 byte 0 + 全零 code
  配对（值 = 0，两侧一致）；
- `tl.exp2` 近似指令 FTZ 会把 2^-127 scale 清零 → E8M0 一律走位构造（`_e8m0_to_f32`）。

### 3.3 B 步：decode 差分（对 L2 golden），冻结阈值

| 层级 | 路径 | 20-seed worst | 冻结阈值（代码常量） |
|---|---|---|---|
| standalone（fp32 输出） | MHA（per-head kernel） | rel_l2 2.61e-7 | `MXFP4_TRITON_FROZEN_REL_L2 = 3.5e-7`，cos ≥ 0.9999997，norm ±2e-7 |
| standalone（fp32 输出） | GQA/MQA（grouped kernel，位构造 + tl.dot） | rel_l2 1.45e-3（splits1） | `MXFP4_TRITON_GROUPED_FROZEN_REL_L2 = 1.85e-3`，cos ≥ 0.9999986，norm ±6e-4 |
| integration（bf16 输出，真实 backend） | 同上两路径合并 | rel_l2 2.59e-3（mqa） | `MXFP4_TRITON_INTEG_FROZEN_REL_L2 = 3.3e-3`，cos ≥ 0.9999956，norm ±1e-3 |

grouped/integration 的残差来源：p→bf16（PV dot 前）+ tensor-core 累加顺序 +
RadixAttention bf16 输出舍入——与 stock triton kernel / FlashInfer 同类语义，
与 L2 PLAIN 冻结值（3.1e-3）同量级。位构造解包与手动解包（fp32 乘 scale 后
cast bf16）逐位一致：dequant 值（E2M1×2^k）在 bf16 精确可表示。

### 3.4 融合写 kernel 契约

`quant_store_kv_mxfp4` vs eager `MXFP4KVQuantizeUtil.batched_quantize` vs OCP
oracle 三方 bit-exact：随机 bf16/fp16（dim 32/48/64/128/256）、特殊值
（NaN/±Inf/±0/bf16 subnormal/极值）、RNE tie 中点、slot 0 保留、门控（fp32/
CPU/非连续末维 → 回退 eager）。L2 的 pool 写契约测试与集成写断言在融合路径
下继续通过。

### 3.5 CUDA graph

kernel 级：decode（stage1+stage2）与融合写 kernel 各自 capture → 两次 replay
字节一致且与 eager 字节一致。服务级：`(flashinfer, triton)` + CUDA graph 启动
capture 成功，decode batch 日志 `cuda graph: True`，greedy 生成确定性
（8/64 token 请求重复输出一致）。

### 3.6 端到端回放（真实模型 layer 3）

`(flashinfer, triton)` 服务 + `--debug-tensor-dump-output-folder`，54-token
prompt greedy 8 步，`compare_mxfp4_decode.py` 回放：8/8 PASS，
rel_l2 3.1e-3 ~ 6.4e-3、cos ≥ 0.999979 —— 与 L2 PLAIN 回放（3.9e-3~7.7e-3）
同量级，服务级硬上限内。

### 3.7 性能总表（RTX 5090，mem-fraction 0.80，除注明外）

| 指标 | A = FP8 flashinfer | B = mxfp4 PLAIN | C = mxfp4 triton native（最终态） |
|---|---|---|---|
| TPOT 短文（512/128） | 23.7ms | 152.7ms | **24.5ms** |
| TPOT 长文（4096/128） | 23.8ms | 152.7ms | **25.9ms** |
| bench 总吞吐（长文） | 36.6 tok/s | 6.3 tok/s | 40.7 tok/s |
| TTFT 长文（无 radix 命中） | 652ms | 7050ms | 986ms |
| max_total_num_tokens | 55188 | 84096 | 84096（容量 1.52×） |
| CUDA graph | ✅ | ❌（hook 强制） | ✅ |

### 3.8 优化轨迹（每步：现象 → 根因 → 改动 → 证据）

初始实现（T0）为 per-head grid + splits=8，长文 TPOT 47.5ms。三轮优化：

| 步骤 | 现象/根因 | 改动 | kernel 级证据 | 服务级证据 |
|---|---|---|---|---|
| T1 grouped kernel | GQA（24q/4kv）下同一 kv_head 的 packed K/V 被 6 个 q head 程序各读一遍（读放大 6×）；短文被 L2 掩盖、长文暴露 | 每 program 服务一个 kv head 的整个 query group，K/V 解包一次共享，`tl.dot`（bf16 tile，dequant 值 bf16 无损） | （与服务级同批验证） | 长文 47.5→30.5ms；短文 26.7→24.6ms |
| T2 splits 8→32 | bs=1 长文网格仅 kv_heads×splits = 32 CTA vs 170 SM，kernel 是 **latency-bound 而非带宽-bound**（sweep 显示 seq512 时间与 splits 无关、4096 强相关） | `triton_backend` mxfp4 分支提升 `max_kv_splits` floor 至 32（显式 server arg 优先） | 4096 seq：155.5µs（per-head+splits8）→ 66.7µs（grouped+splits32，T1+T2 组合） | 长文 30.5→25.7ms |
| T3 位构造解包 | 解包热路径含 SFU `exp2` + fp32 乘法链（实验 1 归因出 ~97µs/层解包代价） | `_e2m1_scale_to_bf16`：纯 int32 移位/或直接拼 bf16 位模式（含 normal/subnormal 桥、±0、inf、NaN 优先级） | 4096 seq 68.9→63.8µs（−7%）；8192 seq 98.4→88.1µs（−10%） | 25.9ms±噪声（收益 ~0.1ms，在测量噪声内） |

参数 sweep 补充结论（micro-bench）：BLOCK_N=128 比 64 慢 2~5×（tile 过大）；
num_warps=8 一致优于 4；seq=512 存在 ~48µs 的 eager launch 固定下限
（CUDA graph 下更小）。

### 3.9 roofline 与带宽账本：为什么 KV 带宽节省对 TPOT 收益不显著

**decode 每 step 的 DRAM 读取构成（bs=1，seq=4096）**：

| 项 | FP8 KV | MXFP4 KV | 说明 |
|---|---|---|---|
| 权重（64 层 + embed/lm_head） | 21.5 GB（≈12.0ms @1.79TB/s） | 21.5 GB | **不随 KV dtype 变化** |
| KV cache 读取 | 128 MiB（71µs） | 68 MiB（38µs） | 16 层全注意力；GDN 48 层不读 KV |
| **KV 差异** | — | — | **33µs ≈ TPOT 的 0.14%** |

结论一：**decode memory-bound 的主体是权重，不是 KV**。27B 权重每 step 必读
21.5GB，而 KV 只有 0.07~0.13GB——KV dtype 从 32KiB/token 降到 17KiB/token
省出的 60MiB/step，在 23.8ms 的 TPOT 里理论上只值 33µs。这不是 kernel 实现
问题，而是 workload 结构：该模型只有 16 层全注意力（其余 48 层是状态固定的
GDN 线性注意力），且单卡 32GB 限制了 bs×seq 的规模。

结论二：**KV 带宽收益 ∝ batch×seq_len**。令 KV 占比超过 TPOT 的 20% 需要
`bs × seq × 15KiB（节省量） > 0.2 × 21.5GB`，即 `bs × seq > ~30 万 token`
（例如 bs=16×19k、bs=32×9k）。此时理论 TPOT：FP8 ≈ 21.0ms vs MXFP4 ≈ 16.7ms
（20% 优势）。但 32GB 单卡不可达：mxfp4 容量 134k slots（扩 mamba 后）且
mamba 状态池把并发 cap 在 8~16，fp8 侧容量更小。

结论三（归因实验）：同 bs=1 长文下，FP8+flashinfer 23.80ms / FP8+triton
24.15ms / mxfp4+triton 25.67ms（位构造前）→ backend 实现差异仅 0.35ms，
mxfp4 的 1.55ms（≈97µs/层）是**解包的计算代价**——mxfp4 kernel 读的字节
更少（省 38µs/层带宽）但解包更贵（费 ~97µs/层 ALU/延迟），净亏。位构造
后解包代价大幅收敛，服务级差距保持 ~2ms（其中 attention kernel 差
~0.7ms，其余为 launch 结构与 stage2）。

结论四（tensor core 澄清）：attention 的 FLOPs 仅 ~1.6 GFLOP/step
（24 heads × 4096 × 256 × 2 × 2 × 16 层）≈ **8µs @ 209 TFLOPS**——TC 利用率
根本不构成 TPOT 瓶颈；`tl.dot`（bf16 tensor core）已在使用。瓶颈在喂给
TC 之前的解包 ALU/延迟（T3 已收敛）与 kernel 占用率（T2 已收敛）。

**最终价值定位**：在单卡 32GB + 该 hybrid 架构（16 全注意力 + 48 GDN）+
bs≤8 的约束下，mxfp4 KV 的当前可兑现价值是**容量 1.52×**（同显存多装
52% context）+ 与 FP8 同级的 decode 速度（差距 ~9%）；吞吐优势场景
（大 batch × 长 context，KV 带宽占比 >20%）需要更大显存部署。若未来继续
压缩 TPOT，方向在权重路径与 GDN 层（占 ~97%），而非 attention kernel。

### 3.10 nsys 证据

`nsys profile`（`/tmp/mxfp4_kernel_profile.nsys-rep`；容器无 GPU 计数器
权限，`--gpu-metrics-devices` 不可用，仅 kernel 时间线），4096 seq /
splits32 / BN64 / warps8，25 iterations：

| kernel | 中位时间 | 占比 |
|---|---|---|
| `_mxfp4_grouped_decode_stage1_kernel` | 34.6µs | 83.5% |
| `_mxfp4_decode_stage2_kernel` | 6.2µs | 15.0% |
| 合计每层 | **40.8µs** | — |

16 层 × 40.8µs ≈ 653µs = **最终态 TPOT（25.9ms）的 ~2.5%**——attention
kernel 的进一步优化（persistent kernel、stage 融合、TMA 预取等）对端到端
收益上限 <1%，优化到此收敛。对照：优化前（splits8）stage1 ≈155µs/层。

### 3.11 SM120 feature 结论（tl.dot_scaled / FP4 MMA）

triton 3.7.1 的 `tl.dot_scaled` 在 **sm_120 编译通过**，e2m1/e8m0 packing
约定与池布局完全一致（低 nibble = 偶数元素，scale 沿归约维每 32 元素），
隔离 QK-only probe 比 手动解包快 34%（14.5 vs 21.9µs）。但在完整 decode
loop 内反而慢 2.2×（153.7 vs 68.9µs）：rhs 列数小（BLOCK_H=16）、与 PV
手动解包/softmax 混排、每迭代 64 行小 tile，FP4 MMA 固定开销吞掉收益；
且 PV 的 scale 沿**输出维**（非归约维），dot_scaled 语义无法表达。保留
`use_dot_scaled` 实验开关，默认关。probe：`probe_dot_scaled.py`。

### 3.12 关键发现

| # | 现象 | 根因 | 处置 |
|---|---|---|---|
| G1 | triton jit 读普通 Python 全局报 NameError | constexpr 全局须 `tl.constexpr(...)` 实例化（注解形式不支持） | 模块级 `_MXFP4_BLOCK = tl.constexpr(...)` |
| G2 | 手写 E2M1 值表错（按 2 位尾数实现） | E2M1 是 1 位尾数：`exp=(code>>1)&3, mant=code&1` | A 步差分即时捕获；修正后全 table 位精确 |
| G3 | `tl.exp2(-127)` 结果为 0（FTZ） | approximate ex2 flushes subnormals | E8M0 一律位构造（`_e8m0_to_f32`）；fp32 乘法不 FTZ |
| G4 | decode 长文 TPOT 47.5ms | GQA 逐 head 重读 K/V（6× 读放大）+ splits=8 网格 32 CTA vs 170 SM | grouped kernel + splits floor 32 |
| G5 | `tl.dot_scaled` 完整 kernel 内慢 2.2× | 小 rhs tile + 与 PV/softmax 混排；PV scale 沿输出维语义不匹配 | 保留实验开关默认关 |
| G6 | prefill graph capture 显存耗尽（mamba 扩容场景） | breakable prefill graph 42 段 capture 吃尽余量 | bs 扩容实验需 `--disable-prefill-cuda-graph` |
| G7 | human_eval 评分 fork 崩溃 | filelock≥3.32 fork 审计 | spawn 启动方法（`run_humaneval_spawn.py`） |

### 3.13 复跑指南

```bash
# L3 差分 + 表征（CUDA）
python -m pytest test/registered/attention/unittests/dense/test_triton_mxfp4_native_decode.py -q
SGLANG_MXFP4_TRITON_CHARACTERIZE=1 python -m pytest \
  test/registered/attention/unittests/dense/test_triton_mxfp4_native_decode.py -q -k Characterization
python -m pytest test/registered/attention/unittests/dense/test_triton_mxfp4_native_integration.py -q
# L3 写契约
python -m pytest test/registered/kernels/ops/quantization/test_mxfp4_quant_store.py -q
# kernel micro-bench / SM120 probes
python scripts/playground/bench_mxfp4_decode_kernel.py --seqs 4096 --splits 32
python scripts/playground/probe_bitconstruct_dequant.py
python scripts/playground/probe_dot_scaled.py
# 服务级（生产组合）
python -m sglang.launch_server --model-path <Qwen3.8-27B-NVFP4> \
  --kv-cache-dtype mxfp4 --prefill-attention-backend flashinfer \
  --decode-attention-backend triton --mem-fraction-static 0.80 --context-length 32768
python3 compare_mxfp4_decode.py --dump-dir <dump> --layers 3 --max-steps 8
```

### 3.14 精度验收（2026-09-04~05 补录）

| 评测 | A = FP8 flashinfer | C = mxfp4 triton native | 差异归因 |
|---|---|---|---|
| **Teacher-forcing PPL**（32,760 tokens，仓库文档 8×4096 段） | **4.081**（mean_nll 1.406） | **16.862**（mean_nll 2.825） | **mxfp4 量化的直接 KV 精度代价：Δnll +1.42（PPL 4.1×）** |
| HumanEval 60 题 × 5 samples（pass@1） | **0.883** | **0.700** | 18.3pp，与 PPL 方向一致 |
| AIME25 8 题 greedy（pass@1） | **0.875** | **0.875** | n=8 无鉴别力（见 caveat） |

- **PPL 方法**：`/generate` + `input_ids + return_logprob + logprob_start_len=0`
  读 `meta_info.input_token_logprobs`（teacher forcing，每位置的 attention 读
  pool 中量化后的 KV，与生成路径完全同源）；文本源为仓库文档 8×4096 段；
  跨启动确定（C 重启两次 seg1 mean_nll 完全一致 2.3042）。脚本：
  `eval_ppl_kv.py`。
- **PPL 是最直接的精度证据**：不经过采样/提取/任务执行环节，逐位置 NLL
  直接度量量化 KV 下的预测质量。4.1× 的 PPL 代价与 §2.6 的早分叉记录、
  HumanEval 的 18.3pp 完全自洽——**mxfp4（block-32 E2M1）对 KV 的量化精度
  远低于 fp8 E4M3**，这是格式特性而非实现缺陷（kernel 已由差分链排除）。
- **标准框架内的精度提升自由度已排查并排除**：
  (a) scale round mode（OCP 合法自由度，floor vs ceil）——合成分布实验
  （uniform/normal/spiky 各 40 万 block）：floor 10.9~16.9% vs ceil
  11.4~26.6% 相对 RMS，**当前 floor 实现在 KV 类分布下已是最优**（ceil 把
  值域搬进 E2M1 步长更粗的中段），排除；
  (b) 纳入实现链的各环节（融合写 kernel bit-exact、位构造解包精确、
  attention bf16 tile + fp32 accumulate）均已验证无损或达标；
  (c) **剩余损失 = E2M1 格式固有**（1-bit mantissa，元素相对 RMS ~11-17%，
  为 fp8 E4M3 的 3~4 倍），在保留标准 mxfp4 的前提下不可消除。
- HumanEval 18.3pp 的归因：长代码生成输出长、对累积 KV 误差敏感，
  PPL 代价在此任务上放大为通过率损失；AIME 的一致（n=8）不能作为
  无回归证据（无鉴别力，见 caveat）。
- **任务差异与 AIME 样本量的统计 caveat**：HumanEval 与 AIME 的表现差异
  主要来自评分机制（代码 = 全部隐藏测试零容错执行；数学 = 最终 boxed 答案
  匹配，推理链有路径容错）与样本量（300 vs 8 个二值样本）。18.3pp 在
  HumanEval 上 ~5σ 显著；而 AIME 的 "0.875 = 0.875" 在 n=8 下无鉴别力：
  若 mxfp4 在 AIME 上同样存在真实差距，观察到当前一致结果的概率
  仍约 19%。mxfp4 的精度代价应视为全局性的（PPL 已直接量化）；
  如需按任务细分，需跑 30 题全集 × 多样本（后续可选）。
- **评分基础设施修复**（对本仓库共享评测文件）：
  `simple_eval_humaneval.find_code` 原实现未剥离 thinking 模型的
  `<think>...</think>` 块，签名定位 `find(":\n    ")` 落在推理文本中，
  导致多数 completion 提取失败（首跑 A 假分数 0.233、C 假分数 0.017）。
  修复后先取最后一个 `</think>` 之后的文本再提取。附带产物：
  `run_humaneval_spawn.py`（spawn 绕开 filelock fork 审计）、
  `eval_humaneval_5sample.py`（逐样本持久化、防覆盖的独立评测器）、
  `rescore_humaneval_html.py`（从报告 html 离线重评分）。
- **数据事故记录**：`run_eval` 报告文件名固定（`humaneval__<model>.*`），
  同 model 的先后评测相互覆盖，A 首跑与 C 首跑的原始 html 已丢失；
  C 的修正分数系覆盖前离线重放所得（1-sample 子集 0.684，与 5-sample
  0.700 交叉一致）。此后所有产物归档 `eval_results_l3/<配置>_<口径>_<时间戳>`。
- B（PLAIN）缩减版未跑：C 的 decode kernel 已对 L2 golden（PLAIN 同源 cache
  语义）冻结阈值达标，B 的评分必然落在同一 cache 精度水平，不再重复验证。

### 3.15 结论

L3 验收完成：`--kv-cache-dtype mxfp4 --prefill-attention-backend flashinfer
--decode-attention-backend triton` 在 CUDA graph 开启下端到端可用；native
decode kernel 对 L2 golden 冻结阈值达标，融合写 kernel 与 OCP oracle
bit-exact；TPOT 与 FP8 生产基线差距收敛到 ~9%（25.9 vs 23.8ms，bs≈1），容量
1.52×。decode 每 step 的 DRAM 读取主体是权重（~97%），KV 带宽收益在单卡
小 batch 场景天然稀释——mxfp4 的当前价值定位为容量 1.52× + 与 FP8 同级的
decode 速度；吞吐优势场景（大 batch × 长 context）需更大显存部署。

精度验收：**teacher-forcing PPL（32.8k tokens）16.86 vs 4.08（4.1×）是
mxfp4 KV 精度代价的直接量化**；HumanEval 0.700 vs 0.883 与其方向一致
（零容错代码执行的放大），AIME25 小样本（n=8）无鉴别力。kernel 实现已由
差分链排除，PPL 代价系 mxfp4（block-32 E2M1）量化的格式固有精度限制，
且标准框架内的自由度（scale round mode）已实验排除。综合判定：
L3 kernel 实现通过验收；但 **mxfp4 KV 的精度代价（PPL 4.1×、
HumanEval −18pp）显著高于 fp8 KV**，其部署价值需按任务重新权衡：
容量 1.52× 换来的是对精度敏感任务的可测损失。若需降低 PPL，可行方向
均超出标准 mxfp4 存储格式：K/V 混合精度（V 对误差更敏感可用 fp8）、
低秩/稀疏高精度残差补偿、或接受代价用于容量优先场景（后续可选）。
