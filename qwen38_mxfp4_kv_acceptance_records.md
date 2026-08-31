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
