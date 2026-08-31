# MXFP4 L2 Codec 与 FlashInfer PLAIN 验证开发计划

## 范围与已确认决策

- 以已完成的 L1 为基线：保留 `fp8_decode_attention.py`、FP8 冻结阈值和现有测试行为；L2 只替换 KV codec，不改变 radix gather、GQA、attention scaling、fp32 softmax/PV 数学。
- codec 严格采用 OCP MXFP4：每个 KV head 沿 `head_dim` 独立按 32 个元素分块，E2M1 数据、E8M0 共享 scale；不把现有 `fp4_mx_block16` 政名或偷偷改成 block-32。
- §11.6 开放显式 `--kv-cache-dtype mxfp4`，首个支持组合只承诺 FlashInfer prefill + FlashInfer decode，实际交给 FlashInfer 的 K/V 为 PLAIN BF16。
- PLAIN 是正确性验证路径：每层读取时整层反量化，禁止 CUDA graph 并预留一层 BF16 K/V 临时空间；不宣称吞吐收益。prefill 继续使用现有 FlashInfer attention，不新增 prefill kernel。
- 依赖保持现有环境版本；不引入新包。非目标包括 Triton/CUDA fused decode、量化写入优化、其他 attention backend、MLA/cross-attention、HiCache/offload/disaggregation/speculative 组合。

## 严格 OCP MXFP4 Codec

### 生产 codec

在 `python/sglang/srt/layers/quantization/kvfp4_tensor.py` 新增 `MXFP4KVQuantizeUtil`，保留 `FP4MXBlock16KVQuantizeUtil` 原行为：

- API：
  - `batched_quantize(tensor) -> (packed_uint8, scale_uint8)`；
  - `batched_dequantize(packed_uint8, scale_uint8, *, logical_dim, dtype) -> Tensor`。
- 输入逻辑形状为 `[tokens, kv_heads, head_dim]`；block 不能跨 KV head。scale 形状为 `[tokens, kv_heads, ceil(head_dim / 32)]`，数据形状为 `[tokens, kv_heads, ceil(head_dim / 2)]`；末块和奇数 nibble 以 `+0` padding，反量化后裁回 `logical_dim`。
- OCP finite-path 算法固定为：
  ```python
  amax_pow2 = 2 ** floor(log2(amax))
  scale = clamp(amax_pow2 / 4, 2**-127, 2**127)
  element = saturating_round_ties_to_even(x / scale, E2M1)
  ```
  其中 4 是 E2M1 可表示的最大 2 的幂，最大有限幅值仍为 6；E2M1 code 为 `{±0, ±0.5, ±1, ±1.5, ±2, ±3, ±4, ±6}`，偶数元素放低 nibble、奇数元素放高 nibble。
- E8M0 按原始 byte 保存（指数偏置 127），并由 `scale_buffer_view_dtype()` 暴露为 `torch.float8_e8m0fnu`；全零块固定为最小 scale `2^-127`（byte `0x00`）和零 data。
- 特殊值策略显式锁定：含 NaN 的块写 E8M0 NaN（`0xFF`），解码整块为 NaN；只有 ±Inf 时 scale 取最大有限 E8M0，元素按符号饱和到 ±6。生产实现保持纯 tensor、无 `.item()`/host sync，并覆盖 CPU/CUDA。

### 独立 conformance oracle

新增 `python/sglang/test/kits/attention_unittest/attention_methods/mxfp4_decode_attention.py`：

- 用独立、清晰、非 `torch.compile` 的实现提供 `mxfp4_quantize_reference`、`mxfp4_dequantize_reference`，不得调用生产 `MXFP4KVQuantizeUtil`，避免自证。
- 提供 `Mxfp4DecodeDiagnostics`，保存 packed bytes、scale bytes、dequant K/V、scores、probs、output。
- 提供 `torch_mxfp4_radix_decode_reference(...)`：按 `req_to_token` 同时 gather packed data 与 scale，解码有效 K/V 后调用 L1 的同一 attention 数学核心。

## 复用同一 Torch Decode 数学

最小重构 `python/sglang/test/kits/attention_unittest/attention_methods/fp8_decode_attention.py`：

- 抽出 `torch_radix_decode_from_effective_kv(q, k, v, *, scaling, return_diagnostics=False)`，统一执行 KV-head→Q-head GQA 映射、fp32 QK、softmax、PV。
- `torch_fp8_decode_dequant_first_reference` 和新 MXFP4 wrapper 都调用该核心；`torch_fp8_radix_decode_reference` 保留当前 BMM-boundary scale placement，继续与通用核心交叉验证。
- 保持现有函数名、返回值和 imports 兼容，先运行全部 L1 测试证明重构零回归。

## MXFP4 Quant Method、Pool 与容量

在 `python/sglang/srt/layers/quantization/fp4_kv_cache_quant_method.py` 新增独立 `MXFP4KVCacheMethod`：

- `name="mxfp4"`、`SCALE_BLOCK_SIZE=32`、无 checkpoint global scale；实现 `create_buffers`、`quantize_and_store`、`dequantize_kv_tensor`、`dequantize_prev_kv`、`compute_cell_size`。
- packed K/V 使用 uint8；scale buffer 使用 uint8 原始位，物理形状 `[slots, heads, ceil(dim/32)]`；PLAIN 解码输出 BF16；不分配 `dq_k_buffer/dq_v_buffer`。
- `quantize_and_store` 必须 bit-exact 匹配独立 oracle，并以 graph-safe masked scatter 保持保留 slot 0 的 data/scale 不变；非连续 loc、重复 padding loc 和未写 slot 均写测试。
- 当前 pool quantized API 只传一个 `head_dim`，因此本阶段显式要求 `v_head_dim == head_dim`；不满足时在 pool 构造阶段报可读错误。
- 在 `KV_CACHE_ATTENTION_ACCESS_REGISTRY` 中只为 `mxfp4` 注册 FlashInfer 的 prefill/decode `PLAIN` 规则，`attention_kv_dtype=torch.bfloat16`；在 `KV_CACHE_QUANT_REGISTRY` 注册新 method，并让 `resolve_kv_cache_quant("mxfp4")` 正常解析。
- `get_raw_kv_buffer()` 继续提供 packed data + E8M0 typed scale view，为 L3 Triton kernel 保留稳定存储契约；现有 slot move 路径须验证 data/scale 同步移动。

修改 `python/sglang/srt/model_executor/pool_configurator.py` 的 FP4 容量估算，使其按 recipe 区分 block size 和临时内存：

- Qwen 配置的物理 MXFP4 KV 为 17 KiB/token（16 层 packed K/V 16 KiB + scale 1 KiB）。
- PLAIN 运行还需为当前一层整池 BF16 K/V 保留 4 KiB/token，因此 L2 容量预算按约 21 KiB/token 计算，避免池按 17 KiB/token 吃满显存后在整层反量化时 OOM。
- 使用 K/V 各自维度及 `ceil_div` 计算 packed/scale 字节；增加 method 公式与 configurator 公式一致性测试，避免当前硬编码 block-16 漂移。

## CLI、兼容性与 FlashInfer PLAIN 适配

### 参数与构建链

- `python/sglang/srt/server_args.py`：将 `mxfp4` 加入 `--kv-cache-dtype` choices/help；仍仅显式启用，不改变 `auto` 行为。
- `python/sglang/srt/mem_cache/kv_cache_dtype.py`：把 `mxfp4` 映射到 `torch.float4_e2m1fn_x2` 存储 tag，并沿用 PyTorch/CUDA 版本检查。
- `python/sglang/srt/arg_groups/kv_cache_hook.py`：把 `mxfp4` 纳入 FP4 校验；仅允许非 MLA 的 FlashInfer prefill+decode，要求 `--disable-cuda-graph`，对其他 backend/graph/不支持的 pool 组合 fail fast；相关 `prefill-only-disable-kv-cache` 和错误消息同步包含 `mxfp4`。
- `kv_cache_configurator.py` 和 `HybridLinearKVPool` 沿用现有 quant_method 注入链；只补必要的错误检查，不新增专用 pool。Qwen 混合架构仅 16 个 full-attention layers 使用 MXFP4，linear-attention state 不变。

### FlashInfer dtype 修正

在 `python/sglang/srt/layers/attention/flashinfer_backend.py`：

- 当当前 FlashInfer phase 的 access 为量化 `PLAIN` 时，从 `access.attention_kv_dtype` 解析 wrapper 的 KV dtype，而不是继续使用 storage tag `model_runner.kv_cache_dtype=torch.float4_e2m1fn_x2`；MXFP4 prefill/decode 均解析为 BF16。
- 初始化时校验 FlashInfer 活跃 phases 的 effective KV dtype 一致；不一致直接报错，避免一个 wrapper dtype 隐式服务两种数据格式。
- forward 路径继续调用现有 `pool.get_kv_buffer(layer_id)`，由 `_get_key_buffer/_get_value_buffer` 整层解包到 BF16；不走 FP8 dequant workspace，不新增 attention kernel。
- `kv_cache_scales_valid` 必须为 false，确保 Qwen checkpoint 的 FP8 `k_scale/v_scale` 不会再次应用到 MXFP4 写入或 FlashInfer 读取。

## 测试与阈值建立

### 1. OCP codec conformance（CPU 为主，CUDA 镜像）

新增 `test/registered/unit/layers/quantization/test_mxfp4_kv_codec.py`：

- 逐 code 验证 E2M1 16 个 nibble、正负零、低/高 nibble 顺序及 E8M0 指数 byte。
- 覆盖所有 RNE midpoint、饱和、subnormal/underflow、scale 指数切换点、全零、NaN、±Inf。
- 覆盖 head 隔离、block 32 边界、partial block、非连续输入；production util 与独立 oracle 的 packed/scale bytes 必须 bit-exact，dequant 必须逐元素一致。
- CPU 是必跑门禁；CUDA 对相同固定向量做 byte parity，确保设备无语义漂移。

### 2. Method/registry/pool 契约

扩展 `test/registered/unit/layers/quantization/test_fp4_kv_cache_quant_method.py` 并新增必要的 pool CUDA 测试：

- factory/resolve/access rule、buffer shape/dtype、E8M0 view、17 KiB physical footprint 与 21 KiB PLAIN capacity budget。
- `set_kv_buffer` 的 packed data/scale bit-exact、slot 0 保留、未写 slot、随机非连续 loc；`get_kv_buffer` 的整层 BF16 输出与 oracle 一致。
- slot move 同时移动 packed K/V 和 K/V scales；旧 `nvfp4`、`fp4_mx_block16` 行为不变。

### 3. Torch MXFP4 与 FlashInfer PLAIN 差分

新增 `test/registered/attention/unittests/dense/test_flashinfer_mxfp4_plain.py`，并仅按需扩展 `dense_attention.py` 以注入 `quant_method`：

- decode 使用与 L1 相同的 MHA/GQA/MQA、Qwen `24/4/256`、batch、page boundary 和 non-monotonic/interleaved loc 矩阵。
- 实际路径：当前 token 由 `FlashInferAttnBackend.forward_decode` 写入 MXFP4 pool，随后 PLAIN BF16 整层反量化；reference 从同一 raw packed bytes + scale bytes 按 page table gather 后独立解码。
- 增加 prefill 功能测试（不新增 prefill kernel）：覆盖无 prefix 与 prefix reuse/chunked extend，证明现有 FlashInfer prefill 能消费 PLAIN BF16 cache。
- 注入明显不同的 checkpoint `k_scale/v_scale`，验证 MXFP4 路径完全忽略 FP8 checkpoint descale。
- “同一 packed cache 的实现等价性”沿用 L1 硬上限：rel_l2 ≤ 2e-2、cos ≥ 0.999、norm_ratio ∈ [0.98, 1.02]；先跑 20 个固定 seeds，再按 worst×1.25 冻结更紧的 `MXFP4_PLAIN_FROZEN_*`。
- “格式质量”单独报告 Torch MXFP4 vs Torch FP8/BF16 的 cache、score、prob、output 指标，不拿 L1 紧阈值要求不同 codec，也不通过放宽 PLAIN 等价阈值掩盖接入错误。

### 4. 参数、启动与回归

- 扩展 `test/registered/unit/server_args/test_server_args.py` 及 dtype/compatibility 测试：`mxfp4` 可解析；非法 backend、CUDA graph、MLA、prefill-only-no-cache 等组合在模型加载前失败；`auto` 与旧 FP4 recipe 不变。
- 运行 codec/method CPU tests、CUDA codec parity、L1 FP8 golden、现有 FP4 method tests、FlashInfer dense attention 全量回归。

## 真实 Qwen 端到端与离线回放

新增 `compare_mxfp4_decode.py`，复用 `compare_fp8_decode.py` 的 dump 解析、QK norm、partial RoPE 和 metrics，不修改已冻结的 FP8 脚本：

1. 用 Qwen3.8-27B-NVFP4 启动 `--kv-cache-dtype mxfp4 --attention-backend flashinfer --disable-cuda-graph`；使用保守 `--mem-fraction-static`/context，预留 PLAIN BF16 临时内存。
2. 固定 prompt、greedy、8 个 decode steps，采集第一个 full-attention layer 3；再覆盖一个 prefix-cache reuse 或 chunked-prefill 请求。
3. 离线重建 q/k/v，将历史与当前 K/V 用独立 OCP oracle 编成 MXFP4，再执行同一 Torch decode core，与生产 MXFP4 PLAIN dump 的 gate 前 attention output 对比；要求每步满足 PLAIN 等价硬上限且无 NaN/Inf。
4. 用相同请求的 FP8 基线记录 MXFP4 相对 FP8 的逐层输出、top-k/logprob、首个 token 分歧和生成文本；该结果是 codec 质量表征，不与 backend-equivalence 混为一项阈值。
5. 将 20-seed 冻结阈值、端到端结果、17/21 KiB 两种容量口径及 PLAIN 限制补入 `qwen38_attention_survey_notes.md`。

## 完成标准

- OCP 固定向量与 production/reference codec 在 CPU、CUDA 上 packed data 和 scale byte 全部一致；旧 block-16 recipe 无回归。
- `--kv-cache-dtype mxfp4` 能在明确受限的 FlashInfer + `--disable-cuda-graph` 配置下完成 prefill、prefix reuse 和 decode；storage 为 MXFP4，attention 输入为 BF16，checkpoint FP8 scales 未误用。
- FlashInfer PLAIN 与 Torch MXFP4 对同一 packed cache 满足冻结后的紧阈值；真实 Qwen 回放逐步通过服务级硬上限。
- 所有限制在参数校验或启动阶段可见失败；L2 不承诺性能。验收后才能进入 L3 Triton inline-dequant decode kernel，并移除 PLAIN 的整层反量化与临时内存约束。