# FP8 KV Torch Decode Golden 开发计划

## 范围与实现选择

- 文档中的 L0 仍是现有生产路径：FlashInfer prefill + FlashInfer FP8 decode；本计划交付的是复现 L0 数值语义的 L1 Torch decode golden。
- 采用用户确认的“测试参考实现 + 离线回放”方案：不增强 `TorchNativeAttnBackend`，不增加公开 backend 参数，不要求完整服务切换到 Torch decode。
- 不实现 Torch prefill 或任何新 prefill attention kernel。测试中的历史 KV 由直接填充或现有 FlashInfer prefill 产生。
- Golden 只支持本阶段需要的 decoder self-attention：单 token/query、paged radix KV、GQA、`qk_head_dim == v_head_dim`、无 sliding window、无 logit cap；其他模式显式报错，避免无意扩展范围。

## Pure-Torch FP8 Decode Reference

### 新增测试公共实现

在 `python/sglang/test/kits/attention_unittest/attention_methods/fp8_decode_attention.py` 新增可被后续 MXFP4 复用的测试态组件：

- `fp8_cache_quantize_reference(x, scale)`：精确复现当前 `MHATokenToKVPool.set_kv_buffer`，先在源 dtype 上 `clone().div_(scale)`，再 cast 到 `torch.float8_e4m3fn`；不能擅自改为 FP32 divide。
- `fp8_cache_dequantize_reference(cache, scale)`：用于观测有效 K/V，但不作为 FlashInfer 运算顺序的唯一假设。
- `torch_fp8_radix_decode_reference(...)`：
  1. 用 `req_to_token[req_pool_idx, :seq_len]` gather 逻辑 KV，绝不假设物理 slot 连续；
  2. 将 KV heads 按 GQA ratio 扩展到 Q heads；
  3. 按 FlashInfer 的 BMM scale 位置计算：
     ```python
     scores = einsum(q.float(), k_fp8.float()) * (layer.scaling * k_scale)
     probs = softmax(scores, dim=-1, dtype=torch.float32)
     out = einsum(probs, v_fp8.float()) * v_scale
     ```
  4. 输出转回 Q/model dtype，并可选返回 cache bytes、dequant K/V、scores、probs、output 等诊断中间量。
- 同时保留“先反量化 K/V 再 attention”的语义参考，仅用于诊断有限精度运算顺序差异；正式 golden 采用上述 BMM-boundary scale 顺序。

## Attention Test Fixture 扩展

对 `python/sglang/test/kits/attention_unittest/attention_methods/dense_attention.py` 做向后兼容的最小扩展：

- `MockModelRunner` / `build_dense_attention_fixture` 增加可选参数：`kv_cache_dtype`、`kv_cache_dtype_str`、`k_scale`、`v_scale`；默认值保持现有所有测试行为不变。
- FP8 用例创建 `MHATokenToKVPool(dtype=torch.float8_e4m3fn)`，并设置 `RadixAttention.k_scale/v_scale` 及对应 `*_scale_float`。
- `_populate_prefix_kv` 在 FP8 用例中将 distinct K/V scales 传给 pool，使历史 token 与生产写入语义一致。
- Decode 当前 token 仍由真实 `FlashInferAttnBackend.forward_decode` 写入 cache，再由同一份最终 FP8 cache 同时供 FlashInfer 和 Torch reference 对比，避免两套写入造成混淆。
- 不改现有 `_dense_attention_reference`；FP8 reference 独立存在，防止影响 BF16/FP16 backend 回归测试。

## 自动化测试

在 `test/registered/attention/unittests/dense/test_flashinfer_fp8_decode.py` 增加 CUDA + FlashInfer 条件测试。

### 1. FP8 cache 写入契约

- 使用真实 `MHATokenToKVPool.set_kv_buffer`，验证目标 slot 的 FP8 `uint8` bit pattern 与 Torch reference 一致。
- 使用明显不同的 `k_scale`、`v_scale`，捕获漏传、互换或重复应用 scale。
- 覆盖全零、随机 BF16、接近 E4M3 最大有限值、随机非连续 loc、未写 slot 保持不变。
- 单独验证 decode 当前 token 也经过相同 QDQ 后进入 attention。

### 2. 单层 FlashInfer FP8 decode 差分

固定随机种子，以相同 Q、最终 FP8 cache、page table 和 scales 比较 FlashInfer output 与 Torch golden：

- 小型定位矩阵：MHA `4/4`、GQA `4/2`、MQA `4/1`，head_dim=64。
- Qwen 主配置：Q heads=24、KV heads=4、head_dim=256、BF16 Q、FP8 E4M3 KV。
- Batch/长度：batch 1 与多 batch；短序列；`15/16/17`、`127/128/129` 等 page 边界。
- Page/layout：page_size 1、16、128（仅保留当前 FlashInfer 支持组合）；contiguous、shuffled、interleaved/random physical slots。
- Scale：`1.0/1.0` 与接近 checkpoint 的 distinct `0.0275/0.0245`。
- 逐级断言：cache bytes → gather 后 KV → output；失败时输出 max abs、mean abs、relative L2、cosine、norm ratio 及最差 head/request。

### 3. 容差建立方法

- Cache 写入必须 bit-exact，不使用浮点容差。
- Torch helper 的两份独立数学实现先以严格 FP32 容差互验。
- FlashInfer 差分先在本机 SM120 上对完整矩阵跑 20 个固定 seeds，记录各指标分布；提交测试时冻结观测最差值的 1.25 倍为阈值。
- 设置不可放宽的上限：relative L2 ≤ 2e-2、cosine ≥ 0.999、norm ratio ∈ [0.98, 1.02]。超过上限必须检查 scale 放置、QDQ dtype、GQA 或 page mapping，不能通过放宽 `DENSE_ATOL/DENSE_RTOL` 掩盖。
- CI 常规回归保留 3 个固定 seeds；20-seed characterization 作为本地/手动慢测。

## 真实 Qwen Decode 离线回放

新增 `compare_fp8_decode.py`，复用 `locate_fa_bug.py` 的 checkpoint 权重读取、QK norm/RoPE 重建方法及现有 debug tensor dump：

- 读取同一请求的 prefill pass 与后续 decode pass；从 `model.layers.<id>.qkv_proj`、positions/seq_lens 重建 RoPE 后 Q/K/V。
- 按真实逐层 `k_scale/v_scale` 将历史和当前 K/V 写成 FP8 cache；每个 decode step 调用公共 Torch reference。
- 与 dump 中 gate 前的 `model.layers.<id>.attn`（FlashInfer 输出）逐层、逐 token 比较。
- 默认验证第一个 full-attention layer 3，并支持传入全部 full-attention layer ids；报告 cache/QDQ、attention output 的 cosine、relative L2、norm ratio。
- 该脚本只消费离线 dump，不修改服务端执行路径；`compare_hf_sglang.py` 继续负责 HF/logprob 上限基线，不混入本次 kernel-equivalence 判定。

## 验收与交付边界

- 自动测试证明 FP8 cache 写入 bit-exact，且 Qwen shape、GQA、page boundary 和非连续 radix loc 下 Torch decode 满足冻结后的误差阈值。
- 离线真实 Qwen 回放中，各 full-attention 层无固定倍率偏差、无 NaN/Inf，逐 token 指标稳定且满足同一误差上限。
- 测试失败可由中间量定位到 write/QDQ、page gather、GQA、QK scale、softmax 或 V scale，而非只有最终 output 差异。
- 不改 `flashinfer_backend.py`、`torch_native_backend.py` 或服务参数；不新增 prefill attention；不做性能承诺。
- 验收后把实测阈值、硬件/依赖组合和回放结果补入 `qwen38_attention_survey_notes.md`，再以该 Torch FP8 decode 核心为基础替换 MXFP4 codec。