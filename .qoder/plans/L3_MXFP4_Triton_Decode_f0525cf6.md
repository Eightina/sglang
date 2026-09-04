# L3 开发计划：MXFP4 Native Triton Decode Kernel

## 已锁定决策（来自前期讨论，不再变更）

- 宿主：扩展现有 triton backend，不新建 backend；
- scale 布局：保持 flat `(slots, head_num, 8)` uint8，不做 MXFP8 式 interleave；
- 新 triton kernel 只做 native mxfp4 decode（直读 packed + E8M0 scale，inline dequant），不消费 PLAIN、不做 extend 变体；
- 生产组合：prefill/extend = flashinfer（PLAIN→BF16，L2 已验收），decode = triton（native）；fail-fast 强制该配对；
- decode access 保留两条（native→triton 新增，plain→flashinfer 保留），用于 L3 服务级 A/B 验收；
- 开发序列：dequant 微内核 → decode kernel → 集成 → 融合写 kernel。

## Phase 1：独立 kernel + 数值验证（零生产代码改动）

新文件 `python/sglang/kernels/ops/attention/mxfp4_decode_attention.py`：

1. **dequant 微内核**：输入 packed uint8 `(slots, H, head_dim//2)` + scale `(slots, H, head_dim//32)`，输出 bf16/fp32。nibble 解包（低 4 位 = 偶数索引，与生产 codec 一致），scale 用 `tl.exp2(s.to(tl.float32) - 127)`。
2. **native decode attention kernel**：两段式 split-KV，镜像 [decode_attention.py](file:///sgl-workspace/sglang/python/sglang/kernels/ops/attention/decode_attention.py) 的 `_decode_att_m_fwd` + `_decode_softmax_reducev_fwd` 结构（复用 `_extract_kv_strides`，data 与 scale 各取一组 stride）；每 BLOCK_N tile 内 load 128B packed/token → 解包 → 乘 E8M0 scale → fp32 QK/softmax/PV；支持 PAGE_SIZE 地址数学；不实现 lean/score_mod/DCP/SWA 分支。

新测试 `test/registered/attention/unittests/dense/test_triton_mxfp4_native_decode.py`：

- **A 步**：dequant 微内核 vs [mxfp4_decode_attention.py](file:///sgl-workspace/sglang/python/sglang/test/kits/attention_unittest/attention_methods/mxfp4_decode_attention.py) 的 OCP oracle（`mxfp4_quantize_reference`），逐元素精确（全 E2M1 code、scale 指数边界、全零块、partial block）；
- **B 步**：decode kernel vs L2 Torch golden（共享核心 `torch_radix_decode_from_effective_kv` + OCP codec 编码的 cache），矩阵沿用 L1/L2：MHA/GQA/MQA × head_dim 64/128/256 × page size 1/16/32/64 × 4 种物理 loc 布局；
- **表征与冻结**：env 开关（如 `SGLANG_MXFP4_TRITON_CHARACTERIZE=1`）跑 20-seed，按 worst × 1.25 冻结阈值写入测试常量（预期略宽于 PLAIN 的 3.1e-3，因 dequant 时机与累加顺序不同）。

## Phase 2：框架集成（decode 路径切换）

1. [fp4_kv_cache_quant_method.py](file:///sgl-workspace/sglang/python/sglang/srt/layers/quantization/fp4_kv_cache_quant_method.py#L930-L933)：MXFP4 access 条目追加 `_native_fp4(_DECODE, {"triton"}, _MXFP4_SCALE, _TORCH_FP4)`，保留现有两条 PLAIN。
2. pool 侧：确认/补齐 mxfp4 的 raw 读取接口——packed buffer 走既有 `k_buffer/v_buffer`（uint8），scale 暴露类似 `MHATokenToKVPoolMXFP8.get_kv_scale_buffer` 的按层访问器（若通用 quantized pool 尚无则新增）。
3. [triton_backend.py forward_decode](file:///sgl-workspace/sglang/python/sglang/srt/layers/attention/triton_backend.py#L2062-L2089)：检测 pool 为 mxfp4 quantized 时分发到新 kernel（传 packed buffer + scale buffer）；`kv_cache_scales_valid` 已天然为 False；对 mxfp4 拒绝 lean attention / score_mod / DCP / sliding window / MLA 分支（fail-fast）。
4. [kv_cache_hook.py L48-L65](file:///sgl-workspace/sglang/python/sglang/srt/arg_groups/kv_cache_hook.py#L48-L65)：mxfp4 允许 `(flashinfer, flashinfer)` 与 `(flashinfer, triton)` 两种配对；拒绝整组 triton 与其他 decode 后端；`--disable-cuda-graph` 要求本阶段保留。
5. 端到端验证：
   - `--prefill-attention-backend flashinfer --decode-attention-backend triton --kv-cache-dtype mxfp4` 启动，dump 回放 `compare_mxfp4_decode.py` 对 OCP oracle 达标（服务级硬上限 rel_l2 ≤ 2e-2 / cos ≥ 0.999 / norm_ratio ∈ [0.98, 1.02]）；
   - 同服务 A/B：decode 后端切回 flashinfer（PLAIN 路径）对比 triton native，差异应与 B 步差分同量级；
   - 回归：L1/L2 全量套件（246 passed 基线）不回退。

## Phase 3：融合写 kernel + CUDA Graph 解禁

1. 新文件 `python/sglang/kernels/ops/quantization/mxfp4_quant.py`（参考 `quant_store_kv_mxfp8` 的封装形态）：单 triton kernel 完成 block-32 amax → E8M0 scale → E2M1 saturating RNE pack → 按 loc scatter data + scale；无 host sync，CUDA graph 安全；保留 `reserved_skip_index=0`（slot 0 保留）契约。
2. [MXFP4KVCacheMethod.quantize_and_store](file:///sgl-workspace/sglang/python/sglang/srt/layers/quantization/fp4_kv_cache_quant_method.py#L757) 切换到融合 kernel，替换 eager `MXFP4KVQuantizeUtil` 路径。
3. 写契约测试（沿用 L1 A 步模式）：生产写入的 packed/scale 字节 vs OCP oracle bit-exact，覆盖 A1-A4 四项语义（slot 0 保留、in-place 快照、scale 形态、uint8 字节对比）；eager codec 保留为 oracle/回退参考。
4. hook 解除 `--disable-cuda-graph` 限制；验证 capture/replay 下 decode 输出确定性（replay 前后字节一致）。

## Phase 4：性能与精度基准、验收归档

三种服务配置同机（RTX 5090）对比：**A** = FP8 生产基线（`--kv-cache-dtype fp8_e4m3`）；**B** = mxfp4 + flashinfer PLAIN（`(flashinfer, flashinfer)`）；**C** = mxfp4 + triton native（`(flashinfer, triton)`，Phase 3 后含 cuda graph）。

1. 性能对比（同 prompt 集）：
   - C vs B（验证 inline dequant 收益，两者消费同一 cache 语义）；
   - C/B vs A（TPOT / 吞吐 / max_total_num_tokens，预期容量约 1.88×）；
   - 扫 context length / batch size / page size。
2. 精度对比（数据集：AIME + HumanEval）：
   - 入口复用仓库现有评测 kit：HumanEval 用 [eval_accuracy_kit.py](file:///sgl-workspace/sglang/python/sglang/test/kits/eval_accuracy_kit.py) 的 `HumanEvalMixin`（对活服务走 OpenAI API）；AIME 用 sgl_eval 驱动（`_run_sgl_eval`，仓内 NPU 精度测试已用 `aime25`/`aime26` 任务同模式）；若 sgl_eval 未安装则按其提示 pip 安装；
   - **硬门禁（C vs B）**：同一 cache 字节、仅 decode kernel 不同，两配置分数差应在采样噪声内（固定 greedy/固定 seed 跑 HumanEval pass@1；AIME 用固定 temperature+seed、n_repeats 取均值）；超出即视为 kernel 数值回归，回查 Phase 1 冻结阈值；
   - **表征记录（B/C vs A）**：MXFP4 相对 FP8 的分数降幅只记录不设通过阈值——验收记录 §2.6 已表明 block-32 E2M1 有真实精度代价（单层 rel_l2 0.41~0.75、首 token 即分叉），该数据作为 L3 归档的质量背景与启动日志 accuracy-drop 警告的实证；
   - 阈值取值：先跑 A 建立本 checkpoint 的基线分（不套用其它模型的 CI 阈值，如 test_eval_accuracy_large.py 的 0.64），B/C 的判定相对 A 的实测基线给出。
3. 验收归档：[qwen38_mxfp4_kv_acceptance_records.md](file:///sgl-workspace/sglang/qwen38_mxfp4_kv_acceptance_records.md) 追加 §3 "L3 验收"（交付物、A/B 测试、冻结阈值、性能与精度数据、commit 索引），并更新 §0 索引表与笔记 §11 路线图状态。

## 测试计划（命令）

```bash
# Phase 1
python -m pytest test/registered/attention/unittests/dense/test_triton_mxfp4_native_decode.py -q
SGLANG_MXFP4_TRITON_CHARACTERIZE=1 python -m pytest test/registered/attention/unittests/dense/test_triton_mxfp4_native_decode.py -q
# Phase 2/3 回归
python -m pytest test/registered/attention/unittests/dense/ test/registered/unit/layers/quantization/ test/registered/unit/server_args/test_kv_cache_hook_mxfp4.py -q
# 端到端回放
python3 compare_mxfp4_decode.py --dump-dir /tmp/sgl_mxfp4_triton --layers 3 --max-steps 8
# Phase 4 精度（对活服务，A/B/C 三配置各起一个 server）
python -m pytest test/manual/eval/test_eval_accuracy_large.py -q   # 参考形态；实际用自建脚本/mixin 跑 HumanEval + AIME
```

## 范围外（明确不做）

- prefill/extend attention kernel（全程复用 flashinfer）；triton 消费 PLAIN；
- MLA、speculative verify、HND layout、interleaved scale；
- FlashInfer 扩展与 sgl-kernel CUDA 变体（路线图备选，本期不启动）。

## 假设

- 目标模型 head_dim=256 可被 32 整除（每 head 8 个 scale 块，无 partial block 生产路径）；
- MHA-only、CUDA/SM120；GDN 48 层与 KV 量化无关不受影响。