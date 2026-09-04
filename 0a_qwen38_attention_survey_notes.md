# Qwen3.8-27B 基线摸底笔记（为 MXFP4 KV cache 开发准备）

日期：2026-08-30　硬件：RTX 5090 32GB（SM120）　代码库：sglang 0.5.16（/sgl-workspace/sglang）

## 1. 环境与依赖（最终可用组合）

- torch 2.13.0+cu130（仓库 pyproject pin；2.11 与 sglang-kernel wheel 的 torch ABI 不兼容）
- flashinfer-python 0.6.18 + flashinfer-jit-cache 0.6.18+cu130（索引 https://flashinfer.ai/whl/cu130/）
  - 注意：旧版包名是 flashinfer-cubin，0.6.17+ 改为 flashinfer-jit-cache；两者版本必须一致
- sglang-kernel 0.4.6.post1（SM120 走其 sm100 变体，"precise math for compatibility"）
- transformers 5.12.1（仓库 pin；升到 5.16 会与仓库自带 qwen3_asr AutoConfig 注册冲突）

## 2. 基线启动命令与结果

```
python -m sglang.launch_server \
  --model-path /sgl-workspace/models/Qwen3.8-27B-NVFP4 \
  --kv-cache-dtype fp8_e4m3 --mem-fraction-static 0.85 \
  --context-length 32768 --port 30000

# experimental torch mxfp4 kvcache（L2：PLAIN 读，验证路径）
/venv/main/bin/python -m sglang.launch_server \
  --model-path /sgl-workspace/models/Qwen3.8-27B-NVFP4 \
  --kv-cache-dtype mxfp4 --attention-backend flashinfer \
  --disable-cuda-graph --mem-fraction-static 0.80 \
  --context-length 32768 --port 30001

# native mxfp4 kvcache（L3：triton native decode + CUDA graph，生产组合）
/venv/main/bin/python -m sglang.launch_server \
  --model-path /sgl-workspace/models/Qwen3.8-27B-NVFP4 \
  --kv-cache-dtype mxfp4 --prefill-attention-backend flashinfer \
  --decode-attention-backend triton --mem-fraction-static 0.80 \
  --context-length 32768 --port 30001

```

- 权重：21.48 GB（compressed-tensors 混合量化：0-55 层 MLP = NVFP4 W4A4，56-63 层 MLP/attn 投影/lm_head = FP8 W8A8）
- kv_cache_dtype=fp8_e4m3 已生效；该 checkpoint 的 kv_cache_scheme 使 auto 也会解析为 FP8，现显式指定以固定基线。
- max_total_num_tokens=82479（FP8 KV，context 32768 时）；bf16 KV 精度上限基线为 41239 tokens。
- FP8 KV logprobs 对比 PASS（top-1 差 0.101），贪心续写与 HF 完全一致。

### 停止服务

```
# 前台运行：直接 Ctrl+C（SIGINT，会优雅停掉 scheduler/detokenizer 子进程）

# 后台运行：先对主进程 TERM，未退出再 KILL
pgrep -f "sglang.launch_server"        # 主进程 PID；另有多个 sglang:: 子进程
kill <PID> && sleep 5 && kill -9 <PID> 2>/dev/null

# 一键清掉全部 sglang 相关进程（含 scheduler 子进程，前后各打一次 nvidia-smi）
bash scripts/killall_sglang.sh
# bash scripts/killall_sglang.sh all   # 连其它 GPU 进程一起清，会误伤其他任务，慎用
```

验证已停干净：

```
pgrep -f sglang || echo clean                                   # 应输出 clean
curl -s --max-time 2 http://127.0.0.1:30000/health || echo "server down"
nvidia-smi                                                      # 显存应释放（权重 ~21.5GB + KV cache）
```

- 注意：只 kill -9 主进程会留下 `sglang::scheduler_TP*` 等子进程继续占显存，
  用 killall_sglang.sh 按进程名模式全清更稳妥。
- 若 nvidia-smi 仍显示显存占用，说明有残留 CUDA context：
  `bash scripts/killall_sglang.sh all` 或 `fuser -k /dev/nvidia*`。

## 3. 解析出的后端

| 项 | 值 | 来源 |
|---|---|---|
| MHA attention_backend | flashinfer | SM120 默认（get_default_attn_backend：trtllm_mha 不支持 SM120，回退 flashinfer） |
| prefill / decode | 均为 flashinfer（未拆分） | server_args dump |
| linear_attn_backend | triton | 混合 GDN 模型 SM120 允许 triton/trtllm_mha/flashinfer |
| 包装结构 | HybridLinearAttnBackend | attention_registry.py（hybrid_gdn_config 分支） |

## 4. 调用链(静态快照;完整逐层讲解见 §8,行号以 §8 实测为准)

### 全注意力层（16 层：layer_types 中每 4 层一个）
```
Qwen3_5Attention.self_attention            models/qwen3_5.py:1328
  → forward_prepare_cuda_fused             (fused qk-norm + gate 提取 + partial RoPE 0.25)
  → self.attn(q,k,v) = RadixAttention      layers/radix_attention.py:157
  → get_attn_backend().forward → FlashInferAttnBackend.forward_extend/forward_decode
                                            layers/attention/flashinfer_backend.py:1289/1450
  → fused_sigmoid_mul(attn_out, gate)      (swish 输出门控，attention 之后)
  → o_proj
```
- 注意：gate 与 q 一起从 qkv_proj 出来（q_size*2），gate 不进 KV cache；KV 仍是纯 K/V，
  head_dim=256、4 KV heads——mxfp4 KV 量化作用于此。
- KV 写入：flashinfer wrapper 在 attention kernel 内写 paged KV（save_kv_cache）；
  量化 KV 的插入点是 MHATokenToKVPool.set_kv_buffer / quant_method.quantize_and_store
  （mem_cache/memory_pool.py；混合架构经 kv_cache_configurator._build_hybrid_linear_kv_pool
  把 quant_method 传入 HybridLinearKVPool → 内层 MHA pool）。

### 线性注意力层（48 层，Gated DeltaNet）
```
Qwen3_5GatedDeltaNet.forward               models/qwen3_5.py:166 起
  → RadixLinearAttention.forward           layers/radix_linear_attention.py:80
  → GDNAttnBackend (linear/gdn_backend.py) ← HybridLinearAttnBackend 分发
```
与 KV cache 量化无关（循环状态按请求固定大小，不按 token 增长）。

## 5. 实际 kernel 名（profiler 实测，/start_profile 抓 32 token 生成）

全注意力（flashinfer）：
- `flashinfer::BatchPrefillWithPagedKVCacheKernel`（prefill 与 decode 都用；两种 KernelTraits：
  MaskMode 0 = extend 分段、MaskMode 1 = 常规/decode）
- `flashinfer::PersistentVariableLengthMergeStatesKernel`（chunked prefix 状态合并）
- `create_flashinfer_kv_indices_triton`（page table 索引构建）
- KV 写入由 paged kernel 内联完成（未见独立 copy kernel）

线性注意力（triton/自研）：
- prefill：`chunk_gated_delta_rule_fwd_kernel_h_blockdim64`、
  `chunk_gated_delta_rule_fwd_kkt_solve_kernel`、`chunk_fwd_kernel_o`、
  `chunk_local_cumsum_scalar_kernel`、`fused_gdn_gating_kernel`、`_causal_conv1d_fwd_kernel`
- decode：`fused_recurrent_gated_delta_rule_packed_decode_kernel`、
  `_fused_qkvzba_causal_conv1d_update_contiguous_kernel`
- 状态跟踪：`track_mamba_states_all_layers_kernel`

MLP 路径可见 `tensorrt_llm::quantize_with_block_size`（W4A4 激活在线量化），与 KV 无关。

## 6. 对后续 MXFP4 KV cache 开发的要点(08-30 快照;最终路线图见 §11)

1. 量化插入点：`fp4_kv_cache_quant_method.py` 新增 `MXFP4KVCacheMethod`（block-32、UE8M0），
   注册进 `KV_CACHE_QUANT_REGISTRY`、解除 `resolve_kv_cache_quant` 中 "mxfp4" 的保留报错、
   `server_args.kv_cache_dtype` choices 加入 "mxfp4"。
2. 当前 decode 后端是 flashinfer（SM120 默认），而现有两个 FP4 recipe 的 decode 访问规则
   （NATIVE_FP4→trtllm_mha、PLAIN→triton/torch_native/flex/trtllm_mha）都不含 flashinfer；
   新 recipe 需要声明 flashinfer decode 的访问方式（最可能是 PLAIN：读取时反量化为
   bf16/fp8 供 flashinfer），或基线改用支持的 decode 后端。这是开发时第一个要决策的点。
3. 混合架构（HybridLinearKVPool）已能把 quant_method 传给 16 个全注意力层的 MHA pool，
   无需额外改动该层。
4. 每 token KV 规模：16 层 × 2 × 4 heads × 256 dim = 32,768 元素
   （bf16 64KB；mxfp4 预计 16KB 数据 + 1KB scale = 17KB/token）。

## 7. 数值问题诊断（2026-08-30）：FA 层 attention 输出异常

现象：贪心解码下渐进崩坏（首 token 对 → 逐 token 漂移 → 乱码/循环），
RAW/CHAT 皆然，输出逐字可复现（确定性数值错误）。

定位（详见 compare_hf_sglang.py，含 HF golden 对照）：
- 已排除：template/采样/radix cache/CUDA graph/overlap/后端选择
  （flashinfer 与 triton 皆崩）、GDN kernel、NVFP4 W4A4（单层手算
  rel_err=9.6% 正常）、FP8 W8A8（2.6% 正常）、lm_head（#34895 的
  scale 丢失在本 HEAD 已修复）、checkpoint 本身（数据自洽）。
- 首个发散点：layers.3（第一个 full-attention 层）attention 分支
  cos=0.397；qkv_proj 投影输出与 HF 逐位一致（cos=0.99998）。
- 核心异常：RadixAttention 输出 norm=2.23 vs HF 87.3（**小 39 倍**），
  下游全部层级被污染。两引擎 RoPE 频率/配置一致，gate 值一致。
- 换后端无法解决：两个 FA 后端共享 q/k 准备路径与模型实现；
  SM120 无第三 FA 后端（trtllm_mha 不支持）。
- 关联：issue sgl-project/sglang#34895（同 checkpoint 同症状），
  fix PR #34904 open；vLLM 同 checkpoint 正常。

工具：`python3 compare_hf_sglang.py --collect-hf / --logprobs / --compare`。
注：修复前 mxfp4 KV cache 开发无法做精度对比基线，需先等 FA 路径修复
或换 vLLM 做参考实现。

### 根因与修复（已定位并修复）

根因：**框架代码 bug**（非模型/kernel 算子）。checkpoint 的
kv_cache_scheme 自带 FP8 KV descale（k_scale=0.0275, v_scale=0.0245），
`--kv-cache-dtype bf16` 时 K/V 从未按 scale 量化存储，但 backend 仍把
`layer.v_scale_float` 无条件作为 descale 传给 kernel → attention 输出
被乘 0.0245（反解出的 P 行和恰为 0.0245）→ 输出缩小约 41 倍。
读写不对称：写入路径 `_kv_write_scales` 有 `needs_global_scale` 豁免，
读取侧没有。

修复（本仓库，未上游）：
- `flashinfer_backend.py`：新增 `self.kv_cache_scales_valid =
  flashinfer_kv_cache_dtype ∈ {fp8e4m3, fp8e5m2}`；3 处读取点的
  k_scale/v_scale、`_kv_write_scales` 写入点均按此条件化。
- `triton_backend.py`：同条件属性 + 4 处 descale 判断 + 写入
  `_set_kv_buffer` 传参条件化。
- FP8 KV 模式行为不变（scale 合法）；修复后 HF 对比：top-1 logprob 差
  0.963 → 0.107，贪心续写与 HF 完全一致，长生成正常。
- 其它 backend（fa3/flashmla 等）存在同样模式，上游 PR 时应一并处理。

定位脚本：`locate_fa_bug.py`（fused norm kernel → RoPE → attention kernel
逐步二分，A/B 步 rel≈0.002 清白，故障在 descale 因子）。

### 补充验证：FP8 KV 是官方正常路径（2026-08-30）

`--kv-cache-dtype fp8_e4m3` 实测：KV 容量翻倍（82479 tokens），
logprobs 对比 PASS（top-1 差 0.101），贪心续写与 HF 完全一致。
结论：FP8 KV 下读写对称、scale 应用合法——**checkpoint 的官方意图
就是 FP8 KV，该路径从未坏过**。当时的 bug 只在“显式 bf16 + checkpoint
自带 KV scales”的组合下触发（写不缩放/读缩放不对称）；官方默认
auto→FP8 掩盖了它。修复后两种模式均自洽:bf16 用作精度上限基线,
FP8 用作官方部署基线。

---

## 8. Radix Attention 全路径调用详解(2026-08-31 代码走读,行号当日实测)

### 8.1 鸟瞰

```
模型层                    调度层                      后端层                      存储层
qwen3_5.self_attention → RadixAttention.forward → HybridLinearAttnBackend(分发)
                              │                        ├─ FlashInferAttnBackend(全注意力 16 层)
                              │                        └─ GDNAttnBackend(线性 48 层,与 KV 量化无关)
                              │                              │
                              │                    forward_extend / forward_decode
                              │                              │
                              │                    ┌─────────┴──────────┐
                              │                    ▼ 写                  ▼ 读
                              │        pool.set_kv_buffer        pool.get_kv_buffer
                              │                    │                        │
                              ▼                    ▼                        ▼
                     fused_sigmoid_mul      MHATokenToKVPool        dequant(PLAIN / workspace)
                       → o_proj             .quantize_and_store       → flashinfer paged kernel
```

### 8.2 模型层(models/qwen3_5.py)

- 入口 `self_attention`(1331-1374)。CUDA 平台走 `forward_prepare_cuda_fused`
  (1227-1255):一个融合 kernel 完成 QK GemmaRMSNorm + 部分 RoPE(0.25)+
  gate 解交织(`fused_qk_gemma_rmsnorm_rope_gate`)。
- **gate 不进 KV cache**:`attn_output_gate` 时 `qkv.split([q_size*2, kv_size, kv_size])`,
  gate 与 q 交错,但只有纯 K/V 进 `self.attn(q,k,v)`;attention 之后
  `fused_sigmoid_mul(attn_out, gate)` → o_proj。
- KV 形状 (num_tokens, 4, 256)(4 KV heads × head_dim 256)= mxfp4 量化对象。
- `self.attn` 即 RadixAttention 实例;其 `quant_method` 是**权重**量化方法
  (quant_config.get_quant_method),与 KV cache 量化无关,勿混淆;
  `k_scale/v_scale` = checkpoint 自带 KV descale(§7 修复 bug 所在字段)。

### 8.3 RadixAttention 薄分发层(layers/radix_attention.py)

`forward`(157-298)不含 attention 数学:
1. k/v reshape 为 (tokens, kv_heads, head_dim);
2. piecewise CUDA graph 上下文 → `unified_attention_with_output` custom op
   (426-463):预分配 output buffer、按真实 token 数切片、layer_id 经
   context.attention_layers 查表;MLA 双实例(attn_mqa/attn_mha)经
   mha_companion_layers 区分(351-357);
3. eager 分支直接 `get_attn_backend().forward(...)`(289-298)。
- 输出 dtype 规则(212-219):跟随 v.dtype;fp8 q/v 仍产 bf16 输出。

### 8.4 后端分发

attention_registry.py:hybrid_gdn_config 分支把后端包成 HybridLinearAttnBackend,
按 layer 分发:全注意力 → FlashInferAttnBackend,线性 → GDNAttnBackend。
模型代码无感知,统一接口 `forward(q, k, v, layer, forward_batch, save_kv_cache)`。

### 8.5 FlashInfer 后端(flashinfer_backend.py)

`forward_extend`(1299-1457)与 `forward_decode`(1460-1516)对称,三步:
1. **写**:`set_kv_buffer(layer, KVWriteLoc(...), k, v, *self._kv_write_scales(layer))`。
   FP4 场景 scales 为 None(NVFP4 全局 scale 由 quant_method 自管,needs_global_scale)。
2. **取**:`kv_cache = pool.get_kv_buffer(layer_id)`,或 dequant workspace 路径
   (1327-1339 / 1489-1501,见 8.7)。
3. **算**:wrapper.forward(q, kv_cache, ...) → flashinfer paged kernel。
   **kernel 只读 KV**;新 token 写入全部发生在第 1 步(§5 "paged kernel 内联写"
   的观察实为 set_kv_buffer 的 scatter kernel 不显眼,注意区分)。
- `k_scale/v_scale` 只在 `kv_cache_scales_valid`(KV dtype ∈ {fp8e4m3, fp8e5m2},
  368-371)时传 kernel——§7 修复点。**MXFP4 时必须保持 None**,否则复现
  写不缩放/读缩放不对称事故。
- `flashinfer_kv_cache_dtype` 解析(355-363):workspace 路径强制 fp8e4m3,
  注释 "FP4 fake-quant ... exposes an FP8 workspace to FlashInfer"
  → 证明 flashinfer kernel 只认 fp16/bf16/fp8(§10 原因 1)。

### 8.6 KV 写路径(memory_pool.py)

`MHATokenToKVPool.set_kv_buffer`(2375-2451)分发:
```
is_quantized_kv_cache → _set_quantized_kv_buffer(2512) → quant_method.quantize_and_store(2525)
否则 → div_(scale) → .to(dtype) → view(store_dtype) → scatter 写入(或 HND/5D 变体)
```
- `is_quantized_kv_cache` = quant_method 不是 UnquantizedKVCacheMethod(1975-1977)。
- `_quantized_scales`(2502-2510):NVFP4 型方法(k_scales_gpu 存在)在调用方
  未传 scale 时自动补全局 scale。
- graph 安全:capture mode 走 alt_stream K/V 重叠分支;quantize_and_store
  内部必须无 host sync(kvfp4_tensor 用 @torch.compile 纯 tensor 保证)。

### 8.7 KV 读路径:三种 access kind

- **PLAIN**:needs_plain_kv_dequant_read() 时 `_get_key_buffer/_get_value_buffer`
  (2320-2358)对**整层 buffer** `dequantize_kv_tensor` → bf16 再交 backend。
  每层全量 dequant,性能差但零新 kernel,天然支持所有 backend(含 flashinfer)。
- **DEQUANT_WORKSPACE**(NVFP4 prefill):`get_raw_kv_buffer`(2545)取
  packed+scale;`_prepare_dequant_extend_workspace`(2662-2719)/
  `_prepare_dequant_decode_workspace`(2721-2749)按 request prefix 索引批量
  反量化到共享 FP8 workspace(dq_k/dq_v,层间复用)再喂 flashinfer。
  Python for 循环:prefill(低频)撑得住;decode + CUDA graph 撑不住。
- **NATIVE_FP4**:kernel 直读 packed FP4+scale。trtllm_mha decode 用
  get_raw_kv_buffer + TRTLLM-gen fp4 decode kernel(trtllm_mha_backend.py:1168、1275)。

### 8.8 启动期构建链

1. server_args.py:592-618:--kv-cache-dtype choices("mxfp4" 当前不在列表)。
2. kv_cache_dtype.py `configure_kv_cache_dtype`(67-79):
   "nvfp4"/"fp4_mx_block16" → torch.float4_e2m1fn_x2(需 torch 2.8+/CUDA 12.8+)。
3. kv_cache_configurator.py `_build_fp4_quant_method`(281-293):
   is_float4_e2m1fn_x2(kv_cache_dtype) → resolve_kv_cache_quant(str→recipe 名)
   → get_kv_cache_quant_method 实例化 → load_scales_from_model。
4. hybrid 路径 `_build_hybrid_linear_kv_pool`(1565-1618)把 quant_method 传
   HybridLinearKVPool(memory_pool.py:3613-3704)→ 内层 MHA pool 以
   quant_method kwarg 构造。**该层已确认无需为 mxfp4 改动**。
   ("mxfp8" 分支演示了另一条路:full_pool_class 直接换成专用 pool 类。)
5. 层 id:HybridLinearKVPool.full_attention_layer_id_mapping 把全局 layer id
   映射到 16 层 dense pool 的局部 id。

## 9. KV cache 量化三层架构(fp4_kv_cache_quant_method.py)

文件头(14-34)设计声明:**quant_method(纯计算)► Pool(buffer+批量 dequant)► Backend(view 适配)**。

- `KVCacheQuantMethodBase`(110):create_buffers / quantize_and_store /
  dequantize_prev_kv / dequantize_kv_tensor / compute_cell_size /
  needs_global_scale + access 解析(resolve_attention_access)。
- `KVCacheAttentionAccess`(79-107):phase × kind × backend_matcher(exact/tags/any)
  × storage_dtype × attention_kv_dtype × scale_recipe × workspace_dtype。
  一个 recipe 用一组 access 声明"哪个 phase、哪些 backend、以什么方式消费 KV"。
- 三种 kind(53-60):PLAIN / DEQUANT_WORKSPACE / NATIVE_FP4。
- 注册表:KV_CACHE_ATTENTION_ACCESS_REGISTRY(777-790)、
  KV_CACHE_QUANT_REGISTRY(794-797)、resolve_kv_cache_quant(800-827;
  819-824 对 "mxfp4" 报"保留"错误 = 解除点)。

### 现有两个 recipe 对比

| | NVFP4KVCacheMethod(333) | FP4MXBlock16KVCacheMethod(559) |
|---|---|---|
| scale | 两级:全局 FP32 + block-16 FP8 E4M3 | 单级 block-16,uint8 存 exp+127 |
| block | 16 | 16(明确注明"不是真 MXFP4") |
| prefill | DEQUANT_WORKSPACE:flashinfer → FP8 | PLAIN:triton/torch_native/flex/fa4 → BF16 |
| decode | NATIVE_FP4:trtllm_mha 直读 packed | PLAIN:triton/torch_native/flex/trtllm_mha → BF16 |

- flashinfer 不在任何 FP4 decode 列表(原因见 §10)。
- NVFP4 的 SM100/SM120 scale bridge(412-421):XQA 期望 amax/448,
  checkpoint 存 amax/(6*448),SM100 乘 E2M1_MAX 修补。
- 量化数学 = FP4MXBlock16KVQuantizeUtil(kvfp4_tensor.py:58-149):
  @torch.compile 纯 tensor、CUDA graph 安全(无 .item()/host sync);
  E2M1 用 BOUNDS (0.25,0.75,1.25,1.75,2.5,3.5,5.0) 计数法(37-55);
  双 fp4 打包 uint8(低 4 位 = 偶数元素);scale = ceil(log2(amax/6)),
  存 uint8(exp+127)——**位语义上就是 UE8M0**,只是 dtype 标注为 uint8;
  mxfp4 可直接用 torch.float8_e8m0fnu(MXFP8 先例,memory_pool.py:3294-3345)。

## 10. NVFP4 prefill/decode 不对称的根因(2026-08-31)

三层原因,均为硬约束而非偏好:

1. **FlashInfer 没有 FP4 KV 输入 kernel**(只认 fp16/bf16/fp8):
   flashinfer_backend.py:355-363 注释 "FP4 fake-quant ... exposes an FP8
   workspace to FlashInfer" → FP4 想用 flashinfer 只能先反量化到 FP8 workspace。
2. **TRTLLM native FP4 kernel 只有 decode 变体**:
   trtllm_mha_backend.py:1275-1279 `forward_extend` 直接 raise:
   "TRTLLM MHA with native FP4 KV cache supports decode only; use a separate
   prefill backend such as flashinfer or triton." → 两条路径 = 各自能力交集。
3. **prefill compute-bound vs decode memory-bound**:
   - prefill 计算量大,一次反量化进 workspace 可摊销(NVFP4 上游自己也这么选)。
   - decode 每步读全部 KV:workspace 方案 = 读 FP4 + 写 workspace + kernel 读
     workspace 三倍读写,量化带宽收益清零 → 必须 kernel 内 inline dequant(native)。
   - 附加:decode 走 CUDA graph,workspace 构建的 Python for 循环 capture 不友好。

结论:prefill/decode 不对称是**原则**。access 框架用 (phase, backend, kind)
声明,就是让 recipe 按"哪条路通走哪条"组合。

## 11. MXFP4 开发路线图（2026-08-31 修订：先建立 FP8 Torch golden）

### 11.1 核心原则：建立分层、可归因的参考链

之前“直接实现 MXFP4 Torch golden”的顺序会同时引入两个变量：
1. FlashInfer kernel → Torch attention 的实现差异；
2. FP8 KV → MXFP4 KV 的数值格式差异。

一旦输出不一致，无法判断误差来自 attention 数学、Radix/page-table 语义、
FP8 scale 处理，还是 MXFP4 编解码。因此改为四层参考链：

```
L0：现有生产基线
    FlashInfer decode attention + FP8 E4M3 KV（checkpoint k/v scale）
      ↓ 先只替换 decode attention 实现，KV 数值语义不变
L1：FP8 Torch decode-attention golden
    Torch decode attention + 与 L0 完全相同的 FP8 写入/QDQ/scale/page-table 语义
      ↓ 再只替换 KV codec，Torch decode attention 数学不变
L2：MXFP4 Torch decode-attention golden
    Torch decode attention + block-32 E2M1 + UE8M0 scale
      ↓ 以 L2 为数值 oracle
L3：高性能 decode 实现
    Triton / FlashInfer 扩展 / sgl-kernel CUDA（不开发新的 prefill attention kernel）
```

对比关系必须固定：**L1 对 L0 验证 Torch decode attention 等效性；L2 对 L1 测量
纯 MXFP4 量化损失；L3 对 L2 验证 decode kernel 正确性。** BF16 继续作为不含 KV
量化误差的精度上限，但不再是第一层直接对照。

### 11.2 L0 的 FP8 KV 准确数值语义（Torch golden 必须复现）

当前基线明确使用 `--kv-cache-dtype fp8_e4m3`，解析为
`torch.float8_e4m3fn`。Qwen3.8 checkpoint 还提供逐层、per-tensor 的
`layer.k_scale` / `layer.v_scale`（本次实测约 0.0275 / 0.0245，各层以实际值为准）。

**写入语义**（flashinfer_backend.py `_kv_write_scales` →
MHATokenToKVPool.set_kv_buffer）：

```python
K_fp8 = (K_bf16 / k_scale).to(torch.float8_e4m3fn)
V_fp8 = (V_bf16 / v_scale).to(torch.float8_e4m3fn)
```

**读取/attention 语义**（FlashInfer wrapper 接收相同 k_scale/v_scale）：

```python
K_effective = K_fp8.to(compute_dtype) * k_scale
V_effective = V_fp8.to(compute_dtype) * v_scale
scores = (Q @ K_effective.transpose(-1, -2)) * layer.scaling
output = softmax(scores) @ V_effective
```

这表示 FP8 golden 不能只做 `K.to(fp8).to(bf16)`；必须完整复现
**除 global scale → E4M3FN 舍入/范围处理 → 乘回 global scale**。同时：

- Q 不进 KV cache，保持模型计算 dtype；只有 K/V 做 FP8 QDQ。
- `kv_cache_scales_valid=True` 仅对实际 FP8 KV 成立；MXFP4 阶段必须为 False，
  不能再叠加 checkpoint 的 FP8 descale（§7 的历史 bug）。
- decode：当前 token 先写入 FP8 cache，再与历史 token 一起从 cache 读取，
  因此当前 token 的 K/V 也经过 FP8 QDQ。
- **prefill 不在自研 attention 实现范围内**：继续使用现有 FlashInfer prefill，
  只负责计算 prompt attention 并把 K/V 写入 cache。以下分支只作为理解“decode
  cache 是怎样产生的”的背景，不需要为其编写 Torch/MXFP4 prefill attention：
  - paged 分支：先写 FP8 cache，再由 paged kernel 读取，当前 chunk 也经过 QDQ；
  - ragged、无 prefix：attention 直接使用当前 bf16 K/V，计算后才写 FP8 cache；
  - ragged、有 prefix：当前 chunk 用 bf16 K/V，cached prefix 用 FP8 QDQ，
    FlashInfer 分别计算并 merge state。
- 将来 prefix reuse/chunked prefill 若需要读取 MXFP4 cache，只实现反量化到
  bf16/fp8 的适配并复用现有 FlashInfer prefill，不开发新的 prefill attention kernel。

### 11.3 L1：先实现 FP8 Torch decode-attention golden

目标不是新生产 backend，也不是重写 prefill，而是得到一个简单、透明、可逐项
检查的 decode reference。测试可直接构造/填充 KV cache，或由现有 FlashInfer
prefill 产生 cache；Torch 只替换 decode attention。它应显式完成：

1. 根据 `req_to_token[req_pool_idx, :seq_len]` 从物理 KV pool gather 每个请求的
   逻辑 token 序列，不能假设 cache loc 连续或按请求排列。
2. 按 §11.2 执行 FP8 E4M3 QDQ；支持真实逐层 k_scale/v_scale，也测试 scale=1。
3. GQA：将 4 个 KV heads 按组映射/`repeat_interleave` 到 Q heads。
4. decode：每个请求一个 query，使用包含当前 token 在内的完整 `seq_len` KV；
   覆盖不同 batch、长度、page 边界和物理 cache 布局。
5. attention reference 优先显式执行 fp32 score、softmax、V reduction，最后转回
   模型输出 dtype；可另用 `torch.nn.functional.scaled_dot_product_attention`
   做第二份交叉验证，但不依赖其内部 kernel 作为唯一 oracle。
6. 保留中间量用于分层比较：cache FP8 字节、dequant K/V、score、softmax、
   attention output，避免只看最终 token/logprob。

**不能直接把当前 TorchNativeAttnBackend 当作 FP8 golden。** 它是很好的
Radix/page-table/GQA 框架参考，但当前 `set_kv_buffer` 调用没有传
`layer.k_scale/v_scale`，读取时也只是把 FP8 K/V cast 到 query dtype，没有乘回
scale。因此在带 checkpoint KV scales 的 Qwen3.8 上，它与 FlashInfer FP8 的
数值语义不等价。实现时应二选一：

- 首选：在 attention unittest 框架内新增独立 `fp8_radix_attention_reference`，
  显式注入 codec，不改生产 backend；
- 若后续确有调试价值，再给 TorchNativeAttnBackend 增加受控的 FP8 scale-aware
  路径，但必须避免改变其他模型/无 scale 场景的既有行为。

可复用代码：
- `torch_native_backend.py`：decode 请求逐条 gather、GQA 和 page-table 语义；
- `python/sglang/test/kits/attention_unittest/attention_methods/dense_attention.py`
  的 `_dense_attention_reference`：显式 fp32 einsum + softmax reference；
- `test/registered/attention/unittests/dense/test_torch_native.py`：decode page boundary、
  GQA/MQA、non-monotonic/interleaved cache loc 测试矩阵。

### 11.4 L1 测试与验收顺序

**A. FP8 codec 单测**
- 全零、正负极值、subnormal/饱和边界、随机 bf16；
- 不同 k_scale/v_scale；验证写入 FP8 bit pattern、反量化值及无 NaN/Inf；
- 与真实 MHATokenToKVPool 写入后的 cache 内容逐元素/逐 bit 对比。

**B. 单层 decode attention 差分测试**
- 同一组 q/k/v、相同 layer.scaling、相同 req_to_token/out_cache_loc；
- L1 Torch FP8 decode vs L0 FlashInfer FP8 decode；覆盖：
  - batch=1/多 batch、短序列、长序列、page 边界；
  - Qwen 配置：GQA、head_dim=256、4 KV heads；
  - contiguous、shuffled、non-monotonic physical loc；
  - cache 由直接 codec 填充和现有 FlashInfer prefill 产生两种来源。
- 先比较 cache bytes/dequant K/V，再比较 attention output；容差由实测误差分布确定，
  不预先用过宽阈值掩盖 scale/layout 错误。

**C. 模型端到端对齐**
- prefill 始终使用现有 FlashInfer，decode 在 FlashInfer FP8 与 Torch FP8 间切换；
- 固定 prompt、greedy、seed、禁用会改变执行图的非必要功能；
- 比较每步 top-k token/logprob、首个发散 token、逐层 decode attention output；
- 目标是 Torch FP8 decode 与现有 FlashInfer FP8 decode 对齐；HF/BF16 只作上限参考。

只有 A/B/C 均通过，才能认为 decode attention 与 Radix/page-table/FP8 scale 语义
已经被正确复制，进入 MXFP4 阶段。

#### L1 验收记录

已完成（2026-08-31），合入 commit `b960f9e207`（"L0: fp8 torch radix attention done"）。
交付物清单、A/B/C 测试结果、冻结阈值与关键发现的完整记录已拆分并重构至
[qwen38_mxfp4_kv_acceptance_records.md](qwen38_mxfp4_kv_acceptance_records.md) §1。

结论：A/B/C 全部通过——FlashInfer FP8 decode 与 Torch FP8 decode 数值等价，可进入 L2。

### 11.5 L2：在同一 Torch decode attention 上替换为 MXFP4 codec

此阶段保持 L1 的 decode gather、GQA、scaling、softmax 和输出转换完全不变，
只把 FP8 codec 替换为 MXFP4 codec，从而把差异严格归因到 KV 数值格式：

- 新建 `MXFP4KVQuantizeUtil`：block-32、E2M1 data、每 block 一个 UE8M0 scale；
- 先实现透明 Torch 版本：quantize → pack uint8 → unpack → dequantize；
- 规范决策点：OCP MX shared exponent/scale 计算、全零块、overflow/underflow、
  tie/rounding、NaN/Inf、block 轴及不足 32 元素的 padding，必须先写测试锁定；
- 对 Qwen K/V shape `(tokens, 4, 256)`，应优先让每个 head 的 head_dim 独立按
  连续 32 元素分块，避免 scale block 跨 head；每个 head 8 个 scale；
- 比较顺序：MXFP4 codec 单测 → L2 MXFP4 Torch 对 L1 FP8 Torch 的 K/V、
  score、softmax、output 误差 → 固定样本逐层/逐 token logprob → 长生成稳定性；
- 此阶段不追求吞吐，不实现 Torch prefill attention，不引入 Triton/CUDA，
  也不让 backend 差异污染量化评估。

内存估算不变：16 层 × K/V × 4 heads × 256 = 32768 FP4 元素/token，
数据 16KB；block-32 共 1024 个 1B scale，即 1KB；合计约 17KB/token。

### 11.6 L2 接入框架的低性能 PLAIN 验证路径

Torch 数值验证通过后，再把 MXFP4 接入现有框架：

1. 新增 `MXFP4KVCacheMethod(name="mxfp4", SCALE_BLOCK_SIZE=32)`，注册
   KV_CACHE_QUANT_REGISTRY / KV_CACHE_ATTENTION_ACCESS_REGISTRY；
2. 解除 resolve_kv_cache_quant 的 "mxfp4" 保留报错；server_args choices 和
   configure_kv_cache_dtype 增加 mxfp4；
3. 先声明 PLAIN access，pool 读时 dequant 为 bf16，让现有 FlashInfer prefill
   和 decode 跑通端到端；这是功能集成验证，不是新 prefill kernel；
4. `compute_cell_size` 纳入 packed data、scale 及任何临时 workspace；
5. 混合架构继续走 quant_method 注入 HybridLinearKVPool 内层 MHA pool，
   不复制 `MHATokenToKVPoolFP4` 的专用 pool 路径；slot move 必须同时移动 data/scale。

### 11.7 L3：最后开发高性能 kernel（不在 golden 前锁死实现途径）

RTX 5090/SM120 上 trtllm_mha 不可用，候选路径为：

- **Triton（首选原型）**：复用 triton backend 的 paged metadata，最快验证
  inline MXFP4 unpack/dequant + attention；便于直接与 L2 Torch golden 比较。
- **FlashInfer 扩展**：若最终希望保持默认 backend，需要在 FlashInfer 增加原生
  MXFP4 paged attention/JIT kernel 与相应 wrapper；集成一致但要改外部项目。
- **sgl-kernel CUDA/CUTLASS**：适合追求最终性能上限，开发与维护成本最高；
  应在 Torch/Triton 数值路径稳定后再做。

优先级：
1. **decode native kernel 优先**：decode memory-bound，只有 kernel 内 inline
   dequant 才能兑现 FP8 32KB/token → MXFP4 17KB/token（BF16 为 64KB/token；
   MXFP4 相对当前 FP8 约节省 47% KV 空间，理论容量约 1.88 倍，实际还受
   allocator、workspace 和其他显存占用影响）。
2. **量化写入 kernel**：融合 block amax/scale、E2M1 pack、按 loc scatter 与
   scale 写入；参考 `quant_store_kv_mxfp8`。
3. **不开发 prefill attention kernel**：prefill 始终复用现有 FlashInfer；
   MXFP4 prefix cache 读取只做 PLAIN 或 dequant-workspace 适配。

L3 验收：kernel 中间解码值对 L2 codec；单层 output 对 L2；模型 logprob/长生成
对 L2；最后才比较吞吐、TTFT、TPOT、显存和不同 context/batch/page_size。

### 11.8 开发序列（修订后）

```
固定 L0：现有 FlashInfer prefill + FlashInfer fp8_e4m3 decode
  → L1：保持现有 prefill，只实现 FP8 Torch decode golden 与差分测试
  → L2：同一 Torch decode attention 替换 MXFP4 codec + 数值测试
  → MXFP4 PLAIN/dequant 适配现有 prefill + 端到端验证
  → L3：Triton 原型 / FlashInfer 扩展 / sgl-kernel CUDA
       （只开发 decode attention kernel；另做量化写入 kernel）
```

最终形成三重 oracle：**FlashInfer FP8 decode 证明 Torch decode attention 正确；
Torch FP8 隔离现有量化基线；Torch MXFP4 证明新 codec；高性能 decode kernel
只需对 Torch MXFP4 golden。prefill 全程复用现有实现。**

### 11.9 L2 验收记录

已完成（2026-08-31），合入 commit `2f22cc9d79`（"mxfp4 kv cache with torch kernel done"）。
交付物、测试结果、冻结阈值与关键发现（inductor 误编译、容量口径、ragged prefill
语义、fail-fast）的完整记录见
[qwen38_mxfp4_kv_acceptance_records.md](qwen38_mxfp4_kv_acceptance_records.md) §2。

结论：`--kv-cache-dtype mxfp4` 在受限组合（flashinfer + MHA + disable-cuda-graph）
下端到端可用，等价性达标；进入 L3 前的 oracle 链（OCP codec → Torch MXFP4
decode → FlashInfer PLAIN）已闭环，可进入 L3。

### 11.10 L3 验收记录

已完成（2026-09-04），kernel 基础合入 commit `f0a95c27c1`（"triton native mxfp4
decoding attention kernel"），位构造/grouped/splits32 优化同系列后续提交。
完整记录（交付物、双层差分冻结阈值、融合写 kernel 契约、CUDA graph、性能
三配置对比与优化轨迹、SM120 dot_scaled 结论、关键发现 G1-G7、复跑指南）见
[qwen38_mxfp4_kv_acceptance_records.md](qwen38_mxfp4_kv_acceptance_records.md) §3。

结论：`--kv-cache-dtype mxfp4 --prefill-attention-backend flashinfer
--decode-attention-backend triton` + CUDA graph 端到端可用；native decode
kernel 差分达标，融合写 kernel bit-exact；TPOT 与 FP8 基线差距 ~9%，容量
1.52×。生产组合与后端配对决策（triton decode 只做 native、scale 布局保持
flat、后端配对 fail-fast）另见 qwen38_mxfp4_layout.md（含 §6 性能调研）。
精度验收（HumanEval/AIME）另行执行后补录。
