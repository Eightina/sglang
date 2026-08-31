# Qwen3.8-27B-NVFP4 数值崩坏问题：从模型架构到根因修复的完整分析

> 硬件：RTX 5090（SM120, 32GB）｜软件：SGLang 0.5.16-dev + flashinfer 0.6.18 + torch 2.13.0+cu130
> 模型：`unsloth/Qwen3.8-27B-NVFP4`（compressed-tensors 混合量化）
> 结论速览：**框架代码 bug**（attention backend 的 KV descale 读写不对称），
> 非 NVFP4 算子精度问题、非 checkpoint 问题。已在本仓库修复并验证。

---

## 1. 问题现象

启动服务后（`--kv-cache-dtype bf16`，贪心解码 temperature=0）：

- RAW 与 CHAT 两种模式输出**都**胡言乱语 → 排除 chat template
- 同一 prompt 多次请求输出**逐字相同** → 确定性数值错误，非采样/状态串扰
- **渐进式崩坏**：前 10 余个 token 正确（" Paris.\nThe capital..."），随后漂移、
  退化为无意义算式（"3 + 2 = 3"、"-8 - 1"）和重复循环

探针样例（修复前）：

```
Q: The capital of France is
A:  Paris.
   The capital city of France is Paris.
   No, that's not the question.
   A single question with a possible kersue to it. Zero a   <- 开始崩坏
```

---

## 2. 模型架构：Qwen3.8-27B（Qwen3_5 架构）

`Qwen3_5ForConditionalGeneration` = visual tower（本场景不用）+ language model。
语言模型共 64 层，**混合注意力**架构（`layer_types` 每 4 层一循环）：

### 2.1 Full-attention 层（16 层：layer 3, 7, 11, ..., 63）

- GQA：24 个 Q 头 / 4 个 KV 头（repeat 系数 6），head_dim = **256**
- **attn_output_gate**：q_proj 输出 12288 维 = 24 头 × [q(256) | gate(256)]，
  attention 输出在 o_proj 前逐元素乘 `sigmoid(gate)`（值域实验：sigmoid(gate)
  均值仅 ≈0.016，即 gate 是强抑制性的门控）
- **q_norm / k_norm**：Gemma 风格 RMSNorm（`x * rsqrt(mean(x²)+eps) * (1+w)`），
  逐头作用
- **partial RoPE**：`rope_parameters = {mrope_interleaved: true,
  mrope_section: [11,11,10], partial_rotary_factor: 0.25, rope_theta: 1e7}`。
  旋转维度 = 256×0.25 = 64（32 个频率对）。纯文本时 T=H=W 三轴同位置，
  interleaved 重排退化为恒等 → 数学上等价普通 neox RoPE（已被逐频率验证）
- 数据流：`qkv_proj (W8A8) → split → q_norm/k_norm → RoPE → attention
  (flashinfer/triton) → ×sigmoid(gate) → o_proj (W8A8)`

### 2.2 Gated DeltaNet（GDN）线性注意力层（48 层）

- 16 QK 头 / 48 V 头，head_dim 128；`in_proj_qkvz`（W8A8）+ `in_proj_ba`（bf16，
  checkpoint 里被 ignore）+ causal conv1d + gated delta rule 递推 + `out_proj`（W8A8）
- **无 RoPE**——这一点后来成为定位关键（GDN 层全对、FA 层全错）
- SSM 状态与 conv state 存于独立的 mamba cache（`mamba_radix_cache_strategy`）

### 2.3 层间分布（为什么"层 3"特殊）

```
layer:  0(GDN) 1(GDN) 2(GDN) 3(FA) 4(GDN) ... 63(FA)
                          ↑ 第一个 full-attention 层
```

诊断时发现层 0-2 完全正常、层 3 起崩坏——GDN 无 RoPE/无 gate 乘法、
FA 层才有，这个分布特征直接把嫌疑域缩到了 FA 路径。

---

## 3. Checkpoint 量化方案（compressed-tensors 混合量化）

`config.json → quantization_config` 两个 config group + ignore 清单：

### 3.1 group_0：FP8 W8A8

- targets：`re:.*self_attn\.(q|k|v|o)_proj$`、
  `re:.*linear_attn\.(in_proj_qkv|in_proj_z|out_proj)$`、
  `re:.*lm_head`、`re:.*layers\.(56..63)\.mlp\.(gate|up|down)_proj$`
- weights：FP8 E4M3 存储 + **per-channel** scale（BF16，[N,1]）
- input_activations：**dynamic per-token** FP8（gemm 入口瞬时量化）
- 每层还带 `k_scale`/`v_scale`（per-tensor 标量）——注意这**不属于 W8A8**，
  见 §4

### 3.2 group_1：NVFP4 W4A4

- targets：`re:.*mlp\.(gate|up|down)_proj$`（0-55 层生效，56-63 被 group_0
  的更具体 regex 先行匹配）
- weights：FP4 E2M1 + **per-block-16** FP8 E4M3 scale + 全局 scale
  （`weight_global_scale=6400`，编码关系 `W_real = fp4 × s_fp8 / gs`，
  已数值验证：反量化 std=1.02e-2，与同结构 MTP 层 bf16 权重 std=1.26e-2 同量级）
- input_activations：dynamic local FP4（block 16，gemm 入口瞬时量化）
- format：`nvfp4-pack-quantized`（linear 布局，非 trtllm shuffle 版）

### 3.3 ignore（不量化，bf16 存储）

visual tower 全部、GDN 的 `in_proj_b`/`in_proj_a`（96 维小投影）、conv1d、
`re:^mtp.*`（MTP 层整段 bf16——后来成为权重分布的天然对照参考）。

### 3.4 kv_cache_scheme：static per-tensor FP8

```json
每层附带: k_scale = 0.0275 (标量),  v_scale = 0.0245 (标量)
```

声明"**KV cache 存储建议用 FP8**，配这两个 scale"。这是**存储域**的声明，
与上面两个**计算域**的量化 group 完全独立（详见 §4）。
`--kv-cache-dtype auto` 会据此解析为 FP8 KV；本文的问题在显式 `bf16` 时触发。

---

## 4. 核心概念：两个独立的量化域

### 4.1 量化域一：linear 层的计算格式（W8A8 / W4A4）

以 qkv_proj（W8A8）为例的完整数据流：

```
hidden_states (bf16)
  → 逐 token 动态量化 amax/448 → FP8 激活        ← "A8"：瞬时存在
  → FP8 权重 × FP8 激活（FP32 累加）
  → 反量化（× per-channel weight_scale）
  → 输出 Q/K/V = bf16                             ← 回到高精度
```

**为什么输出必须回到 bf16（不能"直出 FP8"）**——这是理解本 bug 的关键：

1. **后续算子在数学上需要高精度**。K/V 写入 cache 前还要经过：
   - `k_norm`（Gemma RMSNorm）：需要 `rsqrt(mean(x²))`——FP8 的 2-bit 指数
     动态范围无法表示方差倒数，且 norm 会改变值域分布
   - `RoPE`：需要 sin/cos 三角函数精度做 64 维旋转
   - gate 的 `sigmoid` 同理
   这些算子若强行在 FP8 域做，误差会逐层爆炸；所有主流实现（HF/vLLM/
   sglang）都在 bf16 域完成 norm+RoPE，之后才谈存储量化。
2. **residual 流是 bf16**。attention 分支输出要与残差流相加，进下一层的
   layernorm 和 MLP——整条主干是 bf16。
3. **A8/W8 是"计算时"的瞬时格式**：激活量化只在 gemm 入口存在，算完即丢；
   它描述的是"这次乘法用什么精度"，不产生持久的 FP8 张量。

W4A4（NVFP4 MLP）同理：激活的 FP4 量化同样只发生在 gemm 入口
（per-block-16 + 全局 scale），输出同样回到 bf16。

### 4.2 量化域二：KV cache 的存储格式（kv_cache_scheme）

```
K/V (bf16, norm+rope 后)
  → 写 cache：÷ v_scale → FP8 E4M3 存储        （若启用 FP8 KV）
  → 读 cache：× v_scale → 还原 bf16 值域       （与写入必须对称）
```

- `v_scale = 0.0245`：校准关系 `scale = amax / 448`，即校准 amax ≈ 11.0。
  本次 prompt 实测 V absmax = 4.84——校准值留有余量（覆盖更长上下文的
  极端值），说明这是**离线统计的存储 scale**，与 gemm 无关
- 两域的 scale 体系完全独立：

| | 量化域一：计算（W8A8/W4A4） | 量化域二：存储（KV cache） |
|---|---|---|
| 作用对象 | gemm 的权重与输入激活 | 写入 KV cache 的 K/V |
| scale 形状 | per-channel [N,1] / per-block-16 [N,K/16] | per-tensor 标量 |
| 生命周期 | gemm 入口瞬时，算完即反量化 | 持久存在于显存 |
| 是否启用 | 由 checkpoint group 决定（固定） | 由 `--kv-cache-dtype` 决定（可选） |
| 语义 | "这次乘法用什么精度" | "KV 用几个 bit 存储" |

### 4.3 结论

K/V 的**原生形态是 bf16 高精度张量**（模型语义），FP8 KV cache 是可选的
**存储压缩**。两者可以自由组合：
- FP8 计算 + bf16 KV 存储 = KV 精度更高（显存翻倍）
- FP8 计算 + FP8 KV 存储 = 官方推荐（省一半 KV 显存）
- 反过来"qkv 直出 FP8 KV"不成立，因为 norm/RoPE 卡在中间

---

## 5. 排查过程

### 5.1 单变量隔离实验（全部无效，排除运行时因素）

| 实验 | 结果 |
|---|---|
| RAW /generate（不套 template） | 仍崩 → 排除 chat template |
| temperature=0 复跑 | 逐字相同 → 确定性数值错误 |
| `--disable-radix-cache` | 仍崩（输出逐字不变）→ 排除 prefix 复用 |
| `--disable-cuda-graph` | 仍崩 → 排除 graph replay |
| `--disable-overlap-schedule` | 仍崩 → 排除调度重叠 |
| `--attention-backend triton` | 仍崩 → 排除 flashinfer 单方 |
| `--linear-attn-decode-backend cutedsl` | 仍崩 → 排除 GDN triton decode |
| `--fp4-gemm-backend flashinfer_cudnn` | 输出与 cutlass **bit 级相同** → 排除 gemm 后端 |

### 5.2 算子级数值验证（全部正确）

- NVFP4 W4A4 单层（复现 sglang 完整调用链 vs bf16 精确参考）：
  rel_err = 9.6%，cos = 0.9954 —— 双重量化理论范围（8-12%）
- FP8 W8A8 单层（per-token + per-channel，`_scaled_mm`）：rel_err = 2.6%
- lm_head：top-10 logprobs 幅度正常（-1.65），scheme 分配正确
  （issue #34895 描述的 lm_head scale 丢失在本 HEAD 已修复）
- checkpoint 自洽：NVFP4 反量化 `W = fp4 × s / gs`（gs=6400）后
  std=1.02e-2，与 MTP 层 bf16 权重（同结构参考）std=1.26e-2 同量级
- RoPE：两引擎 inv_freq 逐值一致（theta=1e7、32 对、partial 64 维）

### 5.3 Golden 对照与逐层 dump（定位发散点）

用 HF transformers（CPU，compressed-tensors 自动解压路径）跑同一 checkpoint
作为 golden 参考；sglang 以 `--debug-tensor-dump-output-folder` dump 逐
module 输出，逐层对照 hidden-states 残差：

```
layer  type   cos     rel_err
 0     GDN    0.99996  0.0098     <- 正常
 1     GDN    0.99912  0.0420     <- 正常
 2     GDN    0.99917  0.0407     <- 正常
 3     FA     0.39662  0.9217     <- 第一个 full-attention 层，首个发散点
 4-15  ...    0.39~0.93           <- 下游全部被污染
```

层 3 内部下钻：

- `qkv_proj` 投影输出与 HF **逐位一致**（q+gate/k/v 三段 cos≥0.998）——
  投影正确，问题在投影之后
- `RadixAttention` 输出 norm = **2.23** vs HF 的 attention 核心 norm = **87.3**
  （HF 手算链闭合验证 cos=0.989）——**attention 输出小约 41 倍**

### 5.4 逐步二分（locate_fa_bug.py）

用 sglang 真实组件离线复现层 3 前向，每步对照手算参考：

```
A: fused_qk_gemma_rmsnorm_with_gate (triton kernel)   rel = 0.0017  ✓ 清白
B: MRotaryEmbedding (partial neox RoPE)               rel = 0.0019  ✓ 清白
C: attention kernel 输出                              ×0.025        ✗ 命中
```

### 5.5 反解法实锤

attention 输出是 V 的凸组合 `attn_out = P·V`。V 已知（5×5 可逆），
从 sglang 的 attn 输出反解出它实际使用的权重矩阵 P：

- **P 的行和恒等于 0.0245**（softmax 行和必为 1）→ 输出被整体乘了
  `v_scale = 0.0245`
- P 的相对形状仍与正确 softmax 权重相关（cos 0.54~0.84）

至此机制完全清楚：**attention 输出 = 正确结果 × v_scale**。

---

## 6. 根因：FP8 KV descale 的读写不对称

### 6.1 机制链

```
checkpoint kv_cache_scheme (FP8 KV)
  → compressed_tensors 给每个 RadixAttention 创建 k_scale/v_scale 参数
     (CompressedTensorsKVCacheMethod.create_weights, 初始 -1.0)
  → load_weights: checkpoint 的 ".v_scale" (0.0245) 加载进参数
     (qwen3_5.py 的映射 ".self_attn.v_scale" → ".attn.v_scale")
  → process_weights_after_loading: layer.v_scale_float = 0.0245
     (quantization/kv_cache.py，无条件)
  → backend 把 descale 无条件传给 kernel：
     flashinfer_backend.py ×3 处  v_scale=layer.v_scale_float
     triton_backend.py   ×4 处  v_descale = layer.v_scale_float
  → kernel 输出 = P·(V × v_scale)   ← bf16 KV 下 V 从未被 ÷scale 存储！
```

### 6.2 读写不对称

- **写入路径** `_kv_write_scales()` 有豁免条件
  （`needs_global_scale()` → 返回 None,None），bf16 pool 实际存原始值
- **读取路径** 无任何条件，descale 直接进 kernel
- 净效应：attn 输出 = 正确值 × 0.0245

### 6.3 为什么 FP8 KV 模式没事、官方没发现

- FP8 KV 时写入 ÷scale 量化、读取 ×scale 还原，**对称** → 数值正确
- `--kv-cache-dtype auto` 默认解析为 FP8 KV（checkpoint 声明了
  kv_cache_scheme），所有人走这条路 → bug 被"官方路径"掩盖
- 只有"显式 bf16 + checkpoint 自带 KV scales"的组合才触发

---

## 7. 修复

原则：**descale 是否合法，取决于 KV cache 的实际存储 dtype**，
而不是 checkpoint 是否附带 scale。

### 7.1 flashinfer_backend.py

```python
# __init__：
self.kv_cache_scales_valid = self.flashinfer_kv_cache_dtype in (
    torch.float8_e4m3fn, torch.float8_e5m2)

# 3 处读取点（forward_extend / swa / forward_decode）：
k_scale=layer.k_scale_float if self.kv_cache_scales_valid else None,
v_scale=layer.v_scale_float if self.kv_cache_scales_valid else None,

# 写入点 _kv_write_scales()：同样加
if not self.kv_cache_scales_valid:
    return None, None
```

### 7.2 triton_backend.py

同条件属性 `self.kv_cache_scales_valid`（TritonAttnBackend.__init__）；
extend/decode/verify 等 4 处 descale 判断加该条件；
写入 `_set_kv_buffer` 的 scale 传参条件化。

### 7.3 验证

| 指标 | 修复前 | 修复后 |
|---|---|---|
| top-1 logprob vs HF | -1.653（差 0.963） | **-0.584（差 0.107）** |
| top-10 集合 | 排序错乱 | 与 HF 一致（FP4 噪声级差异） |
| 贪心续写 | 第 3 token 分歧 | **与 HF 完全一致** |
| RAW "1+1等于几？" | 崩坏 | **"2"** |
| CHAT 自我介绍 | 语无伦次 | 连贯正确，finish=stop |
| FP8 KV 模式 | 正常 | 正常（行为不变，回归安全） |

FP8 KV 附加收益：KV 容量翻倍（41239 → 82479 tokens）。

### 7.4 遗留

其它 backend（fa3 / flashmla 等）存在相同的"无条件使用 v_scale_float"模式，
上游 PR 时应一并按 `kv_cache_dtype` 条件化。

---

## 8. 经验教训

1. **"第一个 token 正确"是弱证据**：高频先验 token（Paris）可以掩盖 prefill
   的分布偏移；崩坏模式（渐进 vs 突发）比单点正确性更有信息量。
2. **尽早建立独立引擎 golden 参考**（HF transformers / vLLM），
   配合 sglang 的 `--debug-tensor-dump-output-folder` 做逐层 dump 对照，
   比纯运行时开关穷举快得多。
3. **反解 P·V 权重矩阵**是诊断 attention 数值问题的锋利手段：
   softmax 行和 ≠ 1 直接暴露缩放因子（本例 0.0245 = v_scale）。
4. **量化域要分清**：linear 的 W8A8/W4A4 是计算格式（输出回 bf16），
   KV cache FP8 是存储格式（读写必须对称）；两者的 scale 体系独立，
   部署选项（--kv-cache-dtype）只应影响后者。

---

## 附录：相关文件与工具

- `compare_hf_sglang.py`：--collect-hf / --logprobs / --compare 三模式
  对比工具（golden 缓存 `hf_golden_cache.pt`）
- `locate_fa_bug.py`：层 3 前向逐步二分定位
- `ask_model.py` / `chat_tui.py`：最小请求探针 / 交互 TUI
- 本仓库修复：`python/sglang/srt/layers/attention/flashinfer_backend.py`、
  `python/sglang/srt/layers/attention/triton_backend.py`
- 上游关联：issue sgl-project/sglang#34895（同 checkpoint 同症状，
  作者归因 lm_head scale 丢失不完整——其 dispatch 修复已在本地 HEAD，
  症状依旧）、fix PR #34904（open）；vLLM 同 checkpoint 正常
- 环境备忘：transformers 需 5.12.1（qwen3_asr 冲突）、torch 2.13.0+cu130
  （sgl-kernel ABI）、accelerate（HF CPU 加载）
