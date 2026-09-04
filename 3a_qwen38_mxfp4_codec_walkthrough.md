# MXFP4 KV codec 逐段讲解：`MXFP4KVQuantizeUtil`

> 代码：`python/sglang/srt/layers/quantization/kvfp4_tensor.py` L152-295
> （`batched_quantize` L166-238、`batched_dequantize` L240-295）。
> 本文讲"这段代码每一步在做什么、为什么"。验收背景见
> [qwen38_mxfp4_kv_acceptance_records.md](qwen38_mxfp4_kv_acceptance_records.md) §2。

## 1. 数学规格（30 秒版）

MXFP4 把每 **32 个连续元素**（一个 block）压成：

- **32 个 E2M1 码**（4-bit：1 符号 + 2 指数 + 1 尾数），打包后 16 字节；
- **1 个 E8M0 scale**（1 字节，存 `exp + 127` 的原始 byte，读侧 `view(torch.float8_e8m0fnu)`）；
- block 沿 head_dim 切、不跨 head：KV 形状 `(tokens, 4 heads, 256)`，每 head 恰好
  `256/32 = 8` 块，无 partial block。

E2M1 码表（`E2M1_VALUES` L37-54，正半轴 8 个值）：

| code | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|---|
| 值 | 0.0 | 0.5 | 1.0 | 1.5 | 2.0 | 3.0 | 4.0 | 6.0 |

（code bit3 = 符号；偶 code 尾数位 = 0，奇 code 尾数位 = 1——RNE tie 用的就是它。）

Encoding Rules:

Normal Numbers
```
value = (-1)sign × 2(exponent - 1) × (1 + mantissa / 2)
```
With 1 mantissa bit, there are only 2 values per power-of-2 interval: 1.0 and 1.5 (times the power of 2). The bias of 1 is calculated as 2(e-1) - 1, where e is the number of exponent bits (2).

Subnormal Numbers
```
value = (-1)sign × 20 × (mantissa / 2)
```

**OCP MX v1.0 §6.3 的 scale 公式**（docstring L155-157）：

```
scale = 2^floor(log2(amax)) / 4        # 4 = E2M1 可表示的最大 2 的幂
⇒ scaled_amax = amax / scale ∈ [4, 8)  # > 6 的部分饱和到最大码 6
```

**每 token 存储量**：每 head `256/2 = 128 B` data + `8 B` scale → 每 token 每 tensor
`4 × 136 = 544 B`，K+V 两份 × 16 层 = **17 KiB/token**（FP8 为 32 KiB）。

## 2. `batched_quantize`：六步流水线

### ① 校验 + fp32 提升（L174-186）

```python
if tensor.ndim != 3: raise ...          # 强制 [tokens, heads, dim]
values = tensor.to(torch.float32)       # bf16 → fp32，无损
```

bf16 的值集是 fp32 的子集，升精度零信息损失。此后除法、log2、距离比较全部在
fp32 域进行——**整条链唯一的舍入点是规范定义的那次 RNE**（见 §5）。奇数 dim 时
pad 到 block 边界（Qwen 的 256 整除，pad 分支实际不触发，但 oracle 对比要求
CPU/CUDA 行为一致，保留）。

### ② 分块（L182-187）

```python
num_blocks = (n + 31) // 32             # 256 → 8
blocks = values.reshape(b, h, num_blocks, 32)
```

`reshape` 沿最后一维切 32——这就是"per-head、不跨 head"的全部实现：块索引
`(token, head, block_in_head)`，任何跨 head/跨 token 的组合在形状上就构不成。

### ③ 特殊块分类（L189-193）

```python
nan_blocks = isnan(blocks).any(-1)              # 含 NaN → 整块作废
inf_blocks  = isinf(blocks).any(-1) & ~nan_blocks
finite_abs = nan_to_num(blocks.abs(), nan=0, posinf=0, neginf=0)  # amax 只看有限值
```

三类块走向不同的 scale 分支（§4 速查表）。NaN 优先级最高：NaN 块 scale=0xFF，
此时块内数据码全部清零（L231-233），NaN 完全由 scale byte 表达。

### ④ scale 计算（L194-214）

```python
amax = finite_abs.amax(-1)                      # (b, h, num_blocks)
scale_exp = floor(log2(amax)) - 2               # 规范公式
scale_exp = where(amax == 0, -127, scale_exp)   # 全零块 → 2^-127（最小规格值）
scale_exp = where(inf_blocks, +127, scale_exp)  # Inf 块 → 2^127
scale_exp = clamp(scale_exp, -127, 127)         # E8M0 指数域
scale_bytes = (scale_exp + 127).to(uint8)       # 存原始 byte；NaN 块覆写 0xFF
```

`floor(log2(amax)) - 2` 即 `log2(2^floor(log2(amax)) / 4)`。`amax` 来自 fp32
值集（bf16 升上来的），`log2` 在 fp32 域精度 24 bit，floor 边界判定稳定。

### ⑤ E2M1 转换：saturating RNE + ties-to-even（L216-229）

```python
scaled = blocks / exp2(scale_exp)               # fp32 除以 2 的幂：只动指数，精确
scaled = nan_to_num(scaled, nan=0, posinf=6, neginf=-6)
abs_vals = scaled.abs().clamp(max=6)            # 饱和（saturating）

distances = (abs_vals.unsqueeze(-1) - e2m1_values).abs()   # 到 8 个码点的距离
tied = distances == distances.amin(-1, keepdim=True)
even_tied = tied & (code_ids % 2 == 0)          # tie 候选中尾数位为 0 的
magnitude_bits = where(even_tied.any(-1), argmax(even_tied), argmax(tied))
```

- **无 tie**：`argmax(tied)` = 唯一最近码。
- **tie**：选偶 code（mantissa bit 0）——OCP MX 的 ties-to-even。例如
  `1.25` tie 于 {1.0(code 2), 1.5(code 3)} → 选 1.0；
  `1.75` tie 于 {1.5, 2.0} → 选 2.0(code 4)。
- **实现消歧点**：`5.0` tie 于 {4.0(code 4), 6.0(code 6)}——两者都是偶 code
  （E2M1 的 4→6 间隔为 2，中点两侧尾数位同为 0），`argmax` 取首个 → **5.0 → 4.0**
  （向下）。这是规范空白处的实现决策，conformance 测试把它锁死。

### ⑥ 符号、打包（L230-238）

```python
fp4_vals = magnitude_bits | (signbit << 3)      # 4-bit 码
fp4_vals = where(nan_blocks, 0, fp4_vals)       # NaN 块数据清零
packed = fp4_vals[..., 0::2] | (fp4_vals[..., 1::2] << 4)   # 两两打包
```

打包约定：**低 4 位 = 偶数 index 元素，高 4 位 = 奇数 index 元素**。

一个 packed byte 的位布局：

```
 bit    7 6 5 4    3 2 1 0
       [奇元素码]  [偶元素码]
        s|e e m    s|e e m   # 每 nibble 内：bit3=符号，bit2-1=指数，bit0=尾数
```

两个 nibble 各自用满 4 bit（码值 0-15），位域互不重叠，`|` 与 `+` 等价，`|`
更明确表达位域拼接。配对机制（以 8 码为例，下标 0-7）：

```python
fp4_vals[0::2]  # 步长2切片 → 下标 0,2,4,6（位置编号为偶）→ [c0,c2,c4,c6]
fp4_vals[1::2]  #            → 下标 1,3,5,7（位置编号为奇）→ [c1,c3,c5,c7]
# 第 j 个 byte 装原数组下标 2j 与 2j+1 这对相邻元素：
packed = [c0|c1<<4, c2|c3<<4, c4|c5<<4, c6|c7<<4]
```

配对方式可任选，硬约束是解包（L273-274 `& 0x0F` / `>> 4` 写回同样切片）
与打包用同一套规则，交错拼回即还原原序列。用 uint8 手工打包而非
`torch.float4_e2m1fn_x2`：切片、位移、查表等纯 tensor 操作及 CUDA graph
捕获、CPU/CUDA bit-exact 对比对 uint8 最直接，且与 NVFP4 路径输出 dtype
（uint8）一致；打包阶段不解读位含义，解包查表后 normal/subnormal 编码
公式才生效。

## 3. `batched_dequantize`：纯查表乘法（L240-295）

反量化**没有任何舍入**，是量化的精确逆（除饱和信息不可恢复外）：

1. **形状契约**（L249-264）：packed 末维 = `(logical_dim+1)//2`，scale 末维 =
   `num_blocks`——两个 buffer 的对齐关系由形状断言锁死。
2. **unpack**（L266-275）：`& 0x0F` 取低 nibble → 偶 index；`>> 4` 取高 nibble →
   奇 index。与打包方向严格对称。
3. **查表**（L277-280）：`code & 0x07` 查 `E2M1_VALUES` 表得幅度，`code & 0x08`
   还原符号。查表无算术 → 无舍入。
4. **scale 应用**（L289-294）：

   ```python
   scale_exp = scale_bytes.to(int16) - 127
   scales = exp2(scale_exp)                       # 0xFF → NaN（L290-293 显式判）
   output = (blocks * scales.unsqueeze(-1)).flatten(-2)[..., :logical_dim]
   ```

   乘法后摊平、裁掉 pad。fp32 乘完 `.to(bf16)` 输出——这是反量化侧唯一的舍入，
   与 L1/L2 golden 的"effective K/V"语义一致。

为什么两侧 block 边界天然对齐：quantize 和 dequantize 用**同一个
`(…, num_blocks, 32)` reshape**（L187 vs L285-287），block 划分由形状推导，
不存在两套边界约定。

## 4. 特殊值语义速查表

| 块内容 | scale byte | scale 值 | data codes |
|---|---|---|---|
| 全零 | `0x00` | 2^-127 | 全 0 |
| 含 NaN | `0xFF` | NaN | 全 0（L231-233 清零） |
| 含 ±Inf（无 NaN） | `0xFE` | 2^127 | Inf 元素 → ±6（饱和），其余元素 ≈ 0 |
| 常规 | `exp+127` | 2^exp | RNE 码 |

## 5. 三个关键设计决策

1. **fp32 中间**：量化链只保留一个舍入点（规范 RNE）。fp32 完全包含 bf16 值集、
   除以 2^k 精确、log2 边界稳定，生产实现与独立 oracle 才能在 CPU/CUDA 双平台
   **bit-exact**。对比同文件的 `FP4MXBlock16KVQuantizeUtil`（bf16 链 + `ceil`
   近似 + half-up 计数法）：它是无 oracle 契约的工程变体，精度语义宽松。
2. **eager 而非 `@torch.compile`**：torch 2.11 inductor 对
   `slice + pad-to-even + nibble-pack` 图形会跳过最后一个输出字节的写入
   （验收记录发现 F1）。bit-exact 场景下编译输出必须逐字节验证后才可信，
   因此保持 eager 纯 tensor（仍可 CUDA graph 捕获），性能留给 L3 kernel。
3. **距离法而非 bounds 计数法**：FP4MXBlock16 用 `sum(abs >= BOUNDS)` 计数
   （round-half-up，无 tie 语义）；MXFP4 必须实现 ties-to-even，只能显式算
   距离 + 奇偶消歧。这是"规范实现"与"近似实现"在代码结构上的分水岭。

## 6. 与测试/oracle 的挂钩

- `test/registered/unit/layers/quantization/test_mxfp4_kv_codec.py`：
  conformance 覆盖全部 E2M1 code、RNE 中点、饱和、scale 指数边界（±127）、
  全零块、NaN/±Inf、head 隔离、partial block；CPU 门禁 + CUDA byte parity。
- `python/sglang/test/kits/attention_unittest/attention_methods/mxfp4_decode_attention.py`
  的 `mxfp4_quantize_reference`：独立 oracle，按同一规格重写（不 import 生产
  实现），两者 packed/scale bytes 逐字节相等——本文 §2-§4 的每条语义都是
  这个等式的组成部分。
