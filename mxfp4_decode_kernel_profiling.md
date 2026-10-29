# MXFP4 KV decode kernel 优化日志

> 承接 `final_lut_profiling.md`（LUT 优化 + 容量账本）。本文档只记录 **decode 算子层面**的
> 归因与优化，按轮次追加。
>
> 环境：单卡 RTX 5090（SM120，**170 SM**，31.4 GiB）；分支 `eightina/qwen3.8-kv-mxfp4`；
> 模型 `unsloth/Qwen3.8-27B-NVFP4`（64 层 = 16 full attention GQA 24Q/4KV head_dim 256 + 48 GDN）。
> 生产 decode 配置：`BLOCK_H=16`、`num_warps=4`、`BLOCK_N=64`、`max_kv_splits=85`（见 §3.2）。

单 SM 硬件预算（`torch.cuda.get_device_properties` 实测）与 stage-1 当时（§2，nibble interleave 之前）的占用：

| 资源 | 容量/SM | stage-1 占用（warps=8，256 线程）| 允许驻留 |
|---|---|---|---|
| 寄存器文件 RF | 65,536 regs = 256 KiB | `n_regs=255`（ISA 硬上限）× 256 = 65,280 → **99.6%** | **1 CTA/SM** ← 瓶颈 |
| 共享内存 SMEM | 102,400 B = 100 KiB（单 CTA 默认 48 KiB / opt-in 99 KiB）| 43,008 B = 42 KiB → 42.0% | 2 CTA/SM |
| 线程 | 1536（单 CTA 上限 1024）| 256 | 6 CTA/SM |

`spills=16`（warps=8）。L2 96 MiB 为全卡共享。

> 这一列是 §2 时的状态。§3.2 把它改成 178 regs / 0 spills / 2 CTA/SM，
> §4.5 给出当前所有 `(BLOCK_N, num_warps)` 组合的完整驻留表。

---

## 1. 逐算子归因（profile）

### 1.1 命令

```bash
# 两套服务（交替占卡，同 flag）
MEM=0.85 MAMBA=8 CTX=200000 EXTRA="--disable-prefill-cuda-graph" /tmp/launch_qwen38_mxfp4.sh
MEM=0.85 MAMBA=8 CTX=200000 EXTRA="--disable-prefill-cuda-graph" /tmp/launch_qwen38_fp8.sh
#   MXFP4: --kv-cache-dtype mxfp4 --prefill-attention-backend flashinfer --decode-attention-backend triton
#   FP8  : --kv-cache-dtype fp8_e4m3 --attention-backend flashinfer
# 池子容量：MXFP4 160,109 / FP8 105,071 token（1.524×，与容量账本一致）

# 抓取 + 按 kernel 总耗时排序
python .prof_kernel_rank.py <port> 98000 /tmp/prof/<name> 48
```

脚本 [`.prof_kernel_rank.py`](.prof_kernel_rank.py) 的核心是服务端 torch profiler：

```python
POST /start_profile {"output_dir": OUT, "activities": ["CPU","GPU"],
                     "profile_by_stage": true, "num_steps": 32}   # 分出 EXTEND / DECODE 两份 trace
POST /generate      {...}                                        # num_steps 到达后自动停
# 解析：聚合 traceEvents 里 cat=="kernel" 的 dur，按 name 求和降序
```

坑：① 请求失败会把 profiler 留在 armed 态，下次 start 报 `already in progress` 并可能带崩服务
（脚本在 start 前先无条件 stop）；② trace 异步落盘，立刻读会 `EOFError`（改为重试）；
③ 输入超池容量直接 400，filler 文本是 **1.0 token/word**。

### 1.2 核心结果（DECODE，32 步）

| 算子 | FP8 @98k | MXFP4 @77k |
|---|---|---|
| cutlass GEMM ×2 | 683.7 ms（**84.0%**）| 684.4 ms（**75.4%**）|
| attention | 66.6 ms（8.2%）| 165.9 ms（18.3%）|
| GDN + norm + quant + 其余 | ~64 ms | ~57 ms |
| **每步合计** | **25.4 ms** | **28.3 ms** |

**单层单步 attention：FP8 0.130 ms vs MXFP4 0.324 ms（2.5×）**（512 次 = 16 层 × 32 步）。

EXTEND 侧另有一笔：MXFP4 的 PLAIN 反量化 eager 链（`where` 818 ms、LUT `index` 763 ms、
`direct_copy` 657 ms、`neg` 521 ms、`BinaryFunctor` 521 ms、uchar 位运算 ~580 ms …）
合计 **~4.6 s，占 prefill kernel 时间 ~28%**，FP8 侧完全不存在。

### 1.3 结论

1. **长上下文 TPOT 掉队 100% 归因于 attention kernel**：每步差 2.9 ms，attention 差值
   99.3 ms ÷ 32 ≈ 3.1 ms，对得上。唯一靶子是 `_mxfp4_grouped_decode_stage1_kernel`，
   基准线是 flashinfer FP8 的 0.130 ms/层/步。
2. **decode 的天花板只有 ~10%**：GEMM 占 75–84% 且两边分毫不差（684 ms，与 KV dtype 无关）。
   attention 即使打平 FP8，每步也只从 28.3 → 25.4 ms。再往下只能靠减少步数（MTP）。
3. **dequant 优化只影响 TTFT**：decode 走 `_native_fp4` 内联读 packed E2M1 + E8M0，
   trace 里确实没有任何反量化算子；eager 链全在 EXTEND。

---

## 2. 第一轮：wave quantization（kv_splits 填满一波）

### 2.1 发现：SM 占用率只有 0.75 波

顺着 §1 的靶子看 stage-1 的 launch 配置，grid 是 `(batch, head_tiles, max_kv_splits)`：

- `head_tiles = cdiv(24, min(BLOCK_H=16, kv_group_num=6)) = 4`
- 生产 `max_kv_splits = 32`
- → **128 CTA，而卡上有 170 个 SM**

而 stage-1 的寄存器占满了 RF（255 regs × 256 线程 = 65,280 / 65,536 = **99.6%**），
**每个 SM 只能驻留 1 个 CTA**。所以 128 CTA = 0.75 波，**42 个 SM 全程空闲**，
而且剩下 128 个 SM 各自要多算 1/0.75 的 KV。

> 更正：`final_lut_profiling.md` §3.3 记的「LUT 后 227 regs + 0 spills」在当前 build 的
> 这个配置下**复现不出来**，实测是 **255 regs + 16 spills**（warps=8）、smem 43,008 B。
> 「1 CTA/SM」的结论不变，且比 227 的口径更紧（卡着 65,536 的上限成立）。

为什么调度器给了 32：设备侧 [`get_num_kv_splits_triton`](python/sglang/kernels/ops/attention/metadata.py)
在按 head grid 分配 core 之前先把 core count 乘了 `log2(seq_len/64)`（98k 时 ≈10.6 倍），
算出 451 splits —— 这个膨胀适合 bf16 kernel 的轻量 CTA，对本 kernel 严重过头；
最终被 backend 里一句硬编码的 `max(max_kv_splits, 32)` 截住，恰好落在 0.75 波。

### 2.2 做法

```bash
python .bench_mxfp4_decode_kernel.py 65536 100000 150000   # CUDA-graph 口径，固定 q，扫配置
```

正确上限就是**让 grid 正好填满一波**：`splits = SM数 / head_tiles = 170 / 4 = 42`。
改 [`triton_backend.py`](python/sglang/srt/layers/attention/triton_backend.py) 里的 floor，
按设备 SM 数和 head grid 推导，不再硬编码：

```python
head_tiles = (self.num_head if kv_group_num == 1
              else triton.cdiv(self.num_head, min(_GROUPED_BLOCK_H, kv_group_num)))
wave_kv_splits = max(1, self.device_core_count // head_tiles)
self.max_kv_splits = max(self.max_kv_splits, wave_kv_splits)   # 显式更大的 server arg 仍然胜出
```

对其它 head 配置同样落在一波内：24Q/4KV → 42 splits（168 CTA）、32Q/8KV → 21（168）、
16Q/16KV(MHA) → 10（160）。

### 2.3 结果：kernel −21%

stage1+stage2，bs=1，CUDA-graph replay，`BLOCK_H=16`/`warps=8`/`BLOCK_N=64`：

| splits | CTAs | 波数 | 65k | 100k | 150k |
|---|---|---|---|---|---|
| **32（原）** | 128 | 0.75 | 0.197 ms | 0.344 ms | 0.515 ms |
| 36 | 144 | 0.85 | — | 0.309（−10.0%）| 0.462（−10.4%）|
| 40 | 160 | 0.94 | 0.160（−18.6%）| 0.284（−17.5%）| 0.417（−19.1%）|
| **42（新）** | **168** | **0.99** | **0.156（−20.8%）** | **0.271（−21.3%）** | **0.399（−22.6%）** |
| 43 | 172 | 1.01 | — | 0.409（**+18.9%**）| 0.465（−9.6%）|
| 48 | 192 | 1.13 | 0.267（+35.7%）| 0.456（+32.7%）| 0.689（+33.7%）|
| 84 / 85 | 336 / 340 | 2.0 | — | 0.279（−18.8%）| 0.408（−20.7%）|
| 128 | 512 | 3.0 | — | 0.293（−14.9%）| 0.430（−16.5%）|

**单层单步 attention：0.344 → 0.271 ms（100k），与 FP8 的 0.130 ms 差距从 2.5× 收到 2.1×。**

两个值得记的细节：

- **43 splits 比 42 只多一个 CTA，就慢 51%**（0.271 → 0.409）。这一个 CTA 独占第二波，
  整个 kernel 等它 —— wave quantization 的边界极其陡峭，调参时必须按 SM 数对齐，不能"大约"。
- 两满波（84/85 splits）与一满波几乎等效（−18.8% vs −21.3%），但没有额外收益，
  且 `attn_logits` 缓冲按 `max_kv_splits` 线性涨（42 时 bs48 下 49.5 MB），所以取一波。

### 2.4 同轮扫掉的其它旋钮（均无收益，已排除）

| 旋钮 | 结果 | 说明 |
|---|---|---|
| `BLOCK_H` 16 → 8 | ±1.4%，输出 **bit-identical** | 本模型 `kv_group_num=6`，16 行的 dot 只有 6 行有效、acc 占 16 KiB 寄存器，本以为是大头；实测无影响 → kernel 不是 dot/寄存器容量瓶颈 |
| `num_warps` 8 → 4 | **+233%** | 彻底否掉 `final_lut_profiling.md` 待办里"spill 清零后试 num_warps=4"的猜测。机制：**每线程 255 regs 是 ISA 硬墙**，减半 warp 数并不能让每线程用双倍寄存器，实测两者 `n_regs` 都是 255，但 tile 元素从 64/线程涨到 128/线程，多出来的只能外溢 —— **spills 16 → 302** |
| `BLOCK_N` 64 → 128 | +60.5% | |
| `BLOCK_N` 64 → 32 | +4.9% | |
| `BLOCK_H` → 4 | rel_err 1.5e+00 | **既存约束**：`BLOCK_H < kv_group_num` 且不整除时，一个 head tile 会跨两个 kv head，结果是错的（上游 stock grouped kernel 同构）。生产 `BLOCK_H=16 ≥ 6` 不受影响 |

### 2.5 下一步

attention 仍比 FP8 慢 2.1×，且已排除 tile/warp 维度 → 瓶颈在**访存与 gather**，
外加一条由 §0 硬件预算表引出的新路径：**SMEM 只用了 42%，恰好够放 2 个 CTA
（2 × 43,008 ≤ 102,400），卡住占用率的纯粹是 RF**。若能把寄存器压到 ≤128/线程，
每 SM 可驻留 2 个 CTA，340 CTA（splits=85）就从两波变一满波，延迟隐藏也跟着翻倍。

1. **nibble interleave**：`byte_idx = offs_d // 2` 让每个 packed byte 被加载两次，
   packed 载入指令数和 code tile 寄存器都能减半 —— 既省访存，又是上面那条
   "压寄存器换 2 CTA/SM"的主要抓手，顺带有机会把残留的 16 个 spill 清零。
   → **已完成，见 §3。**
2. LUT gather 本身：每个 (token, 元素) 一次 `tl.load(E2M1_LUT + idx)`，K/V 各 256 次/行。
   若 nibble interleave 后仍是瓶颈，再考虑 16 项 magnitude 表 + 按 block 广播的指数偏置加法
   （需重新验证次正规/inf/NaN 的 bit-exact 契约）。

---

## 3. 第二轮：nibble interleave

### 3.1 做法

原来按**逻辑元素**寻址：`byte_idx_k = offs_d // 2` → `0,0,1,1,2,2,…`，所以
`[BLOCK_N, 256]` 的 tile 里**每个 packed byte 被加载两次**，还要一次 `tl.where` 选 nibble；
标量表同样被读两次。现在改成按**packed byte** 寻址，一次加载就就地展开两个 nibble：

```python
offs_half = tl.arange(0, HALF_D)        # 128 个 byte 通道，而非 256 个元素
kbyte = tl.load(K_Packed + k_row[:, None] + offs_half[None, :], ...)   # 载入量减半
ksbytes = tl.load(K_Scales + ks_row[:, None] + blk_idx_k[None, :], ...)  # 两个 nibble 共用
k_scale_base = ksbytes.to(tl.int32) << 4
k_lo = tl.load(E2M1_LUT + (k_scale_base | (kbyte & 0x0F)))        # 元素 2j
k_hi = tl.load(E2M1_LUT + (k_scale_base | ((kbyte >> 4) & 0x0F))) # 元素 2j+1
qk = tl.dot(q_lo, tl.trans(k_lo)) + tl.dot(q_hi, tl.trans(k_hi))
```

两个关键正确性依据：

- **点积对 d 的置换不变**。两个半边对应 d 序 `[0,2,…,254]` 和 `[1,3,…,255]`，
  Q 按同样的奇/偶偏移加载即可，两个 partial dot 相加。V 侧置换落在**输出维**上，
  用两次 strided store（`2j` / `2j+1` 偏移）还原。
- **同一 byte 的两个 nibble 必在同一个 E8M0 block 内**：32 元素块 = 16 个 byte，
  而 `2j+1` 是奇数、永远不会跳过 `2j` 没跳过的 32 边界。所以**标量表也只读一次**。

LUT 本身没动，gather 次数不变（仍 256/行）—— 省的是 **packed 载入、scale 载入、nibble select
以及那个 `[BLOCK_N, 256]` 的 uint8 code 中间 tile**。

### 3.2 寄存器：spill 清零，解锁 2 CTA/SM

| | 改前 | 改后 |
|---|---|---|
| warps=8 | 255 regs + **16 spills** | **186 regs + 0 spills** → 1 CTA/SM |
| warps=4 | 255 regs + **302 spills** | **178 regs + 0 spills** → **2 CTA/SM** |

这正是 §2.5 预测的那条路径：warps=4 从「灾难性 +233%」变成最优选项，
而且因为只用 22,784 / 65,536 个寄存器，每 SM 能驻留 2 个 CTA，**一波从 170 CTA 变成 340**。
所以 §2.2 的公式要乘上占用因子，并从 kernel 模块读取每条路径的实测 footprint：

```python
_, ctas_per_sm = stage1_launch_config(kv_group_num, gpu_id)   # grouped: (4 warps, 2)
wave_kv_splits = max(1, ctas_per_sm * device_core_count // head_tiles)   # 2*170//4 = 85
```

24Q/4KV → 85 splits（340 CTA）、32Q/8KV → 42（336）、MHA 16Q/16KV → 10（160，
该路径未改，自动退回 warps=8 / 1 CTA/SM）。

### 3.3 结果：kernel 再 −46%，累计 −65%

stage1+stage2，bs=1，CUDA-graph replay：

| 阶段 | 配置 | 100k | 150k |
|---|---|---|---|
| 原始生产 | warps=8, splits=32 | 0.344 ms | 0.515 ms |
| §2 wave 修复 | warps=8, splits=42 | 0.271（−21%）| 0.399（−23%）|
| **+ nibble interleave** | warps=8, splits=42 | **0.145（−46%）** | **0.209（−48%）** |
| **+ warps=4 / 2 CTA/SM** | **warps=4, splits=84** | **0.120** | **0.171** |
| | 累计 vs 原始 | **−65.1%** | **−66.8%** |

wave 边界在新占用率下同样陡峭：splits=84/85（336/340 CTA）是 0.120 ms，
**88（352）直接跳到 0.186 ms（+28%）**。warps=2 更差（0.189），BLOCK_N=128 最差（0.214）。

> 口径提醒（由 §4.4 补充）：本表是 **CUDA-graph 连续重放的 L2-warm 值，且为 stage1+stage2
> 之和**。冷 L2（生产口径）下 100k 是 0.1434 而非 0.120，拆开为 stage1 0.1259 + stage2 0.0159。
> 表内的*相对*改进幅度仍成立（各行同口径）。

### 3.4 正确性

`test_decode_attention_mxfp4_sm120.py` 全部通过（4 tests + 6 subtests）：

| case | rel_l2 | frozen cap |
|---|---|---|
| `qwen_24_4_256_p16` | 1.155e-03 | 1.85e-03 |
| `gqa_hd64_p16_boundary` | 1.293e-03 | 1.85e-03 |
| `qwen_24_4_256_zero_req`（amax=0）| 1.191e-03 | 1.85e-03 |
| `mha_hd64_p1_contiguous` | 1.220e-07 | 3.5e-07（per-head kernel 未改）|

改变的只是 d 维的累加顺序（单个 K=256 的 dot 拆成两个 K=128 相加），
正是测试文档里写的「residual is kernel accumulation order only」，没有超出原有条带；
LUT 全组合 bit-exact 测试与 CUDA-graph 确定性测试也未受影响。

### 3.5 端到端：已超过 FP8

流式 ITL p50，98k 上下文，96 输出 token，三次重复波动 ≤ 0.02 ms：

| 配置 | ITL p50 | decode | vs FP8 |
|---|---|---|---|
| MXFP4 原始（splits=32）| 29.14 ms | 34.32 tok/s | +17.0% |
| MXFP4 + wave 修复 | 27.78 ms | 35.99 tok/s | +11.5% |
| **MXFP4 + nibble interleave** | **24.65 ms** | **40.57 tok/s** | **−1.1%** |
| FP8 baseline（flashinfer）| 24.92 ms | 40.14 tok/s | — |

**长序列 decode 的负债清完**：MXFP4 从比 FP8 慢 17% 到反超 1.1%，而且 KV 容量仍是
1.524×（160,109 vs 105,071，池子未变 —— splits 翻倍后 `attn_logits` 缓冲从 49.5 MB 涨到
100 MB，吃的是 static 之外的 4.46 GB 余量）。

按 §1.3 的账本核对：attention 从 5.2 ms/步降到 ≈1.9 ms/步，占比从 18.3% 降到 ~7.7%，
已低于 FP8 的 8.2% —— **decode 剩下的全是权重 GEMM 底盘（684 ms / 32 步 = 21.4 ms）**。

### 3.6 下一步

decode 侧的 KV dtype 工作到此为止：attention 已不再是瓶颈，GEMM 占 ~87%。再快只能靠
减少步数（MTP / 投机解码）或动权重量化，与本路线无关。还可做的零散项：

1. **per-head（MHA）kernel 同样改造**：它仍是 255 regs + 16 spills。本模型不走该路径，
   但 MHA 模型会受益；注意其 frozen rel_l2 是 3.5e-07（fp32 逐元素），改累加顺序后需重测。
2. 对其它 head 形状（非 24Q/4KV、非 head_dim 256）重测 `stage1_launch_config` 里的
   regs/smem 常量；当前值是按本模型形状实测的，其它形状下只是调度启发式偏差（不影响正确性）。
3. prefill 侧 PLAIN dequant 融合 kernel（§1.2 的 ~4.6 s eager 链）—— 只影响 TTFT。

> **本节结论已被 §4 部分推翻。** 「attention 已不再是瓶颈」是就*端到端占比*说的（~7.7%），
> 但就*算子自身*而言 MBU 只有 54%，仍有 1.66× 可挖，见 §4.5/§4.6。

---

## 4. Roofline / MFU / MBU：算子还剩多少空间

### 4.1 命令

```bash
# ① 实测本机可达峰值（不用厂商标称值做分母）。读带宽取 2 GiB fp32 的 sum()，
#    算力取大方阵 bf16 matmul；均 CUDA event 计时、多次取均
python - <<'EOF'
import torch
def t(fn, it=50):
    for _ in range(5): fn()
    torch.cuda.synchronize()
    s, e = (torch.cuda.Event(enable_timing=True) for _ in range(2))
    s.record()
    for _ in range(it): fn()
    e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e) / it / 1e3
n = 2 * 2**30
f = torch.empty(n // 4, dtype=torch.float32, device='cuda').fill_(1.0)
print(f"read  {n / t(lambda: f.sum()) / 1e9:8.1f} GB/s")
del f; torch.cuda.empty_cache()
x = torch.randn((8192, 8192), dtype=torch.bfloat16, device='cuda')
print(f"bf16  {2 * 8192**3 / t(lambda: x @ x, 20) / 1e12:8.1f} TFLOP/s")
EOF

# ② roofline + 访存孪生 + dequant 方案对照（冷 L2，含 stage1/stage2 拆分）
python .roofline_mxfp4_decode.py 65536 100000 150000

# ③ 占用率相关 launch 旋钮的冷 L2 重扫
python .sweep_cold_mxfp4.py 100000 150000

# ④ stage2 的流水深度扫参 + 逐位一致性检查（§4.7）
python .sweep_cold_stage2.py 32768 100000 150000

# ⑤ 整个 attention 的同口径对比：flashinfer FP8 vs MXFP4 stage1+stage2（§4.8）
python .bench_fp8_attention.py 8192 32768 65536 100000 150000

# ⑥ 单配置驱动，供 ncu 挂载（本机不可用，见下）
ncu --target-processes all --kernel-name regex:mxfp4 --launch-count 2 --csv \
    --metrics gpu__time_duration.sum,dram__bytes_read.sum,dram__bytes_write.sum,\
lts__t_sector_hit_rate.pct,l1tex__t_bytes.sum,\
l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_ld.ratio,\
sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active \
    python .ncu_mxfp4_decode.py 100000 2

# 查编译期驻留（regs/spills/smem → CTA per SM）
#   清 fn.device_caches.values() 里的 dc[0] 后重新 launch，再读 k.n_regs / k.n_spills /
#   k.metadata.shared（JITFunction 没有 .cache 属性）
```

**ncu 在本机不可用**：报 `ERR_NVGPUCTRPERM`。即使 uid=0，容器的 `CapEff=0xa80405fb` 缺
`CAP_SYS_ADMIN`（bit 21），无法从内部解开，需要宿主以 `--cap-add=SYS_ADMIN` 重启容器。
因此本节的 bytes 用**解析式模型**，并用访存孪生做实测校验（§4.3）。

实测峰值（RTX 5090）：

| | 实测 | 标称 | 比 |
|---|---|---|---|
| 读带宽 | **1686.6 GB/s** | 1792（512-bit × 28 Gbps）| 94% |
| copy（读+写）| 1525.4 GB/s | | |
| bf16 tensor | **241.8 TFLOP/s**（8192³；4096³ 为 227.7）| | |

复测会有 ~1% 漂移（时钟）：上述脚本重跑得 1686.3 GB/s / 243.6 TFLOP/s。
下文所有 MBU/MFU 的分母统一用 **1686.6 GB/s** 与 **241.8 TFLOP/s**。

### 4.2 口径：这些量怎么算的

**useful FLOPs** —— 算法本身必须做的乘加。decode 是两个 batched GEMV：

```
QK^T : 每个 q head 对 S 个 key 各做一次 head_dim 收缩 → 2 · H_q · D · S
P@V  : 同样规模                                        → 2 · H_q · D · S
useful = 4 · H_q · D · S = 4 · 24 · 256 · S = 24576 · S FLOP
```

系数 2 = 一次乘 + 一次加。100k → **2.458 GFLOP**。

**issued FLOPs** —— tensor core 实际被喂进去的量，含 tile 空转。`BLOCK_H=16` 而 GQA 组只有
`kv_group_num=6` 个有效 q head，`head_tiles=cdiv(24,6)=4`，故 M 维填充系数：

```
issued = useful · (head_tiles · BLOCK_H) / H_q = useful · 64/24 = 2.67 ×  → 6.554 GFLOP
```

**bytes** —— KV 的每个字节只被唯一一个 CTA 读一次（4 个 head tile 各对应一个 kv head），
无冗余，所以强制流量就是全量 KV：

```
kv_bytes = 2(K和V) · S · H_kv · (D/2 + D/32) = 2 · S · 4 · (128 + 8) = 1088 · S
```

其中 `D/2` 是 packed E2M1，`D/32` 是 per-block E8M0 标量。再加 `kv_indices`（4 个 head tile
各读一遍 S 个 int32）与 stage1 写 / stage2 读的 `attn_logits`（`H_q × splits × (D+1)` fp32）。
100k：KV 103.8 + indices 1.53 + logits 读写 4.00 = **109.3 MiB**。

**算术强度 AI** = useful / kv_bytes = 24576 / 1088 = **22.6 FLOP/B**。注意它与 S 无关 —— 是这个
kernel 的结构常数，加长上下文不会改变它的 roofline 位置。

**ridge point** = 峰值算力 / 峰值带宽 = 241.8e12 / 1686.6e9 = **143.4 FLOP/B**。AI < ridge 即
内存受限；这里差 **6.3×**，所以算力侧的任何优化都换不来时间。

**roofline 下限** = total_bytes / 峰值读带宽。100k → 109.3 MiB / 1686.6 GB/s = **0.0679 ms**。

**MBU** = total_bytes / 实测耗时 / 峰值读带宽。**MFU** = FLOPs / 实测耗时 / 峰值算力，
useful 与 issued 两种分子都报。

> 分子口径不精确处：§4.5 为可比起见对 stage1 与孪生用了同一个 `total_bytes`，但 stage1 要写
> 2.0 MiB logits 而孪生不写，且 109.3 里含 stage2 读 logits 的 2.0 MiB（两者都不该算）。
> 按各自真实流量重算是 stage1 **53.0%**（107.3 MiB）、孪生 **86.1%**（105.3 MiB），
> 差异 ≤3%，比值仍 1.62×，结论不动。另：`.sweep_cold_mxfp4.py` 的 MBU 列分子只用 `kv_bytes`，
> 所以同一配置比 `.roofline_mxfp4_decode.py` 低约 3 个点。

### 4.3 访存孪生（memory twin）是什么

`.roofline_mxfp4_decode.py` 里的 `_twin_kernel`：**保持 grid、`num_warps`、tile 形状、五笔 global
load（`kv_indices` / K packed / K scales / V packed / V scales）和地址算法完全一致，去掉 LUT
gather、两个 `tl.dot`、online softmax 和 logits 写回**，最后把读到的字节求和存一个标量防止被 DCE。

它回答一个反事实问题：**这个 kernel 如果除了搬这些字节什么都不干，能多快？**作用是把 46% 的
缺口切成两段，各有不同的治法：

```
roofline 下限  0.0679 ms  100%   理论：按峰值带宽纯流式
     ↓ ×1.12          访存模式本身的代价 —— 不改 KV 布局拿不到
访存孪生       0.0760 ms   89%
     ↓ ×1.66          计算没藏进访存延迟 —— 这才是可挖的那段
stage1        0.1259 ms   54%
```

只报「MBU 54% vs roofline 100%」会暗示有 1.85× 可挖。孪生指出其中 1.12× 来自布局事实：
128 B 的行跨 512 B、8 B 的 scale 行跨 32 B、还要过一层 `kv_indices` 间接寻址。
**真实可挖的是 1.66×，方向是提高重叠度而不是改访存模式**（后者已接近打满，coalescing 不是问题）。

孪生不是「正确 kernel 的下限」—— 正确的 kernel 必须做 dequant 和 dot，它只是 load 路径的下限，
衡量重叠得好不好。它同时是流量模型的**自洽校验**：若 bytes 少记，孪生 MBU 会被推到 100% 以上
（修 L2 口径之前确实出现过 148.7%）。现在落在 89.4%，说明模型没有少记。

### 4.4 两处口径更正

1. **§2 / §3 的 kernel 数字是 L2-warm 的。** KV 103.8 MiB vs L2 96 MiB，CUDA-graph 连续重放让
   约 92% 的 KV 常驻 L2，量到的是 L2 带宽而非 DRAM 带宽。冷 L2 下 100k 的 fwd 是
   **0.1434 ms 而非 0.1200**（+19%），65k 从 0.0763 涨到 0.1026（**+34%**）。生产中同一层两次
   attention 之间要流过 ~20 GB 权重，L2 必被冲干净，**冷值才是生产口径**。相对改进幅度仍然
   成立（两侧同口径），端到端 ITL 也不受影响（那是真实服务实测）。
   测法：每次计时前写一个 2× L2 的 192 MiB 缓冲，刷写动作留在 event 窗口之外。

2. **§3.3 表里的「kernel」一直是 stage1 + stage2 之和**，不是 stage-1 单独。拆开（100k，冷）：
   stage1 0.1259 + stage2 0.0159。stage2 **恒为 0.0159 ms，只随 splits 变、与序列长度无关**，
   在 100k 占 fwd 的 11%、65k 占 15%。

### 4.5 结果（冷 L2）

| | 100k | 150k |
|---|---|---|
| AI（useful）| 22.6 FLOP/B（ridge 143.4，差 6.3×）| 同（与 S 无关）|
| **MFU** useful / issued | **8.1%** / 21.5% | 8.5% / 22.7% |
| **MBU** stage1 / 孪生 | **54.0%** / 89.4% | 56.1% / 92.2% |
| stage1/roofline · stage1/孪生 · 孪生/roofline | 1.85× · 1.66× · 1.12× | 1.78× · 1.64× · 1.08× |

**MFU 在这里是干扰指标**：哪怕把 2.67× 的 tile 空转全消掉也换不来时间。该盯的是 MBU。

stage1 时间预算（完整闭合，无不明开销）：

| | 100k | 占比 |
|---|---|---|
| 纯 loads（孪生）| 0.0760 ms | 60% |
| LUT gather | +0.0429 | **34%** |
| dot + softmax | +0.0070 | 6% |

**dequant 方案对照**（相对纯 loads 的增量）—— 原假设「去掉 gather 改纯位运算会更快」被推翻：

| 方式 | 100k | 150k |
|---|---|---|
| LUT4096 fused（现生产）| +0.0429 | +0.0569 |
| **LUT16 × 重建 scale** | **+0.0313（−27%）** | +0.0430（−24%）|
| 纯位运算（无 gather）| +0.0574（**+34%**）| +0.0770 |

收益来自把表从 8 KiB 缩到 32 B（一个 warp 的 32 条 lane 从散落在 8 KiB 变成落在一两个 sector 内，
近似广播），**不是消掉 gather**：10 条整数指令 × 2.05 亿元素比一次 L1 常驻 gather 更贵。
这反过来佐证了当初选 LUT 是对的。重建 scale 用的是 `bits = sb << 7` 直接拼 bf16 的 2^(sb-127)。

**占用率只有 16.7%**（8 warps / 48 上限），且 RF 与 SMEM 正好同时卡在 2 CTA：

| BLOCK_N | warps | regs | spill | smem | 按RF | 按SMEM | CTA/SM | warps/SM | 占用率 |
|---|---|---|---|---|---|---|---|---|---|
| 32 | 4 | 168 | 0 | 25,600 | 3 | 4 | **3** | 12 | 25.0% |
| 32 | 8 | 115 | 0 | 25,600 | 2 | 4 | 2 | 16 | 33.3% |
| **64** | **4** | 178 | 0 | 43,008 | 2 | 2 | **2** | 8 | **16.7%** ← 现生产 |
| 64 | 8 | 186 | 0 | 43,008 | 1 | 2 | 1 | 8 | 16.7% |
| 128 | 4 | 255 | **106** | 77,824 | 2 | 1 | 1 | 4 | 8.3% |
| 128 | 8 | 174 | 0 | 77,824 | 1 | 1 | 1 | 8 | 16.7% |

冷 L2 重扫（100k；MBU 列分子只含 `kv_bytes`，故比上面的表低约 3 点）：

| BLOCK_N | warps | splits | CTA | CTA/SM | 占用率 | fwd | stage1 | stage2 | MBU | vs 生产 |
|---|---|---|---|---|---|---|---|---|---|---|
| 64 | 4 | 85 | 340 | 2 | 16.7% | 0.1443 | 0.1260 | 0.0159 | 51.2% | ref |
| **32** | **4** | **127** | **508** | **3** | **25.0%** | **0.1383** | **0.1133（−10%）** | 0.0235 | **56.9%** | **−4.1%** |
| 32 | 8 | 85 | 340 | 2 | 33.3% | 0.1576 | 0.1394 | 0.0161 | 46.3% | +9.2% |
| 32 | 2 | 170 | 680 | 4 | 16.7% | 0.2112 | 0.1800 | 0.0291 | 35.8% | +46.4% |
| 64 | 4 | 170 | 680 | 2 | 16.7% | 0.1613 | 0.1298 | 0.0297 | 49.7% | +11.8% |

**占用率不是单调有益**：`warps=8` 虽有 33.3% 却更慢（`BLOCK_N=32` 的 dot tile 被切太碎）；
`warps=2` 虽有 4 CTA/SM 但 regs 顶到 255 直接崩。`BLOCK_N=32` 此前从未测过（扫参表只有 64/128）。
注意改 `BLOCK_N` 会变累加顺序，扫参里的 `rel`（3.0e-03）是对 `BLOCK_N=64` 输出的 max-abs 差，
**与单测的 rel_l2-vs-golden 不是同一指标**，落地前必须重跑 frozen 测试。

**stage2 还带着 stage1 治过的病**：`_mxfp4_decode_stage2_kernel[(batch, head_num)]` = **(1, 24)，
24 个 CTA 跑在 170 个 SM 上，只用 14% 的卡**。它搬 3.1 MB 要 0.0235 ms，折合 **133 GB/s**。
这也解释了为什么 `BLOCK_N=32` 让 stage1 收 −10% 而 fwd 只收 −4.1%：splits 85→127 让 stage2
涨了 48%，吃掉一半收益。

### 4.6 下一步：三条有实测支撑的杠杆

按收益/风险比，建议顺序 **C → B → A**：

| | 做法 | 依据 | 预期 |
|---|---|---|---|
| **C** ✓ | stage2 按 head_dim 切 CTA —— combine 的分子对 d 各维独立，只有 lse 分母要全 splits 且是标量 | 24 CTA → ~190；现仅 133 GB/s | fwd ~−10% |
| **B** | `BLOCK_N` 64→32，splits 按 3 CTA/SM 重算为 127 | 冷 L2 实测 | fwd −4.1%（已测）|
| **A** | LUT4096 → LUT16 × 重建 scale | 孪生实测 dequant −27% | stage1 ~−9% |

> **C 已落地，但做法与此处设想的完全不同。** 按 head_dim 切 CTA 实测零效果；真正的因
> 是循环依赖链上的 load 延迟，修法是把分支换成谓词以解锁软件流水。见 §4.7。

叠加估计 fwd 0.1443 → ~0.112 ms（MBU ~64%）。硬下限是孪生 0.0760 + 最小 stage2 ≈ 0.078
（MBU 89%），再往下要改 KV 布局本身。

风险：**B** 改累加顺序，需重跑 frozen 测试；**A** 碰的是已 bit-exact 的 LUT codec，
单测对它要求最严（全 4096 组合 bit-exact），且落地时必须处理 E8M0 的退化标量（0 与 0xFF）——
孪生里的位运算版本为了只测成本并未定义它们；**C** 不改动任何数值。

### 4.7 杠杆 C 落地：stage2 谓词化

#### 两个被推翻的假设

§4.5 把 stage2 的 133 GB/s 归因于「24 个 CTA 只用 14% 的卡」。**这个归因是错的。**

1. **按 d 维切 CTA：零效果。** combine 的分子对 d 各维独立（`e_max`/`e_sum` 只由 per-split
   lse 标量决定），所以切分是 bit-exact 的。但 chunk 从 256 扫到 16、CTA 从 24 涨到 384，
   stage2 始终 0.0157~0.0160 ms（±2% 噪声），**chunk ≤ 32 还更差**。
   原因：chunking 是在用请求尺寸换 CTA 数、**在飞字节恒定**（24×2×1 KiB ≈ 192×2×128 B），
   结构上就不可能提高内存并发度。此路已整体撕除，不要重试。
2. **加大 `num_stages`（带分支时）：零效果。** 2/4/8/16 全部 0.0160 ms。

#### 真正的根因

stage2 耗时**只随 `MAX_KV_SPLITS` 线性变化**，与序列长度、CTA 数、请求尺寸全无关：

| splits | 85 | 127 | 170 | 255 |
|---|---|---|---|---|
| stage2 | 0.0159 | 0.0235 | 0.0297 | 0.0435 |

折合每个 split 固定 ~187 ns ≈ 520 周期 —— 循环携带依赖链的特征。而 `num_stages` 完全无效
指向具体原因：**两笔 load 裹在 `if split_kv_end > split_kv_start` 里，流水线器无法把数据
相关分支内的 load 提出来预取**，于是每轮迭代都在
`e_max → old_scale → acc` 这条链上完整支付一次 load 延迟。

#### 做法

把分支换成谓词：越界 split 贡献 `old_scale == 1`、`exp_logic == 0`，即原封不动地留下
`acc`/`e_sum`/`e_max`，语义与原分支一致。三个 `tl.where` 是必要的：首轮若为空 split，
`e_max - n_e_max` 会是 `-inf - (-inf) = NaN`，裸 `exp` 会污染累加器。

谓词化后 `num_stages` 立即生效（100k，冷 L2）：

| num_stages | 1 | 2 | 3 | **4** | 6 | 8 |
|---|---|---|---|---|---|---|
| stage2 | 0.0147 | 0.0178 | 0.0097 | **0.0080** | 0.0083 | 0.0096 |

注意 `num_stages=2` 的谓词版（0.0178）比原分支版（0.0159）**更慢** —— 谓词化本身是负收益，
收益全部来自它解锁的流水。两者必须成对修改。生产值定为 `_STAGE2_NUM_STAGES = 4`。

#### 结果与正确性

A/B（stash 新旧两版各跑一遍，stage1 两侧完全一致，确认未注入 stage1 变更）：

| seq | stage2 旧 | stage2 新 | |
|---|---|---|---|
| 32,768 | 0.0149 | **0.0080** | −46.3% |
| 100,000 | 0.0159 | **0.0080** | **−49.7%** |
| 150,000 | 0.0161 | **0.0083** | −48.4% |

正确性：14 个用例（seq ∈ {32, 64, 100, 1000, 32768, 100000, 150000} × max_splits ∈ {8, 85}）
按 int32 视图**逐位完全相同**，包括 `seq=32 / max_splits=85`（85 个 split 里 84 个为空，
正是谓词化所替换的路径），且全部 finite。单测 4 tests + 6 subtests 全过。

### 4.8 结果：整个 attention 的同口径对比

§1.2 的 FP8 数字（0.130 ms/层/步）来自服务端 trace，与 §4 的冷 L2 微基准不同口径，
不能并列。把 flashinfer FP8 放进**同一个 harness**（bs=1、24Q/4KV、head_dim 256、page_size 1、
每次计时前刷 L2、K 与 V 用独立张量）：

| seq | FP8 ms | FP8 流量 | FP8 MBU | **MXFP4 ms** | s1 | s2 | MXFP4 流量 | MXFP4 MBU | **MXFP4 vs FP8** |
|---|---|---|---|---|---|---|---|---|---|
| 8,192 | 0.0313 | 16.0 MiB | 31.8% | **0.0283** | 0.0188 | 0.0079 | 12.6 MiB | 27.7% | **−9.4%** |
| 32,768 | 0.0684 | 64.0 | 58.2% | **0.0609** | 0.0513 | 0.0079 | 38.5 | 39.3% | **−10.9%** |
| 65,536 | 0.1209 | 128.0 | 65.8% | **0.0965** | 0.0868 | 0.0082 | 73.0 | 47.0% | **−20.2%** |
| 100,000 | 0.1804 | 195.3 | 67.3% | **0.1351** | 0.1263 | 0.0079 | 109.3 | 50.3% | **−25.1%** |
| 150,000 | 0.2665 | 293.0 | 68.4% | **0.1891** | 0.1799 | 0.0083 | 161.9 | 53.2% | **−29.0%** |

**MXFP4 在所有长度上都快于 FP8（9~29%），且优势随上下文增长。**

两边结构完全同构，逐 kernel 对照（@100k）：

| | FP8（flashinfer）| MXFP4（triton）|
|---|---|---|
| 主 kernel | `BatchDecodeWithPagedKVCacheKernel` 0.1720 | stage1 0.1263 |
| merge / combine | `PersistentVariableLengthMergeStatesKernel` **0.0088** | stage2 **0.0079** |
| 合计 | 0.1809 | 0.1342 |

这给 §4.7 一个独立参照：修复前我们的 stage2（0.0159）**比 flashinfer 的 merge 慢 1.8×**，
修复后（0.0079）略快于它。

#### 要替换的一个错误推断

我曾根据 trace 的 0.130 ms 推断「flashinfer FP8 已在 ~92% MBU」—— **错的**。同口径冷 L2 下
FP8 只有 **67~68% MBU**，两边都远离 roofline。所以 MXFP4 的胜出**全部来自 1.88× 的流量优势**
（每 token/层：FP8 2048 B vs MXFP4 1088 B），而并非执行效率：100k 下 1.88× 的流量优势
只换成 1.33× 的时间优势，差的那一截正是 MBU 50.3% vs 67.3%。

推论：**若 MXFP4 做到 FP8 的 67.3% MBU 就是 0.101 ms（1.79× 于 FP8），做到孪生的 89% 则是
0.076 ms（2.4×）。** 这是杠杆 B/A 剩下的空间。

> 未查清的差异：本节冷 L2 的 FP8 @100k 是 0.1804 ms，而 §1.2 服务端 trace @98k 是 0.130，
> 相差 1.39×。可能来源：服务端的 page_size / KV 布局不同、flashinfer dispatch 到不同内核变体、
> 或我对 trace 的聚合方式。**本表内部两侧同口径，横向比较有效；但不要把本表的绝对值与
> §1.2 的 trace 数字混用。**
