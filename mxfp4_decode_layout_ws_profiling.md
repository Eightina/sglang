# MXFP4 KV decode：KV 布局与 warp specialization

> 承接 `mxfp4_decode_kernel_profiling.md`（§1 逐算子归因 → §6 scale 读取）。那份文档已到
> 856 行，本文档从第五轮起另起一卷，只记录 **KV 布局改造**与 **TMA / warp specialization**
> 两条线。口径、harness、baseline 全部沿用前卷，不重复推导。
>
> 环境：单卡 RTX 5090（SM120，170 SM，L2 96 MiB，SMEM 100 KiB/SM，RF 65,536/SM）；
> 分支 `eightina/mxfp4-cuda-dev`；`unsloth/Qwen3.8-27B-NVFP4`（16 层 full attention
> GQA 24Q/4KV head_dim 256）。生产 decode：`BLOCK_H=16`/`warps=4`/`BLOCK_N=64`/
> `max_kv_splits=85`/`page_size=1`。
>
> 计时口径：`.bench_fp8_attention.py`，bs=1、冷 L2（每次计时前刷 L2）、K 与 V 独立张量、
> CUDA event 只包 launch。对照是同 harness 的 flashinfer FP8 decode。可达峰值读带宽
> 1686.6 GB/s（实测，非标称）。

---

## 1. 起点：36% 的缺口不在内存，在指令流

### 1.1 接手时的状态

前卷第四轮结束时（硬件 FP4 cvt + scale 窄读广播），cold L2：

| seq | FP8 ms | **MXFP4 ms** | s1 | s2 | MXFP4 MBU | **vs FP8** |
|---|---|---|---|---|---|---|
| 8,192 | 0.0315 | **0.0284** | 0.0190 | 0.0079 | 27.6% | **−9.9%** |
| 32,768 | 0.0684 | **0.0598** | 0.0503 | 0.0079 | 40.0% | **−12.6%** |
| 65,536 | 0.1211 | **0.0962** | 0.0862 | 0.0080 | 47.2% | **−20.6%** |
| 100,000 | 0.1802 | **0.1345** | 0.1258 | 0.0077 | 50.5% | **−25.4%** |
| 150,000 | 0.2663 | **0.1896** | 0.1796 | 0.0080 | 53.1% | **−28.8%** |

stage1 编译占用：**205 regs / 0 spills / 43,008 B smem / 2 CTA/SM**（RF 侧 205×128 =
26,240 of 65,536；SMEM 侧 43,008 of 51,200 —— 两边都只剩一点余量）。

参照线：FP8 的 MBU 是 67~68%，load-only 孪生 kernel 是 89.4%。我们 50.5%。

### 1.2 五个探针：缺口与内存行为无关

全部在生产 kernel 上做单点消融，数值可能错，只看时间；s1 @100k：

| 探针 | 改了什么 | s1 | SASS `LDG.E.U8` | 结论 |
|---|---|---|---|---|
| 基线 | — | 0.1258 | 8 | |
| **P1** | scale 瓦片 `[64,8]` → `[64,1]`，每 token 只读 1 字节 | 0.1250 | 8 | 读取**宽度**无关 |
| **P2** | 保留 load 与 E8M0 转换，去掉 hi 两半的逐元素乘 | 0.1258 | 8 | 逐元素**乘法**无关 |
| **P4** | 寻址改 `kv_loc % 256`，工作集压到 ~8 KB 必然命中 L1（延迟 ~400cy → ~30cy），指令数与散列模式不变 | 0.1255 | 8 | **延迟**无关 |
| **P6** | 寻址改 `offs_n`，slot 顺序化（gather → 合并，sector 数 ÷4），指令数与地址算术不变 | 0.1244 | 8 | **访存模式**无关 |
| **P3** | scale 改由已在寄存器里的 `kbyte` 派生：load 与地址运算整个消失，E8M0 转换与逐元素乘保留 | **0.0822** | **0** | **就是这 8 条指令本身** |

P3 的整体效果：@100k 合计 0.0913、MBU **74.4%**、对 FP8 **−49.4%**；
@150k s1 0.1143、合计 0.1241、MBU **81.1%**、对 FP8 **−53.3%**。

补充排除项：`num_stages` 2→3 在 100k 略好（0.1241~0.1244）但 150k 更差
（0.1807~0.1818 vs 0.1797~0.1800），来回各跑两遍都是这个形状，判为无收益。
P3 消融前后寄存器 205 → 224、SMEM 均 43,008、均 2 CTA/SM，**不是占用率问题**。

### 1.3 机制更正：推翻前卷 §6.4 的理由

前卷 §6.4 把这 36% 归因为 **sector 效率**——「每 token 的 8 字节 scale 独占一个 32 B
sector，每 CTA 每轮 128 个散列 sector，6% 的字节吃掉 36% 的时间」，并据此说唯一出路是
把 scale 并进 packed 行以改善 sector。

**P6 把这个理由否了。** 把 slot 顺序化之后 sector 数降到约 1/4、访问完全合并，时间只动
1.1%。加上 P4（延迟无关）、P1（宽度无关），能同时解释五个探针的只有一条：

> 代价来自**循环体里存在第二条同步的全局 load 指令流**（8 条 `LDG` 加上为它们构造
> `[BLOCK_N, NUM_BLK]` 指针瓦片的地址算术），而与这条流的延迟、访问模式、宽度都无关。

packed 数据没有这个代价，不是因为它连续，而是因为它走 `LDGSTS.E.BYPASS.128`——**由
硬件异步搬运、地址在发射时一次算完、不占 consumer 的后续发射槽**。

> 结论（把 scale 并入 packed 行 → 对 FP8 −47%）不变，但**理由必须改**：它的作用不是改善
> sector，而是让 scale 的 load 与寻址折进已有的异步流、从指令流里消失。这个区别决定了
> 哪些替代方案还活着 —— 按旧理由，warp specialization 无用；按新理由，它正对靶心。

### 1.4 外部参照：LeetCUDA 第 15 章（SM120 TMA + mma.sync + WS）

来源：<https://leetcuda.readthedocs.io/en/latest/ch14-sm120-tma-ws.html>。
与本仓已实测的 SM120 ISA 边界（`.gpu-wiki/blackwell/sm100-vs-sm120.md`）一致：TMA 与
mbarrier 有，WGMMA / tcgen05 / TMEM / cluster / multicast 没有，consumer 退回
`mma.sync.m16n8k16` + `ldmatrix`。

它对 WS 收益的归因正是 1.2 节得到的那条：

- **分工层**：producer/consumer 分工消除的是「搬运与计算竞争**发射槽**」的结构性开销，
  与计算指令是哪一代无关；consumer 循环里不再有 gmem 地址算术与提交指令。
- **同步层**：数据驱动的 full/empty mbarrier 替代 `__syncthreads`，warp 间不再互等。
- **搬运层**：TMA 单线程提交整块 2D tile，地址与 OOB 全在 descriptor 里预计算。

实测 2048³ HGEMM 104.66 → 164.64 TFLOPS（1.57×），但作者自己标注 BK / stage 数 / warp
组织同时变了，该比值属整体方案而非单变量因果。且那是 AI = 64 的 GEMM（riding L2），
我们是 AI = 22.6 的 decode，不能直接搬运收益倍数。

### 1.5 两条硬约束 —— 直接否掉「照搬配方」

**① `page_size=1` 下 TMA 用不了。** `cp.async.bulk.tensor` 搬的是**致密 2D box**；我们的
K/V 按 page table 随机映射 slot，是 gather，descriptor 表达不了。所以三层收益里我们只能
拿到分工层与同步层，**拿不到搬运层**——而地址算术恰是 P3 证明值钱的那部分。
要拿到它，必须先让 `page_size ≥ BLOCK_N`，使一个 tile 落在一个连续页内。

**② 寄存器池会把我们逼回 1 CTA/SM。** 笔记的配置是 `__launch_bounds__(kNumThreads, 1)`、
producer `dealloc<40>` + consumer `alloc<232>`，池占用 Σ32·r_w = 4×32×40 + 4×32×232 =
34,816 < 65,536。**两个这样的 CTA 需要 69,632 > 65,536，结构上只能 1 CTA/SM。**
我们是深度内存受限的 decode，2 CTA/SM 是在飞访存并行度的来源；前卷第三轮第一版就因为
SMEM 越界掉到 1 CTA/SM 而 +70%。所以不能照搬 8 warp（128 producer + 128 consumer）的
形态，只能在 128 线程内切 2 producer + 2 consumer，而 consumer 从 4 warp 变 2 warp 会
改变 `[16,128] × [64,128]` 的 dot 分块。这是设计约束，不是细节。

附带记下笔记里三个与本仓已有经验同类的静默失效陷阱（SM120 上 cluster 相关的东西
编译器不报错但不生效，与已实测的「`barrier.cluster.*` 在 sm_120a 照样 ACCEPT」同构）：

- TMA 目的地址写 `shared::cluster` → ptxas（CUDA 13.2）视作隐式 extern-call 边界，
  把同 kernel 的 `setmaxnreg` **全部静默丢弃**，只留 info 级 C7506（需 `-Xptxas -v` 才可见）。
  写 `shared::cta` 则 sm_120a / sm_120f 都完整保留。
- `__launch_bounds__(N)` 缺第二参数 → PTX 不带 `.minnctapersm` → ptxas 以 C7508 丢弃
  `setmaxnreg`。WS kernel 一律写 `(N, 1)`。
- SASS 助记符是 `USETMAXREG`（**没有 n**），`grep setmaxnreg` 在 SASS 上全程假阴性。
- smem 基址须 `__align__(1024)`（128B swizzle 周期）；只对齐 128B 时 TMA 以非零相位写入、
  软件公式按零相位读出，输出系统性错误且不越界不报错。

### 1.6 决策：两条方向攻同一个缺口，先做便宜的那条

| 方向 | 攻击的缺口 | 实测上限 | 代价 / 风险 |
|---|---|---|---|
| **A. scale 并入 packed 行** | 同一个 36% | P3：对 FP8 **−47%**（扣掉 P3 省掉的 6% scale 流量后的估计落点），MBU ~72% | Triton 内可做，不碰 warp 分工、不赌寄存器池；但跨 kernel / KV pool / prefill 三层 |
| **B. page 化 + TMA + WS** | 同一个 36% + 残余 1.2× | 孪生地板 ~0.078 合计、MBU ~89% | 要写 CUDA；要重排 warp 分工；**有掉到 1 CTA/SM 的实质风险** |
| C. 只对 stage2 做 WS | stage2 全部 0.0077（占总量 6%）| 整体 ~3% | 不值得：已 −49.7% 修过，且已略快于 flashinfer 的 merge（0.0088）|

**顺序定为 A → 再评估 B。** 理由：

1. A 和 B 攻的是同一个 36%，**不叠加**，所以先做能用现有工具链拿到的那条。
2. A 不需要改 warp 分工，也不需要赌寄存器池——而 B 的两条硬约束（1.5 节）都还没法消除。
3. 如果 A 的布局顺手做成 **page 对齐**（`[page, head, tok_in_page, 128+8]`），它同时解除
   约束 ①，成为 B 的前提。即 A 不只是 B 的替代，还是 B 的垫脚石。
4. A 落地后 MBU 预期 ~72~74%，已超过 flashinfer FP8 的 67~68%。届时残余只有 ~1.2×，
   再决定值不值得为它写 CUDA。

### 1.7 本轮要先摸清的改动面（A）

不写代码，先确认：

- `MXFP4KVQuantizeUtil` 的量化输出布局与调用方
- KV pool 的 cell 布局与 `compute_cell_size`（含前卷记过的 `plain_scratch_size` 计账预留）
- prefill 侧：MXFP4 走 PLAIN access kind（BF16 flashinfer），读的是同一份 buffer
- `fp4_kv_cache_quant_method.py` 注册表需要动哪几项
- 行宽从 128 B 变 136 B 对对齐与 `LDGSTS.E.BYPASS.128` 的影响（这是 A 最大的未知）

验收仍按前卷的轮次契约：完整 attention stage1+stage2 vs FP8 baseline + MBU，
先读 `n_regs` / `metadata.shared` 再读时间，输出与现版本做逐位或 frozen 阈值对比。

---

## 2. 第五轮：把 E8M0 scale 并进 packed 行（方向 A，kernel 侧）

### 2.1 做法

一个 `(slot, head)` 的 8 个 scale 字节搬进 packed codes 所在的同一行，行宽
128 → **144**（128 codes + 8 scales + 8 pad）。kernel 侧只多一个 constexpr：

```python
offs_sblk_k = SCALE_COL_BASE + tl.arange(0, SCALE_TILE_K)
```

调用方把**同一个张量**既作 `k_scales` 传入、又以 `[..., :128]` 的行跨视图作
`k_packed` 传入，并置 `scale_col_base=128`。于是 `ks_row` 与 `k_row` 成为字面相同的
表达式，编译器只算一次行偏移，而不是再建一个指针瓦片。`SCALE_COL_BASE=0` 是默认值，
两张量布局的行为逐字节不变。

顺带修掉一处隐患：`mxfp4_decode_attention_fwd` 原本对四个 KV 张量各调一次
`.contiguous()`。kernel 的槽位/头偏移全部来自 `_extract_kv_strides`，真正的要求只是
最内维 stride 为 1，于是这四次调用被换成显式检查。原样留着的话，任何行跨视图入参都会
**每次 decode 静默复制整池**——探针第一版就撞上了，51 MB 的拷贝把 combined 测成
+73%，差点让我把方向 A 判死。

### 2.2 行宽清扫：padding 不是浪费，是异步拷贝的前提

同一探针 (`.probe_combined_row.py`)，只改行宽，输出与两张量版逐位相同：

| 行宽 | 构成 | 行基址对齐 | s1+s2 @100k | vs 两张量 |
|---|---|---|---|---|
| 128 | 两张量（基线）| 128 B | 0.1825 | — |
| **136** | 128 + 8 | **8 B** | 0.2015 | **+10.4%** |
| 144 | 128 + 8 + 8 | 16 B | 0.1533 | −16.0% |
| 160 | 128 + 8 + 24 | 32 B | 0.1577 | −13.6% |
| 256 | 128 + 8 + 120 | 128 B | 0.1508 | −17.4% |
| P7 | 行宽仍 128，scale 从行内 120..127 读（数值错，只看时间）| 128 B | 0.1387 | −24.0% |

两个读数定住了设计：

- **136 不但不赚还更慢。** 行 stride 136 不是 16 的倍数，奇偶行错位，凑不出 16 字节的
  异步拷贝。那 8 字节 padding 是**拿来换对齐的**，不是浪费。
- **160 / 256 并不比 144 好。** 既然 16 B 对齐就够，更宽只是白占内存。144 是最窄的可行宽度。
- **P7 给出无额外流量时的上限 −24%。** 真实 144 布局拿到 −16%，差的那 8 个点是
  行宽变宽带来的足迹代价。（此表口径与 §1.1 不同 —— 探针 harness 同步刷 L2、输出 fp32，
  绝对值整体偏高约 36%；同场相对值有效，跨表不可比。§2.4 是生产口径。）

### 2.3 机制：同步标量读变成异步流水拷贝

stage1 的 SASS 指令直方图（nvdisasm，只列有变化的）：

| 指令 | 两张量 | combined 144 | Δ |
|---|---|---|---|
| `LDG.E.U8`（同步标量全局字节读）| 8 | **0** | **−8** |
| `LDGSTS.E`（异步 global→shared 拷贝）| 0 | **4** | **+4** |
| `LDS.U8`（shared→reg 字节读）| 0 | **8** | **+8** |
| `LDGDEPBAR` | 4 | 8 | +4 |
| `LDGSTS.E.BYPASS.128`（packed 流，未变）| 16 | 16 | 0 |
| `UIADD3` | 40 | **20** | **−20** |
| `MOV` | 348 | 314 | −34 |
| `PRMT` | 185 | 158 | −27 |
| `LOP3.LUT` | 66 | 49 | −17 |
| 合计 | 2400 | 2304 | −96 |

这正是 §1.3 预测的那一步，且是逐条可见的：8 条**同步**全局字节读消失，换成 4 条
**异步**拷贝把 scale 预取进共享内存（多出的 1,024 B = 64 行 × 16 B 就是它的暂存缓冲，
`LDGDEPBAR` 翻倍是多出的那级流水），consumer 里只剩 8 条廉价的 `LDS.U8`。同时第二个
指针瓦片的地址算术整体塌掉——`UIADD3` 减半，uniform datapath 的 `MOV`/`PRMT`/`LOP3`
链各掉 34/27/17 条。

值得记一笔：**这就是 LeetCUDA 第 15 章归因给 warp specialization 的那一层收益**
（搬运从 consumer 指令流里移出、交给异步引擎），而这里是在 Triton 内靠改布局拿到的，
没写一行 CUDA。

### 2.4 结果（生产口径：`.bench_fp8_attention.py`，冷 L2，bs=1）

| seq | FP8 ms | 两张量 ms | **combined ms** | s1 两张量 → combined | **vs FP8** | MBU 两张量 → combined |
|---|---|---|---|---|---|---|
| 8,192 | 0.0312 | 0.0294 | **0.0257** | 0.0193 → 0.0157 (−18.7%) | **−17.5%** | 26.7% → 30.5% |
| 32,768 | 0.0687 | 0.0599 | **0.0480** | 0.0505 → 0.0394 (−22.0%) | **−30.2%** | 40.0% → 49.9% |
| 65,536 | 0.1211 | 0.0957 | **0.0786** | 0.0862 → 0.0697 (−19.1%) | **−35.1%** | 47.4% → 57.7% |
| 100,000 | 0.1804 | 0.1348 | **0.1064** | 0.1254 → 0.0980 (−21.9%) | **−41.0%** | 50.4% → 63.8% |
| 150,000 | 0.2666 | 0.1899 | **0.1491** | 0.1798 → 0.1403 (−22.0%) | **−44.1%** | 53.0% → 67.5% |

- stage1 稳定 **−19~22%**；stage2 不动（0.0077 → 0.0079~0.0083），符合预期。
- **@150k MXFP4 的 MBU 67.5% 已经追平 FP8 的 68.3%** —— 在搬运 1.81× 更少字节的前提下
  达到同样的带宽饱和度。这是本项目第一次做到。
- 对照 §1.2 的 P3 天花板（@100k −49.4%），本轮吃掉了其中约 **83%**。剩下的 ~8 个点是
  那 8 条 `LDS.U8` 与残余寻址，只有 WS / TMA 能再动。

占用率（stage1 grouped，`BLOCK_N=64`/4 warps）：

| | 两张量 | combined 144 |
|---|---|---|
| regs / spills | 206 / 0 | **208 / 0** |
| smem | 43,008 B | **44,032 B** |
| CTA/SM | 2（rf 2、smem 2）| **2（rf 2、smem 2）** |

SMEM 只涨 1,024 B，距 2 CTA/SM 的 51,200 B 门槛仍有 7,168 B 余量。

正确性：行宽 136/144/160/256 四种布局的输出与两张量版 `torch.equal` 全部为真；
`test_decode_attention_mxfp4_sm120.py` 4 passed + 6 subtests。

### 2.5 代价：KV cache 多占 5.9%

每 token 每层：两张量 2 × 4 × (128 + 8) = 1,088 B → combined 2 × 4 × 144 = **1,152 B**，
**+5.88%**。流量不变（padding 从不被读，§2.4 的 MiB 列两版相同），涨的是**容量**：同样的
`mem-fraction-static` 下可容 token 数降约 5.6%。换 −41% 的 decode 延迟，这笔交易是划算的，
但必须计进 `compute_cell_size`，不能当成免费。

两条能把这 5.9% 退回去的布局（都更大，留给后续）：

- **K/V 同行**：128 K codes + 128 V codes + 8 + 8 = **272 B = 17 × 16**，零 padding 且行基址
  天然 16 B 对齐。代价是 K/V 不再是两个独立张量，prefill 侧与"K/V 互不蹭 L2 行"的计量假设都要重做。
- **page 化**：一页 16 token 排成 16 × 128 codes + 16 × 8 scales = **2,176 B = 17 × 128**，
  零 padding 且 codes 行跨回到 128 B。需要 `page_size ≥ 16`，而这正是 §1.5 约束 ① 要求的
  TMA 前提——两件事可以一起做。

### 2.6 下一步

落地顺序：先把 144 B 行宽接到 pool / 量化 / prefill 三层（§1.7 的改动面已摸清），
拿到端到端 ITL；再按 §1.6 重评方向 B——届时残余空间只有 P3 口径的 ~8 个点加上孪生地板的
那一段，是否值得写 CUDA 需要重新算。

---

## 3. 第五轮落地：把 144 B 行接到 pool / 量化 / sizing 三层

### 3.1 落地形态：一次分配，四个行跨视图

关键简化：**不需要给 kernel 加任何参数。** `MXFP4KVCacheMethod.create_buffers` 每层每侧
只分配一个 `[slots, heads, 144]` 张量，四个 buffer 全是它的视图：

```python
k_rows = [torch.zeros((size, head_num, row), ...) for _ in range(layer_num)]
k_buffer       = [t[..., :packed_dim]                           for t in k_rows]
k_scale_buffer = [t[..., scale_offset : scale_offset + nblk]    for t in k_rows]
```

scale 视图的 `data_ptr` 自带 +128 偏移，而两个视图的 slot/head stride 完全相同，于是
`ks_row` 与 `k_row` 本来就是同一个表达式——§2 里那个 `SCALE_COL_BASE` constexpr 被撤掉了。
同口径对比两种写法（SASS 指令直方图）：

| | `LDG.E.U8` | `LDGSTS.E` | `LDS.U8` | `UIADD3` | 总指令 |
|---|---|---|---|---|---|
| 两张量 | 8 | 0 | 0 | 40 | 2400 |
| `scale_col_base=128` | 0 | 4 | 8 | 20 | 2304 |
| **四视图（落地形态）** | **0** | **4** | **8** | 21 | **2296** |

视图形态**还少 8 条指令**（列偏移里不用加常量），且不改 kernel API。行宽由
`_row_layout(head_dim)` 统一给出：`(packed_dim, scale_offset, row)`，两处都向上对齐到
`ROW_ALIGN = 16`（scale 的起始也要对齐，否则它自己那条拷贝同样凑不宽）。

### 3.2 四个"假设了连续性"的地方

视图不是连续张量，pool 里有四处隐含假设必须改，其中一处是**真 bug**：

| 位置 | 原逻辑 | 问题 | 处理 |
|---|---|---|---|
| `_init_data_ptrs_and_strides` | `data_strides = prod(shape[1:]) * itemsize` | 视图算出 512 而真实行跨是 576，slot-move 会**写错内存** | 改用 `x.stride(0) * itemsize`，对连续张量取值不变 |
| `_slot_move_pointer_buffers` | 列出 4 个 buffer | 视图的 stride 互相覆盖对方的字节 | 行拥有者存在时返回它们（2 项即覆盖全部，拷贝量也更少）|
| `get_kv_size_bytes` | 累加四个视图 | 少算 padding，低报真实占用 | 走行拥有者 |
| `_pd_registerable_tensors` / `_build_kv_buffer_descs` | 注册 `k_buffer`/`v_buffer` | **原本就漏注册 scale**（PD 传输会丢 scale）| 注册整行，顺带把这个旧缺口修好 |

为此 pool 多了一个 `kv_row_buffers` 字段：recipe 若用一次分配支撑多个视图就填它，否则为
`None`、全部退回原路径。

### 3.3 容量计账的漂移（启动 OOM 的真因）

第一次起服务直接 OOM，而日志里 `#tokens` 还是改造前的 **160,109**。原因不是分配器：

> `pool_configurator._compute_cell_size` 把 MXFP4 的字节公式**内联复制了一份**，
> 而 `MXFP4KVCacheMethod.compute_cell_size`（我改的那个）**只有测试在调用**。
> 于是 pool 按 144 B 分配、sizer 仍按 136 B 预算，静默超配约 0.15 GB。

两处已统一到 `_row_layout` 这一个来源，并在注释里写明它们曾经漂移过。修好后：

| | tokens | K size | V size | avail |
|---|---|---|---|---|
| 两张量 | 160,109 | 1.30 GB | 1.30 GB | 4.46 GB |
| combined 144 | **152,831** | 1.31 GB | 1.31 GB | 4.52 GB |

容量 −4.5%（比 §2.5 估的 5.6% 小，因为 PLAIN scratch 那一项不随行宽变）。
相对 FP8 的 105,071 token 仍是 **1.454×**（原 1.524×）。

### 3.4 端到端：同 binary A/B

`Qwen3.8-27B-NVFP4`，98k 上下文 / 96 输出 token / 流式 ITL p50，各 3 次重复：

| 布局 | tokens | ITL p50 | spread | decode | vs 两张量 |
|---|---|---|---|---|---|
| 两张量 | 160,109 | 24.58 ms | 0.01 | 40.69 tok/s | — |
| **combined 144** | 152,831 | **24.11 ms** | 0.06 | **41.48 tok/s** | **−1.9%** |

（两张量那一臂是把 pool 临时改回四张分配跑的，量完已还原，残留标记已核查为 0。
24.58 ms 与**前卷** §3.5 旧记录的 24.65 ms 吻合，说明两次 harness 可比。）

**核算对得上**：kernel 侧 @98k 每层省 0.1348 − 0.1064 = 0.0284 ms，乘 16 个
full-attention 层 = **预测 −0.454 ms/步**，实测 **−0.47 ms/步**。
kernel 收益 1:1 落到端到端，没有被调度或其它开销吃掉。

同时也量出了这条路线的上限：attention 现在只占一步的 ~5.9%（1.70 / 24.11 ms），
**剩下 94% 是权重 GEMM 底盘**。所以 −41% 的 attention 收益只能兑现成 −1.9% 的 ITL；
这不是实现问题，是分母问题。

### 3.5 回归

`test_decode_attention_mxfp4_sm120.py` + `test_mxfp4_kv_pool.py` +
`test_fp4_kv_cache_quant_method.py`：**29 passed + 8 subtests**。

decode kernel 的用例表原先只造四张连续张量，**完全没覆盖生产布局**（行跨视图在
shape 上看不出来）。已补 `_make_pools(…, pool_layout)`：默认 `combined`，并保留
`qwen_24_4_256_separate_pools` / `mha_hd64_p1_separate_pools` 两例继续覆盖旧形态。

写入侧 `quant_store_kv_mxfp4` 原本就全程走显式 stride，视图开箱可用；实测
fused 写入后的 codes 与 scales 与 eager codec 参考实现 `torch.equal` 为真，行内 padding
保持为零。

### 3.6 下一步

方向 A 结束。当前位置（cold L2 @100k）：0.1064 ms、MBU 63.8%、对 FP8 −41.0%；
@150k MBU 67.5% 已追平 FP8。

按 §1.6 重评方向 B 时要带上 §3.4 这个新事实：

- kernel 侧残余空间 = P3 口径的 ~8 个点 + 孪生地板那一段（MBU 63.8% → ~89%）；
- 但端到端杠杆是 **1 : 0.046**（attention 占 5.9%）。即使再把 attention 砍半，ITL 也只动 ~3%。

所以写 CUDA 做 TMA + WS 的理由不能再建立在"端到端提速"上。若要继续，值得的是
**page 化本身**（§2.5 那条零 padding 的 `[page, head, tok_in_page, …]` 布局）：它既退回
5.9% 的容量，又是 §1.5 约束 ① 的解除条件，收益与风险都比直接写 WS 清楚。

---

## 4. 定向：页表索引读是下一个同类杠杆（更正 §1.5 约束 ①）

目标回到 **kernel 本身**，所以 §3.6 那条"端到端杠杆只有 1:0.046"不构成停止的理由，
只是说明收益要按 kernel 口径记。本节先更正一处事实错误，再给下一轮的定价。

### 4.1 现状：`page_size = 1`，且不是 MXFP4 的限制

实测服务 `server_args` 为 `'page_size': 1`（服务端默认值），
`'enable_page_major_kv_layout': False`。`fp4_kv_cache_quant_method.py` 里**没有任何**
page_size 约束，decode kernel 也有 `PAGE_SIZE > 1` 分支、单测覆盖 page_size=16。

### 4.2 更正：调大 `--page-size` 对 kernel 是净负收益

§1.5 把约束 ① 写成「`page_size=1` 使 K/V 成为 gather，TMA 表达不了，需要
`page_size ≥ BLOCK_N`」。**这个因果链是错的。** 真实情况：

- 池子是 slot-major `[slots, heads, row]`，而 `kv_indices` **无论 page_size 多大都存
  token 级 slot id**（kernel 里 `page_id = kv_loc // PAGE_SIZE` 就是在从 token id 反推），
  所以 kernel 永远是逐 token gather —— 这与 page_size 无关。
- `_extract_kv_strides` 给出 `page_stride = slot_stride * page_size`、
  `tok_stride = slot_stride`，于是
  `page_id * page_stride + tok_in_p * tok_stride ≡ slot_stride * kv_loc`，
  **代数上完全等价**。

实测（head_dim 256 / combined 144 / 同输入）：

| | regs | smem | 总指令 | 输出 |
|---|---|---|---|---|
| `page_size=1` | 208 | 44,032 | 2,296 | — |
| `page_size=64` | 208 | 44,032 | **2,352（+56）** | 逐位相同 |

多出来的 56 条全是把 `kv_loc` 拆开再乘回去的开销（`SHF.R.S32.HI` +16、`IMAD` +16、
`LOP3.LUT` +8、`LEA.HI` +8、`IADD` +8）。所以**现成的 `PAGE_SIZE > 1` 分支在 slot-major
池子上是严格的悲观化**；调 `--page-size` 只改分配器（一页内 slot id 连续），对 kernel 而言
就是 §1.2 的 P6 效果 ≈ −1%。

### 4.3 P8：给页表机制定价 = stage1 的 −7%

> **§5.5 更正**：这个 −7% 是**冷 L2 上界**，不是可兑现预测。暖 L2（索引常驻，
> 即一个 decode step 的 16 层共享 `kv_indices` 的真实情形）下，page 粒度索引的收益
> 为零。落地后 kernel 冷口径 s1 −5.5~5.8%、服务端 ITL −0.17%。数字不改，读法改。

真正的杠杆与 §2 同类：把**逐 token 的索引读及其 gather 寻址**从内层循环拿掉 ，
改成 page 粒度（一个 tile 读 1 个 page id，地址对 `offs_n` 仿射）。探针做法：

```python
kv_loc = cur_batch_kv_start_idx + offs_n   # P8，替掉 tl.load(kv_indices + ...)
```

恒等页表下这与原式**完全等价**，已核：8k 序列输出 `torch.equal` 为真。
所以下表不是数值变化的假象：

| seq | 臂 | s1 现状 | s1 + P8 | Δ | 总计 vs FP8 | MBU |
|---|---|---|---|---|---|---|
| 100,000 | combined 144 | 0.0976 | **0.0912** | **−6.6%** | −40.4% → **−44.4%** | 63.2% → **67.7%** |
| 150,000 | combined 144 | 0.1396 | **0.1288** | **−7.7%** | −44.5% → **−48.1%** | 68.1% → **73.0%** |
| 100,000 | 两张量 | 0.1249 | 0.1202 | −3.8% | −28.3% | 52.5% |

@150k 的 MBU 会到 **73.0%**，首次**超过** FP8 的 68.5%。

### 4.4 修正后的杠杆排序（kernel 口径）

| 杠杆 | 实测定价 | 需要什么 | 风险 |
|---|---|---|---|
| **D. 页表索引改 page 粒度** | s1 **−7%** | `kv_indices` 按 page 存 + kernel 循环重排；要求 `page_size ≥ BLOCK_N`（=64），否则一个 tile 跨多页、仍需多次索引读 | 纯 Triton；动的是 backend 的 metadata 构造，不动数值 |
| E. 残余至孪生地板 | MBU 73% → ~89% | TMA + WS，写 CUDA | §1.5 约束 ②（寄存器池 vs 2 CTA/SM）仍未解除 |

**约束 ① 的正确形式**：TMA 需要的是"一个 tile 来自一段常量跨度的 2D box"。
slot-major 池子里 `[tok_in_page, row]` 的跨度本来就是常量（`slot_stride`），
所以缺的不是布局连续性，而是**一个 tile 只对应一个 page id** —— 即 `page_size ≥ BLOCK_N`
加上 page 粒度的索引。这恰好就是杠杆 D 的前提，两者仍然是一条链，但理由换了。

### 4.5 下一步

> **§5.5 更正**：本节把 P8 的 −6.6%/−7.7% 当成可兑现预测来排序杠杆，这是错的。
> 那是**冷 L2 上界**；暖 L2 下 page 粒度索引的收益为零。杠杆 D 已按本节落地，
> 实际服务端收益 −0.17% ITL。详见 §5.5、§5.7。

做杠杆 D。要落地的面比 §3 小但不在同一层：

- `kv_indices` 的生成侧（`triton_backend` 的 metadata）改成 page 粒度，或在 kernel 内按
  `offs_n // PAGE_SIZE` 只取每页首个；
- `PAGE_SIZE > 1` 分支要重写成"读 page id → 仿射"，而不是现在的"读 token id → 拆回 page"；
- 服务需以 `--page-size 64` 起（BLOCK_N=64），radix cache 的页粒度与命中率随之变化，
  这部分要单独量，不能只看 kernel。
---

## 5. 第六轮：杠杆 D 落地（page 粒度索引），以及一次口径更正

结论先行：**杠杆 D 做成了，但它的收益远小于 P8 给出的定价，原因是 P8（和本文档此前
所有 kernel 数字）只在冷 L2 口径下测过。** kernel 冷口径 s1 −5.5~5.8%、MBU 70.8%
首超 FP8；但索引常驻 L2 时收益归零，服务端实测只有 −0.17% ITL。

### 5.1 kernel 怎么改的

把内层循环里那条 `[BLOCK_N]` 的行偏移向量变成循环不变量：

```python
# 提到循环外
ramp_kp = tl.arange(0, BLOCK_N) * stride_kp_bs
# 循环内
page_slot = tl.load(PageTable + cur_batch * stride_pt_b + start_n // PAGE_SIZE)
slot0     = page_slot * PAGE_SIZE + start_n % PAGE_SIZE
k_row     = slot0 * stride_kp_bs + cur_kv_head * stride_kp_h + ramp_kp
```

每轮从「64 路 gather load + 64 路乘加」变成「1 次标量 load + 1 次标量乘加」。
走 `PAGED_INDEX: tl.constexpr` 守卫，token 级路径完整保留作为 fallback 与对照臂。

三个正确性条件的处理：

| 条件 | 处理 |
|---|---|
| tile 必须整体落在一页内 | `_MIN_BLOCK_KV` 32 → **64**（= BLOCK_N），并在 launcher 检查 `page_size % block_n == 0`；不满足则静默回退 |
| 页内 ps 个 slot 必须连续 | 由索引构造器自己的展开式 `token = entry*ps + pos%ps` 背书，另加单测交叉校验 |
| 末页是部分的 | 现有 `tok_mask = offs_n < split_kv_end` 直接够用，`offs_n` 仍是仿射的 |

`_MIN_BLOCK_KV` 32 → 64 **单独**测过（不加页表），三个长度一致地略正，不是代价：

| seq | comb s1 @32 | comb s1 @64 |
|---|---|---|
| 8,192 | 0.0159 | 0.0158 |
| 100,000 | 0.0975 | 0.0962 |
| 150,000 | 0.1381 | 0.1372 |

### 5.2 正确性：14 个用例 + 负对照

`.probe_paged_index.py`：同一个排列同时生成 token 级 `kv_indices` 与 page 级
`page_table`，两臂必须逐位相同。**乱序排列是重点**——恒等页表下，一个完全忽略页表的
实现也会"通过"。

覆盖：单页 / 部分末页 / 多页 / batch=4 ragged / gqa 16-2 / gqa 8-1 / MHA 回退，
× {恒等, 乱序} = 14 例，`max_abs_diff` 全为 0。

**负对照不可省。** 第一版我用 `page_table.roll(1)` 做对照，结果单页序列 roll 到自己，
对照失效（它正确地报了"路径未启用"）。改成 `(page_table + 1) % npages_pool` 后，
应启用的 13 例全部 `paged_engaged=True`，MHA 那例 `False`（符合预期——MHA kernel
没有页表参数，提示必须被忽略）。

单测里又发现一个我自己造的错：`_split` 按 `seq_lens`（= prefix+1，如 65/201）切扁平
池子，**请求起点没有页对齐**，再加上按 slot 过滤掉 slot 0 又把页结构错开一位，于是
rel_l2 3.6e-2。kernel 没问题（探针已证），是测试映射不自洽。改成按整页分配、整页跳过
page 0。这个坑值得记下来：**页表是按页边界读 `req_to_token` 建的，所以它隐含要求
每个请求的 slot 起点页对齐、且按整页分配**——`--page-size 64` 下分配器满足，但这是
一条真实约束，不是实现细节。

### 5.3 kernel 口径（冷 L2，`.bench_fp8_attention.py`，bs=1，24Q/4KV，hd 256）

两种映射各带自己的 FP8 参照，`tok` 与 `paged` 读同样的字节、同样的顺序：

| seq | map | 臂 | ms | s1 | s2 | MBU | vs FP8 |
|---|---|---|---|---|---|---|---|
| 100,000 | ident | FP8 | 0.1804 | – | – | 67.3% | – |
| 100,000 | ident | comb tok | 0.1052 | 0.0967 | 0.0082 | 64.6% | −41.7% |
| 100,000 | ident | **comb paged** | **0.0998** | **0.0911** | 0.0083 | **67.1%** | **−44.7%** |
| 100,000 | shuf | comb tok | 0.1061 | 0.0969 | 0.0082 | 64.1% | −41.3% |
| 100,000 | shuf | **comb paged** | **0.1015** | **0.0936** | 0.0083 | 66.0% | −43.8% |
| 150,000 | ident | FP8 | 0.2663 | – | – | 68.4% | – |
| 150,000 | ident | comb tok | 0.1480 | 0.1384 | 0.0083 | 68.0% | −44.4% |
| 150,000 | ident | **comb paged** | **0.1402** | **0.1306** | 0.0084 | **70.8%** | **−47.4%** |
| 150,000 | shuf | comb tok | 0.1484 | 0.1386 | 0.0083 | 67.8% | −44.4% |
| 150,000 | shuf | **comb paged** | **0.1412** | **0.1310** | 0.0084 | **70.3%** | **−47.1%** |

s1 增量 −5.5~5.8% @100k/150k；P8 的上界是 −6.6%/−7.7%，落点与"D ≤ P8"一致。
@150k MBU 70.3~70.8%，**首次超过 FP8 的 68.3%**。

@8,192 是 ident +3.1% / shuf −0.8%，即小序列下无收益。长上下文是目标场景，但要记明。

**一个被推翻的担心**：我原以为恒等映射会虚高 token 臂（它的索引流变成完美合并的顺序读）。
实测乱序对两臂几乎都没影响（150k tok: 0.1480 → 0.1484）。所以 §2 的数字没有被恒等
映射抬过，这一点可以放心。

占用率与指令（生产 combined 布局，两臂同口径）：

| 臂 | regs | spills | smem B | CTA/SM | 指令 |
|---|---|---|---|---|---|
| token | 219 | 0 | 44,032 | 2 | 2,272 |
| paged | 219 | 0 | 44,032 | **2** | **2,248** |

机制的 SASS 证据：`LDG.E` 13 → 5（索引 gather 消失）、`IADD.64` −17、`MOV` −28、
`PRMT` −8、`IMAD` −6；separate 布局下还能看到 `R2UR` +5（register → uniform
register，标量页基址进入 uniform 数据通路）、`ISETP.GE.S64.AND` 8 → 0、
`BRA`/`BSSY`/`BSYNC` 各 −2。

顺带解开 §3 留的疑问：`_GROUPED_REGS_PER_THREAD = 211` 对应 **separate** 布局，
§3 实测的 208 对应 combined，不是不一致。

### 5.4 服务端落地与实测

落地三处：

- `triton_backend`：`ForwardMetadata.mxfp4_page_table` 字段；`(max_bs, max_pages)`
  int32 buffer 一次分配、原地填充（cuda-graph replay 地址稳定）；
  `_fill_mxfp4_page_table` **直接复用 `build_trtllm_mha_page_table`**，不新写 kernel；
  eager 与 cuda-graph 两条路都填。
- 任何不支持的配置（DCP、`is_translating` 的 unified pool、页大小不整除 BLOCK_N）
  返回 None → kernel 回到 `kv_indices`。**`kv_indices` 始终是权威的**，页表只是提示。
- kill switch `SGLANG_DISABLE_MXFP4_PAGED_INDEX`，用来在**同一个 page_size** 下 A/B
  两种索引形式。这个开关是必需的：直接比 `--page-size 1` vs `64` 会把分配器、radix
  粒度、容量一起换掉，测出来的差值无法归因。

同 binary、同 `--page-size 64`、同 152,768 tokens，ITL p50（98k ctx，8 次重复，
两臂各跑两轮）：

| 索引形式 | ITL p50 | decode | 轮间复现 |
|---|---|---|---|
| token | 24.11 / 24.12 ms | 41.47 tok/s | 0.01 ms |
| paged | 24.08 / 24.07 ms | 41.53 tok/s | 0.01 ms |

Δ = **−0.04 ms（−0.17%）**。而 kernel 口径核算应为 0.0056 × 16 层 = **−0.090 ms**，
**只兑现 44%**。§3 的 combined 布局是 104%（预测 −0.454，实测 −0.47），所以这不是
harness 分辨率问题。

### 5.5 口径更正：冷 L2 高估了索引的价值

直接测，不猜。给 bench 加 `WARM=1`（不刷 L2），一个 decode step 的 16 层本来就共享
同一份 `kv_indices`，第 0 层之后它就在 L2 里：

| seq | map | tok s1 | paged s1 | Δ |
|---|---|---|---|---|
| 100,000 | ident | 0.0728 | 0.0732 | +0.5% |
| 100,000 | shuf | 0.0736 | 0.0740 | +0.5% |
| 150,000 | ident | 0.1166 | 0.1147 | −1.6% |
| 150,000 | shuf | 0.1195 | 0.1196 | +0.1% |

**暖口径下 paged 索引的收益为零。** 所以冷口径那 −5.8% 几乎全来自索引的 DRAM 流量，
而不是我在 §5.1 声称的"指针向量变循环不变量"的指令收益——SASS 的 −24 指令、
`LDG.E` 13→5 都是真的，但它们不产生可测时间。服务实测 −0.04 ms 正落在冷/暖之间，
因为真实循环里 16 层共享索引，但每层 106 MB 的 KV 流量会把它部分挤出 L2。

**这条更正的适用范围要划清楚。** 它只打击"收益来自索引"的那一类结论，不打击 §2：
暖口径下 separate → comb 在 100k 是 0.1060 → 0.0728（**−31%**），比冷口径的 −24%
**更大**。combined 布局的收益是 KV 主流的，不是索引的，所以它在两个口径下都成立。

对 §4.3 P8 那一节的更正：P8 的 −6.6%/−7.7% 应读作"冷 L2 上界"，不是可兑现预测。

### 5.6 这一轮的账

| 口径 | 结果 |
|---|---|
| kernel 冷 L2 | s1 −5.5~5.8% @100k/150k；总计 vs FP8 −44.7%/−47.4%；MBU 70.8% 首超 FP8 |
| kernel 暖 L2 | 无可测收益（−1.6% ~ +0.5%） |
| 服务端 ITL | −0.04 ms / −0.17%，可复现（轮间 0.01 ms） |
| 正确性 | 14 探针用例 + 3 单测用例逐位/过冻结阈值；负对照证实路径启用 |
| 资源 | regs 219 不变、0 spill、smem 不变、**2 CTA/SM 保住**、指令 −24 |
| 容量 | 152,831 → 152,768 tokens（页对齐，−0.04%） |

**留还是不留：留。** 理由不是收益大，而是它几乎没有成本——逐位精确、不支持的配置
静默回退、指令和寄存器都没变差、占用率没动、有 kill switch。但 headline 必须记成
**−0.17% ITL**，不是 P8 暗示的 −7%。

### 5.7 下一步

此前排的杠杆序列到这里需要重排，因为 §5.5 改变了判据：

1. **所有后续定价都必须在暖口径下复核。** 冷口径仍然有意义（冷启动、单请求长上下文），
   但不能再单独用它给优化定价。`.bench_fp8_attention.py` 现在两个口径都能跑。
2. 杠杆 E（TMA + WS 到孪生地板）的前提是"搬运层"还有空间。暖口径下 comb tok 在 100k
   已经是 **MBU 81.0%**、150k 77.0~79.2%，离孪生地板（~89%）只剩 10 个百分点左右，
   而不是冷口径看到的 68% → 89% 的 21 个点。**E 的可得收益需要按暖口径重估。**
3. 约束 ②（WS 的寄存器池 34,816 × 2 = 69,632 > 65,536 → 结构上 1 CTA/SM）仍未解除。
   在 E 的收益空间缩小到 10 个点的情况下，这条约束的权重相应上升。
4. 未解决的小问题：暖口径下 s2 稳定在 0.0080~0.0083，占 100k 总时间的 9.5%
   （冷口径只占 7.8%）。暖口径下 s2 的相对权重更高，值得单独看一眼。
