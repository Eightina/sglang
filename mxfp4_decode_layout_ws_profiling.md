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
