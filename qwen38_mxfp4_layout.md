我来调研 triton backend 的扩展点和 scale 布局适配性。先看 triton backend 的 decode 路径、现有 kernel 的 buffer 访问模式，以及 MXFP8 先例的 scale 布局。


调研完成。结论先行：**当前的 flat scale 布局 `(slots, head_num, 8)` 适合 triton decode，不需要改**；MXFP8 那种 interleaved 布局是给 FA4 定制的，不要跟。以下是证据链和扩展 triton backend 的具体落点。

## 1. Triton decode kernel 的 buffer 契约：packed data 直接适配

[_extract_kv_strides](file:///sgl-workspace/sglang/python/sglang/kernels/ops/attention/decode_attention.py#L197-L210) 只要求 buffer 是 3-D `[max_slots, head_num, dim]`，提取的是**通用 stride**（slot/head/page/tok），kernel 内的地址数学完全 stride-based、dtype 无关（[_fwd_kernel_stage1 L309-L323](file:///sgl-workspace/sglang/python/sglang/kernels/ops/attention/decode_attention.py#L309-L323)，PAGE_SIZE>1 走 `page_id*page_stride + tok_in_p*tok_stride`）。

现有 MXFP4 data buffer `(size+page_size, head_num, 128)` uint8 满足同一契约：slot_stride=512B、head_stride=128B。且 pool 层已把 `size + page_size` 传给 `create_buffers`（[memory_pool.py L2012](file:///sgl-workspace/sglang/python/sglang/srt/mem_cache/memory_pool.py#L2011-L2017)），padding page 的坑不存在。新 kernel 里每 token 加载 128 连续字节（4 个 32B sector），BLOCK_N=64 的 tile 访存是规整的。

## 2. Scale 布局访问模式分析：flat 够用

Triton decode 的 grid 是 `(batch, q_head, kv_split)`，每个 program 锁定一个 `cur_kv_head`、按 token 循环。对 scale 的需求是：**每 token 8 个连续字节**（`loc*32 + kv_head*8` 处），token 间 stride 32B。

- 流量占比：scale 仅占 KV 读流量的 1/17（8B vs 128B/token/head），绝对量很小；
- 同 kv_group 的多个 q head program 读同一份 scale 字节，L2 会天然广播命中；
- kernel 内用法：load uint8 → `tl.exp2(s.to(f32) - 127)`。E8M0 是纯 2 的幂，乘法无舍入，与 oracle 对齐没有额外数值风险；
- 零初始化安全：uint8 0 = 2^-127（合法极小值），不是 MXFP8 注释里担心的 0xFF=NaN；未写入 slot 不会被 page table 引用。

## 3. 为什么不用 MXFP8 的 interleaved 布局

[MHATokenToKVPoolMXFP8](file:///sgl-workspace/sglang/python/sglang/srt/mem_cache/memory_pool.py#L3309-L3317) 的 5-D interleaved（`(num_pages, head, 32, page_size//32, sf_dim)`，仅 page_size==128）是 **FA4 BlockScaledBasicChunk atom 的硬件布局要求**，配合 block-scale MMA 指令。你的 triton kernel 走 `kv_indices` 逐 token gather，没有 page-atom 约束，interleave 只会让写 kernel 和 PLAIN dequant 变复杂，零收益。**flat 布局同时兼容现有 eager dequant（PLAIN prefill）和未来的 triton decode，保持不动。**

## 4. 扩展 triton backend 的改动落点

| # | 位置 | 改动 |
|---|---|---|
| 1 | 新 kernel（建议放 `kernels/ops/attention/` 新文件，勿改 `_fwd_kernel_stage1`） | MXFP4 decode：load 128B packed → nibble 解包（低 4 位=偶数索引，与 codec 一致）→ 乘 E8M0 scale → fp32 QK/softmax/PV；复用 `_extract_kv_strides`，scale 用同构 3-D stride |
| 2 | [kv_cache_hook.py L48-L65](file:///sgl-workspace/sglang/python/sglang/srt/arg_groups/kv_cache_hook.py#L48-L65) | 放开 `decode_backend != "flashinfer"` 拦截，允许 prefill=flashinfer + decode=triton（框架本身支持拆分后端，`attention_backends_of` 已返回二元组）；cuda-graph 拦截先保留，kernel 稳定后再放 |
| 3 | [access registry L930-L933](file:///sgl-workspace/sglang/python/sglang/srt/layers/quantization/fp4_kv_cache_quant_method.py#L930-L933) | 给 mxfp4 增加一条 decode access：`_native_fp4(_DECODE, {"triton"}, _MXFP4_SCALE, _TORCH_FP4)`（NATIVE kind = kernel 直读 packed+scale），prefill 的 PLAIN→flashinfer 不动 |
| 4 | [triton_backend.py forward_decode](file:///sgl-workspace/sglang/python/sglang/srt/layers/attention/triton_backend.py#L2062-L2089) | 检测 pool 为 quantized mxfp4 时改走 `get_raw_kv_buffer` + scale buffer，分发到新 kernel；`kv_cache_scales_valid` 对 mxfp4 已是 False（[L274-L277](file:///sgl-workspace/sglang/python/sglang/srt/layers/attention/triton_backend.py#L274-L277)），checkpoint FP8 descale 不会误入——§7 事故天然免疫 |
| 5 | 初期 fail-fast 收窄面 | lean attention、score_mod、DCP、sliding window 分支先对 mxfp4 拒绝，只走标准 split-KV 路径（`_decode_att_m_fwd` + `_decode_softmax_reducev_fwd` 的对应物） |

## 5. 后端组合决策（2026-09-03 定稿）：triton decode 只做 native，不消费 PLAIN

新 triton kernel 是 native mxfp4（kernel 直读 packed + E8M0 scale），不承担
PLAIN 读。由此锁定唯一生产组合：

```
prefill/extend = flashinfer（PLAIN→BF16，L2 已验收）
decode         = triton（native mxfp4，L3 新 kernel）
```

推论：

- extend/prefill batch 永远路由到 flashinfer，triton `forward_extend` 不会
  碰到 mxfp4 → 无需为 triton extend 补 PLAIN 差分对照；
- fail-fast（kv_cache_hook.py）强制上述配对：放开
  `(flashinfer, triton)` 与现状 `(flashinfer, flashinfer)`，拒绝整组 triton
  （triton prefill 消费 PLAIN 超出范围）与其他 decode 后端；
- decode access 保留两条：新增 `_native_fp4(_DECODE, {"triton"}, ...)`，
  同时保留 `_plain(_DECODE, {"flashinfer"}, ...)` 作验证路径——L3 验收时
  同服务切换 decode 后端即可做 triton native vs flashinfer PLAIN 的 A/B；
- speculative verify 走 extend 形态 forward，会落 flashinfer PLAIN，首期不
  纳入范围，hook 对 speculative 组合维持拒绝。

**建议的第一步不变**：先在 unittest 框架内写独立 triton dequant + decode kernel 对 L2 golden 差分（上表 #1），跑通后再动 #2-#4 的集成。scale 布局这个问题到此可以关闭：**保持现状**。

## 6. L3 性能调研（2026-09-04，nsys + micro-bench + roofline）

服务级 TPOT（bs≈1，RTX 5090）：

| 配置 | 短文 512/128 | 长文 4096/128 |
|---|---|---|
| A = FP8 flashinfer | 23.7ms | 23.8ms |
| B = mxfp4 PLAIN flashinfer | 152.7ms | 152.7ms |
| C = mxfp4 triton native（grouped + splits32） | 24.5ms | **25.7ms** |
| C 优化前（per-head grid + splits8） | 26.7ms | 47.5ms |

### 6.1 实施的两项优化

1. **grouped stage1 kernel**（`_mxfp4_grouped_decode_stage1_kernel`）：原 grid 按
   q head 展开，同一 kv_group 的 6 个 q head 各自重读 packed K/V（读放大 6×）。
   改为每 program 服务一个 kv head 的整个 query group，K/V 解包一次共享，
   `tl.dot`（bf16 tile，dequant 值 bf16 无损）。重新表征冻结：单测 grouped 阈值
   rel_l2 1.85e-3 / 集成 3.3e-3（p→bf16 + 累加顺序残差，与 stock/flashinfer 同级）。
2. **split 并行度 8→32**（triton_backend mxfp4 分支内提升 floor，
   `--triton-attention-num-kv-splits` 显式更大值优先）：bs=1 长文 grid 仅
   kv_heads×splits 个 CTA（4×8=32 vs 170 SM），kernel 是 latency-bound。
   nsys 实测：stage1 4096-seq 从 ~155µs（splits8）降至 **34.6µs**（splits32）。

### 6.2 SM120 feature 调研结论（tl.dot_scaled / FP4 MMA）

- triton 3.7.1 的 `tl.dot_scaled` 在 **sm_120 编译通过**，e2m1/e8m0 打包约定与
  池布局完全一致（低 nibble = 偶数元素，scale 沿归约维每 32 元素），隔离 probe
  中比手动解包快 34%（14.5 vs 21.9µs，QK-only）。
- 但在**完整 decode kernel 内反而慢 2.2×**（153.7 vs 68.9µs @4096/splits32）：
  rhs 列数小（BLOCK_H=16）、与 PV 手动 dot/softmax 混排、每次迭代 64 行小 tile，
  FP4 MMA 的固定开销吞掉收益。且 PV 无法用 dot_scaled（V 的 scale 沿输出维而非
  归约维，语义不匹配）。**保留为 `use_dot_scaled` 实验开关，默认关**。
- probe 脚本：`scripts/playground/probe_dot_scaled.py`；bench 脚本：
  `scripts/playground/bench_mxfp4_decode_kernel.py`；nsys profile：
  `/tmp/mxfp4_kernel_profile.nsys-rep`（容器无 GPU 计数器权限，
  `--gpu-metrics-devices` 不可用，仅 kernel 时间线）。

### 6.3 roofline 与后续空间判断

- 单层 kernel（4096 seq，splits32）：stage1 34.6µs + stage2 6.2µs = 41µs，
  KV 流量 4.5MB → 110 GB/s ≈ **6% roofline**（1.79 TB/s）。kernel 仍是
  latency/occupancy-bound，理论还有空间（更大 split、Persistent kernel、
  stage1/2 融合、TMA descriptor 预取等）。
- **但端到端已到收益边界**：16 层 × 41µs ≈ 656µs，仅占 TPOT 25.7ms 的
  **~2.6%**。其余 97% 为权重读取（21.5GB/step ≈ 12ms，NVFP4/FP8 混合权重，
  与 attention kernel 无关）、48 层 GDN 线性注意力与 o_proj/qkv_proj。
  A（FP8 flashinfer，CUDA kernel）与 C 的剩余 1.9ms 差距中 attention 只占
  ~0.7ms。
- **结论：L3 decode kernel 的端到端优化到此为止**。若未来要继续压缩 TPOT，
  方向在权重路径与 GDN 层，而非 attention kernel。PLAIN prefill 的 prefix
  命中路径（全池 BF16 物化，TTFT 不可控）可用 roadmap §11.7 允许的
  dequant-workspace 适配优化（独立后续项）。

### 6.4 顺带发现与修复

- filelock≥3.32 的 fork 审计会杀死 human_eval 的 multiprocessing 沙箱
  （`os.fork is unsafe`）；评测驱动改为 spawn 启动方法：
  `scripts/playground/run_humaneval_spawn.py`。
- sgl-eval CLI 默认 n-repeats=16，greedy 下纯浪费；评测命令显式
  `--n-repeats 1`。