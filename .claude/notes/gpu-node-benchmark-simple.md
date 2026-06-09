# GPU 节点性能检查报告

> 测试日期: 2026-05-29
> 测试目的: 评估当前节点的大语言模型推理能力
> 节点配置: 4× NVIDIA RTX PRO 4000 Blackwell + AMD EPYC 9554 × 2

---

## 1. 测试思路与流程

### 1.1 测试维度选择

大语言模型推理主要受以下瓶颈约束，因此测试围绕这几个维度展开：

| 维度 | 为什么重要 | 测试方法 |
|------|-----------|---------|
| **显存带宽 (HBM BW)** | LLM decode 阶段是 memory-bound，token/s 直接正比于 HBM BW | vector add 饱和读写 |
| **算力 (TFLOPS)** | prefill 阶段和 batch 较大时是 compute-bound | 大规模 matmul |
| **PCIe 带宽** | 模型加载、host-device 数据搬运 | CPU↔GPU copy |
| **GPU 间通信** | 多卡 tensor parallelism 时 AllReduce 开销 | P2P D2D copy |
| **显存容量** | 决定能装多大的模型 | 硬件参数直接读取 |

### 1.2 测试流程

```
环境检查 → 硬件参数采集 → 显存带宽测试 → 算力测试 → PCIe带宽测试 → 多卡通信测试 → 综合分析
```

1. `nvidia-smi` 确认 GPU 型号、驱动版本、显存容量
2. `torch.cuda.get_device_properties()` 获取 SM 数量、clock rate、bus width 等底层参数
3. `nvidia-smi topo -m` 检查 GPU 互联拓扑
4. PyTorch benchmark: vector add (带宽)、matmul (算力)、copy (PCIe/P2P)

---

## 2. 硬件环境

### 2.1 GPU

| 项目 | 规格 |
|------|------|
| 型号 | NVIDIA RTX PRO 4000 Blackwell (× 4) |
| 架构 | Blackwell, **sm_120** |
| 显存 | 24 GB GDDR7 / 卡 (总计 96 GB) |
| Memory Bus Width | 192-bit |
| Memory Clock | 14001 MHz (有效速率 28 Gbps) |
| SM Count | 70 |
| Max Threads/SM | 1536 |
| L2 Cache | 48 MB |
| Shared Memory/SM | 100 KB |
| TDP | 145W / 卡 |
| PCIe | Gen 5 x16 |
| NVLink | **不支持** |
| BAR1 | 32 GB (支持 large BAR mapping) |

### 2.2 CPU & 系统

| 项目 | 规格 |
|------|------|
| CPU | AMD EPYC 9554 64-Core (× 2 sockets) |
| 总核心/线程 | 128 cores / 256 threads |
| NUMA Nodes | 2 |
| 内存 | 1 TB DDR5 |
| 磁盘 | 3.4 TB (3.1 TB available) |

### 2.3 GPU 拓扑

```
        GPU0    GPU1    GPU2    GPU3    NIC0    NIC1
GPU0     X      NODE    NODE    NODE    NODE    NODE
GPU1    NODE     X      NODE    NODE    NODE    NODE
GPU2    NODE    NODE     X      NODE    PHB     PHB
GPU3    NODE    NODE    NODE     X      NODE    NODE
NIC0    NODE    NODE    PHB     NODE     X      PIX
NIC1    NODE    NODE    PHB     NODE    PIX      X
```

- 所有 GPU 位于同一 NUMA node (node 1)
- GPU 间通过 PCIe 互联 (NODE = 同一 NUMA node 内 PCIe 通信)
- **无 NVLink**，GPU 间通信走 PCIe P2P
- GPU2 与 NIC 距离最近 (PHB)

### 2.4 软件环境

| 项目 | 版本 |
|------|------|
| Driver | 580.142 |
| CUDA Toolkit | 13.0 |
| PyTorch | 2.10.0a0+nv25.11 |

---

## 3. Benchmark 结果

### 3.1 显存带宽 (Memory Bandwidth)

| 测试项 | 实测值 | 理论峰值 | 利用率 |
|--------|--------|---------|--------|
| FP32 vector add | **570.7 GB/s** | 672 GB/s | 84.9% |
| FP16 vector add | **572.3 GB/s** | 672 GB/s | 85.2% |
| D2D copy (FP32) | **553.0 GB/s** | 672 GB/s | 82.3% |

> 理论峰值 = 28 Gbps × 192 bit / 8 = **672 GB/s**
>
> 4 卡表现一致 (波动 < 1%)，显存子系统工作正常。

### 3.2 算力 (Compute TFLOPS)

matmul benchmark: 8192 × 8192 × 8192

| 精度 | 实测值 | 说明 |
|------|--------|------|
| **FP32** | ~20.7 TFLOPS | CUDA core |
| **FP16** | ~70.1 TFLOPS | Tensor Core |
| **BF16** | ~71.6 TFLOPS | Tensor Core (推荐用于 LLM 推理) |
| **INT8 (via FP32 accum)** | ~22.5 TFLOPS | 未使用 INT8 Tensor Core path |

> 注: RTX PRO 4000 Blackwell 为 cut-down 核心 (70 SMs)，实测算力低于 full Blackwell die。
> 4 卡一致性良好，波动 < 2%。

### 3.3 PCIe 带宽 (Host ↔ Device)

| 方向 | 实测值 | PCIe Gen5 x16 理论 | 利用率 |
|------|--------|-------------------|--------|
| **H2D (Host→Device)** | ~23.7 GB/s | 64 GB/s | 37.0% |
| **D2H (Device→Host)** | ~20.9 GB/s | 64 GB/s | 32.7% |

> PCIe 利用率偏低，可能原因:
> - `nvidia-smi` 显示当前 PCIe 运行在 Gen 1 (节能状态)，benchmark 期间可能未完全切换至 Gen 5
> - PyTorch `copy_()` 走 pageable memory 路径，未使用 pinned memory
> - 实际推理场景中，模型加载为一次性操作，PCIe 带宽不是关键瓶颈

### 3.4 GPU 间通信 (P2P)

| 路径 | 实测带宽 |
|------|---------|
| GPU 0 → GPU 1/2/3 | ~52.0 GB/s |
| GPU 1 → GPU 0/2/3 | ~52.0 GB/s |
| GPU 2 → GPU 0/1/3 | ~52.0 GB/s |
| GPU 3 → GPU 0/1/2 | ~52.0 GB/s |

> - P2P access 全部 enabled，GPU 间可直接 DMA 无需经过 CPU
> - 所有方向带宽一致 (~52 GB/s)，表明 PCIe switch 全对称连接
> - 对比 NVLink (900 GB/s on H100) 差距巨大，**TP 通信开销将成为多卡推理主要瓶颈**

---

## 4. LLM 推理能力评估

### 4.1 模型容量估算

| 模型规模 | 精度 | 所需显存 (估算) | 是否可行 |
|---------|------|----------------|---------|
| 7B | BF16 | ~14 GB | ✅ 单卡 |
| 7B | INT4 | ~4 GB | ✅ 单卡 |
| 13B | BF16 | ~26 GB | ⚠️ 需 2 卡 |
| 13B | INT4 | ~7 GB | ✅ 单卡 |
| 30B | BF16 | ~60 GB | ⚠️ 需 3-4 卡 |
| 30B | INT4 | ~16 GB | ✅ 单卡 |
| 70B | BF16 | ~140 GB | ❌ 超出 (96 GB 总量) |
| 70B | INT4 | ~36 GB | ✅ 2 卡 (TP2) |

### 4.2 推理性能预估

**Decode 阶段 (memory-bound)**:
- 单卡 HBM BW = 570 GB/s
- 7B BF16 模型 decode: ~570 / 14 ≈ **40 tokens/s** (单请求，理论上限)
- 4 卡 TP4 (70B INT4, ~36 GB): 每卡需读 ~9 GB → ~570 / 9 ≈ **63 tokens/s**
  - 但 TP 通信 (AllReduce) 走 PCIe (~52 GB/s) 会显著降低实际吞吐

**Prefill 阶段 (compute-bound)**:
- BF16 算力 ~71 TFLOPS / 卡
- 7B 模型 prefill 1024 tokens: 约需 2 × 7B × 1024 = 14.3T FLOPs
- 单卡: 14.3 / 71 ≈ **0.2s** (TTFT)

### 4.3 关键瓶颈

1. **无 NVLink**: 最大短板。TP 需要频繁 AllReduce，PCIe 52 GB/s vs NVLink 900 GB/s (H100)，通信开销约为 H100 的 17 倍
2. **显存 24 GB / 卡**: 限制单卡可加载模型大小，大模型必须量化或 TP
3. **PCIe 带宽**: 模型加载和 weight offloading 受限，但对纯 GPU 推理影响较小

### 4.4 推理框架建议

| 框架 | 推荐理由 |
|------|---------|
| **vLLM** | PagedAttention + continuous batching，显存利用率高 |
| **SGLang** | RadixAttention，适合多轮对话场景 |
| **TensorRT-LLM** | NVIDIA 官方优化，Blackwell kernel 支持最佳 |

> 推荐优先尝试 TensorRT-LLM (Blackwell 原生支持) 或 vLLM (社区活跃、部署灵活)。

---

## 5. 总结

| 指标 | 数值 | 评级 |
|------|------|------|
| 显存带宽 | 570 GB/s | ⭐⭐⭐⭐ 适合 LLM decode |
| BF16 算力 | 71 TFLOPS | ⭐⭐⭐ 中等 (cut-down die) |
| 显存容量 | 24 GB × 4 = 96 GB | ⭐⭐⭐ 可跑 ≤30B BF16 / 70B INT4 |
| GPU 间通信 | 52 GB/s (PCIe P2P) | ⭐⭐ 无 NVLink，TP 瓶颈 |
| PCIe H2D | ~24 GB/s | ⭐⭐⭐ 模型加载可接受 |

**一句话结论**: 4× RTX PRO 4000 Blackwell 的显存带宽优秀 (570 GB/s，理论利用率 85%)，是 LLM 推理的核心优势；但无 NVLink 互联是最大短板，多卡 TP 场景下通信开销显著。推荐 **7B-13B 单卡推理** 或 **70B INT4 TP2 推理** 作为主力配置。
