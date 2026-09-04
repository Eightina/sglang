# 简历项目条目

> 简历正文见下；完整梳理与数据出处见 [5_qwen38_mxfp4_star_project.md](5_qwen38_mxfp4_star_project.md)。

---

MXFP4 KV cache 支持与 Triton native decode kernel（SGLang / Qwen3.8-27B / 单卡 32GB）

* 为在固定显存下扩展上下文，把 Qwen3.8-27B（64 层中 16 层全注意力，GQA、head_dim 256）的 FP8 KV cache 换成 OCP MXFP4，32→17 KiB/token；框架与生态均无可用 FP4 KV 实现，codec 与 decode kernel 全自研
* 前置修复基线 bug：bf16 KV 下输出渐进崩坏，逐层 dump 对 HF golden 定位到首个全注意力层，层内二分并反解 P·V 得行和恒为 v_scale——读路径无条件用 checkpoint KV descale 而写路径按存储 dtype 豁免；条件化后贪心输出与 HF 逐 token 一致
* 按 L0 FlashInfer FP8 → L1 torch FP8 golden → L2 MXFP4 codec + PLAIN → L3 Triton native 分层差分，每层只换一件事以分离 attention 实现、量化格式与 kernel 三类误差；codec 严格按 OCP MX 实现（per-head block-32、fp32 中间仅一次 RNE），与独立 oracle 双平台 bit-exact
* 由 PLAIN（读时整层反量化）的带宽倒贴与禁 CUDA graph 定出形态：decode inline dequant + 独立融合量化写 kernel；Triton split-KV kernel 经 grouped 消 GQA 6× 读放大、kv_splits 8→32 解 latency-bound、int32 位构造替 exp2 链，长文 TPOT 47.5→25.9ms
* 结果容量 55k→84k tokens（1.52×）、TPOT 25.9 vs FP8 基线 23.8ms 且 CUDA graph 可用；roofline 归因权重占每 step 读取 97%、KV 差异仅 0.14% TPOT，判定 kernel 优化收敛；PPL 4.08→16.86、HumanEval −18.3pp 经对照实验归因 E2M1 格式固有，定位为容量优先场景
