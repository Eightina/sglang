#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""层 3 FA 前向逐步二分定位：fused norm/gate kernel -> RoPE -> attention kernel。

数据源：hf_golden_cache.pt 里层 3 的 q/k/v 投影输出（已验证与 sglang 逐位一致）。
每一步用 sglang 的真实组件跑，与手算参考对照，找到第一个出错的环节。
"""

import json

import torch
from safetensors.torch import safe_open

CACHE = "/sgl-workspace/sglang/hf_golden_cache.pt"
CKPT = "/sgl-workspace/models/Qwen3.8-27B-NVFP4"
PRE = "model.language_model.layers.3.self_attn"
SEQ = 5
NH, NKV, HD = 24, 4, 256


def load_ckpt(k):
    idx = json.load(open(f"{CKPT}/model.safetensors.index.json"))["weight_map"]
    with safe_open(f"{CKPT}/{idx[k]}", framework="pt", device="cpu") as f:
        return f.get_tensor(k)


def gnorm_ref(x, w):
    """HF 语义的 per-head Gemma RMSNorm: x * rsqrt(mean(x^2)+eps) * (1+w)"""
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6) * (w + 1)


def rope_ref(x, cos32, sin32):
    """neox partial rope: 每头前 64 维（32 对）"""
    x1, x2 = x[..., :32], x[..., 32:]
    return torch.cat([x1 * cos32 - x2 * sin32, x1 * sin32 + x2 * cos32], dim=-1)


def main():
    dev = "cuda"
    cache = torch.load("/sgl-workspace/sglang/hf_golden_cache.pt", weights_only=False)
    m = cache["modules"]

    # ---- 输入（两引擎已验证一致）：q_gate (5,12288) / k (5,1024) / v (5,1024) ----
    q_gate = m["model.layers.3.qkv_q"].reshape(SEQ, -1).to(dev)
    k_in = m["model.layers.3.qkv_k"].reshape(SEQ, -1).to(dev)
    v_in = m["model.layers.3.qkv_v"].reshape(SEQ, -1).to(dev)
    qn_w = load_ckpt(f"{PRE}.q_norm.weight").to(dev).float()
    kn_w = load_ckpt(f"{PRE}.k_norm.weight").to(dev).float()
    o_w = (load_ckpt(f"{PRE}.o_proj.weight").to(dev).float()
           * load_ckpt(f"{PRE}.o_proj.weight_scale").to(dev).float())

    # ---- 手算参考链 ----
    qh = q_gate.cpu().view(SEQ, NH, 2 * HD)
    gate_ref = qh[:, :, HD:].to(dev)
    q_norm_ref = gnorm_ref(qh[:, :, :HD].to(dev), qn_w)   # rope 前
    k_norm_ref = gnorm_ref(k_in.cpu().view(SEQ, NKV, HD).to(dev), kn_w)   # rope 前
    q_ref = q_norm_ref.clone()   # rope 后
    k_ref = k_norm_ref.clone()
    inv = (10000000.0 ** (-torch.arange(0, 64, 2, dtype=torch.float64) / 64)).float().to(dev)
    ang = torch.arange(SEQ, device=dev).float()[:, None] * inv[None, :]
    c32, s32 = ang.cos().unsqueeze(1), ang.sin().unsqueeze(1)
    q_ref[:, :, :64] = rope_ref(q_ref[:, :, :64], c32, s32)
    k_ref[:, :, :64] = rope_ref(k_ref[:, :, :64], c32, s32)
    att = (torch.einsum("thd,shd->hts", q_ref, k_ref.repeat_interleave(6, dim=1)) / 16.0).softmax(-1)
    attn_core_ref = torch.einsum("hts,shd->thd", att, v_in.cpu().view(SEQ, NKV, HD).to(dev).repeat_interleave(6, dim=1)).reshape(SEQ, -1)
    gated_ref = attn_core_ref * torch.sigmoid(gate_ref.reshape(SEQ, -1))
    o_ref = gated_ref @ o_w.T

    def rep(tag, a, b):
        cos = torch.nn.functional.cosine_similarity(a.reshape(-1), b.reshape(-1), dim=0).item()
        rel = ((a - b).norm() / (b.norm() + 1e-9)).item()
        print(f"{tag:<46} cos={cos:.5f} rel={rel:.4f}  |ref|={b.norm().item():.3f} |out|={a.norm().item():.3f}")
        return rel

    # ========== Step A: sglang fused qk-norm+gate triton kernel ==========
    print("== Step A: fused_qk_gemma_rmsnorm_with_gate (sglang triton kernel) ==")
    from sglang.srt.models.utils import fused_qk_gemma_rmsnorm_with_gate
    q_a, k_a, gate_a = fused_qk_gemma_rmsnorm_with_gate(
        q_gate, k_in, qn_w, kn_w, 1e-6, HD, NH)
    q_a2 = q_a.reshape(SEQ, NH, HD)
    rep("A1 q_normed (rope 前)", q_a2, q_norm_ref)
    rep("A2 k_normed (rope 前)", k_a.reshape(SEQ, NKV, HD), k_norm_ref)
    rep("A3 gate", gate_a.reshape(SEQ, NH, HD), gate_ref)

    # ========== Step B: sglang MRotaryEmbedding ==========
    print("\n== Step B: MRotaryEmbedding (sglang) ==")
    from sglang.srt.layers.rotary_embedding import get_rope
    rope = get_rope(
        head_size=HD, rotary_dim=HD, max_position=262144,
        rope_scaling={"mrope_interleaved": True, "mrope_section": [11, 11, 10],
                      "partial_rotary_factor": 0.25, "rope_theta": 10000000,
                      "rope_type": "default"},
        base=10000000, partial_rotary_factor=0.25, is_neox_style=True,
        dtype=torch.bfloat16)
    rope = rope.to(dev)
    positions = torch.arange(SEQ, device=dev)
    q_b, k_b = rope(positions, q_a2.reshape(SEQ, -1).to(torch.bfloat16),
                    k_a.reshape(SEQ, -1).to(torch.bfloat16))
    q_b = q_b.float().reshape(SEQ, NH, HD)
    rep("B1 q after rope", q_b, q_ref)
    rep("B2 k after rope", k_b.float().reshape(SEQ, NKV, HD), k_ref)

    # ========== Step C: flashinfer 单序列 prefill kernel ==========
    print("\n== Step C: flashinfer single_prefill_with_kv_cache ==")
    import flashinfer
    q_c = q_b.to(torch.bfloat16).transpose(0, 1)[None]   # (1, NH, S, D)
    k_c = k_b.to(torch.bfloat16).transpose(0, 1)[None]
    v_c = v_in.cpu().view(SEQ, NKV, HD).to(dev).to(torch.bfloat16).transpose(0, 1)[None]
    o_c = flashinfer.single_prefill_with_kv_cache(q_c, k_c, v_c, causal=True)
    attn_core_c = o_c[0].transpose(0, 1).float().reshape(SEQ, -1)
    rep("C1 attention core (kernel vs 参考)", attn_core_c, attn_core_ref)

    # ========== Step D: 手算参考 gated -> o_proj，对照 HF golden o_proj ==========
    print("\n== Step D: 参考链闭合性（应与 HF o_proj 一致）==")
    hf_cache_o = m["model.layers.3.o_proj"].reshape(SEQ, -1).to(dev)
    rep("D1 o_ref vs HF o_proj", o_ref, hf_cache_o)


if __name__ == "__main__":
    main()
