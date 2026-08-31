#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""真实 Qwen3.8 FP8 decode 离线回放：Torch golden vs FlashInfer dump。

用途（qwen38_attention_survey_notes.md §11.2，L1 验收的第 C 层）
----------------
用 --debug-tensor-dump-output-folder 采集一次真实请求的 prefill + decode
dump 后，本脚本离线重建每个 full-attention 层每一步 decode 的：

  1. FP8 E4M3 KV cache（真实逐层 k_scale/v_scale，QDQ 语义与
     MHATokenToKVPool.set_kv_buffer 一致）；
  2. 纯 Torch decode attention（fp32 数学，K descale 折入 QK^T、
     V descale 折入 PV，与 FlashInfer kernel 的 scale 位置一致）；

并与 dump 中 gate 乘法前的 `model.layers.<L>.attn` 输出（FlashInfer
decode kernel 的真实输出）逐层逐 token 对比。

采集方式
--------
```
python -m sglang.launch_server \
  --model-path /sgl-workspace/models/Qwen3.8-27B-NVFP4 \
  --kv-cache-dtype fp8_e4m3 --mem-fraction-static 0.85 \
  --context-length 32768 --port 30000 \
  --debug-tensor-dump-output-folder /tmp/sgl_fp8_dump \
  --debug-tensor-dump-layers 3
# 另发一条请求（prefill 一次 + decode N 步）后 Ctrl-C 停服
python3 compare_fp8_decode.py --dump-dir /tmp/sgl_fp8_dump --layers 3
```

已知误差来源（均已计入阈值）：
- Q/K/V 重建用 fp32 参考链（gemma norm + partial RoPE）再 cast bf16，
  与 sglang fused kernel 的输出有 ~0.2% 相对差（locate_fa_bug.py 实测
  A/B 步 rel≈0.002 清白），该误差在 FP8 量化前进入 cache。

判定阈值与单测一致：rel L2 <= 2e-2、cosine >= 0.999、norm ratio
[0.98, 1.02]；本脚本同时打印实测值供收紧阈值用。
"""

import argparse
import glob
import json
import os
import sys

import torch
from safetensors.torch import safe_open

BASE = "/sgl-workspace/sglang"
MODEL_DIR = "/sgl-workspace/models/Qwen3.8-27B-NVFP4"

SEQ = None  # prefill 长度，从 dump 自动识别
NH, NKV, HD = 24, 4, 256
SCALING = HD ** -0.5  # 1/16，与 RadixAttention(scaling=head_dim**-0.5) 一致
ROPE_DIM = 64  # partial_rotary_factor 0.25 * 256
ROPE_PAIRS = 32
ROPE_THETA = 10000000.0

# 与 test_flashinfer_fp8_decode.py 的硬上限一致
REL_L2_CAP = 2e-2
COSINE_CAP = 0.999
NORM_RATIO_RANGE = (0.98, 1.02)

from sglang.test.kits.attention_unittest.attention_methods.fp8_decode_attention import (  # noqa: E402
    decode_output_metrics,
    fp8_cache_quantize_reference,
    format_metrics,
    torch_fp8_radix_decode_reference,
)


# ---------------------------------------------------------------- checkpoint
def load_ckpt(key):
    idx = json.load(open(f"{MODEL_DIR}/model.safetensors.index.json"))["weight_map"]
    with safe_open(f"{MODEL_DIR}/{idx[key]}", framework="pt", device="cpu") as f:
        return f.get_tensor(key)


def load_layer_scales(layer):
    """生产语义：0-dim float32 CPU tensor（见 kv_cache.py create_weights）。"""
    pre = f"model.language_model.layers.{layer}.self_attn"
    k = load_ckpt(f"{pre}.k_scale")
    v = load_ckpt(f"{pre}.v_scale")
    return (
        torch.nn.Parameter(torch.tensor(float(k), dtype=torch.float32)),
        torch.tensor(float(k), dtype=torch.float32).item(),
        torch.nn.Parameter(torch.tensor(float(v), dtype=torch.float32)),
        torch.tensor(float(v), dtype=torch.float32).item(),
    )


# ------------------------------------------------------- norm / RoPE 参考链
def gnorm_ref(x, w):
    """HF 语义 per-head Gemma RMSNorm: x * rsqrt(mean(x^2)+eps) * (1+w)。"""
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6) * (w + 1)


def build_rope_tables(positions, device):
    """neox partial RoPE：每头前 64 维（32 对），theta=1e7。

    纯文本下 mrope_interleaved 的 T=H=W 重排为恒等（compare_hf_sglang 已验证），
    与 locate_fa_bug.py 的参考链一致。
    """
    inv = (ROPE_THETA ** (-torch.arange(0, ROPE_DIM, 2, dtype=torch.float64) / ROPE_DIM)).float().to(device)
    ang = positions.float()[:, None] * inv[None, :]  # (pos, 32)
    return ang.cos(), ang.sin()


def rope_ref(x_rot, cos32, sin32):
    x1, x2 = x_rot[..., :ROPE_PAIRS], x_rot[..., ROPE_PAIRS:]
    return torch.cat([x1 * cos32 - x2 * sin32, x1 * sin32 + x2 * cos32], dim=-1)


def reconstruct_qkv(qkv_raw, positions, device, qn_w, kn_w):
    """从 qkv_proj 原始输出（pre-norm）重建 RoPE 后的 q/k/v（bf16）。

    qkv_raw: (T, 12288+1024+1024)；positions: (T,)。
    返回 q (T,24,256)、k (T,4,256)、v (T,4,256)，dtype bf16（与生产一致）。
    """
    qkv_raw = qkv_raw.to(device).float()
    q_gate = qkv_raw[:, : NH * 2 * HD].view(-1, NH, 2 * HD)
    k_raw = qkv_raw[:, NH * 2 * HD : NH * 2 * HD + NKV * HD].view(-1, NKV, HD)
    v_raw = qkv_raw[:, NH * 2 * HD + NKV * HD :].view(-1, NKV, HD)

    cos32, sin32 = build_rope_tables(positions, device)
    cos32 = cos32.unsqueeze(1)  # (T,1,32) broadcast over heads
    sin32 = sin32.unsqueeze(1)

    q = gnorm_ref(q_gate[..., :HD], qn_w)
    q = q.clone()
    q[..., :ROPE_DIM] = rope_ref(q[..., :ROPE_DIM], cos32, sin32)

    k = gnorm_ref(k_raw, kn_w)
    k = k.clone()
    k[..., :ROPE_DIM] = rope_ref(k[..., :ROPE_DIM], cos32, sin32)

    v = v_raw
    return q.to(torch.bfloat16), k.to(torch.bfloat16), v.to(torch.bfloat16)


# ---------------------------------------------------------------- dump 解析
def qkv_len_of(d):
    """该 pass 的 token 数（由 qkv_proj 第一维推断；无该模块返回 None）。"""
    for k in d:
        if k.endswith(".qkv_proj") and isinstance(d[k], torch.Tensor):
            return d[k].shape[0]
    return None


def load_passes(dump_dir):
    """返回 (prefill_pass, [decode_pass...])。

    prefill = 第一个 token 数 > 1 的 pass；decode = prefill 之后 token 数 == 1
    的 pass（warmup pass 位于 prefill 之前，自动排除）。"""
    files = sorted(
        glob.glob(os.path.join(dump_dir, "*", "Pass*.pt"))
        + glob.glob(os.path.join(dump_dir, "Pass*.pt")),
    )
    if not files:
        sys.exit(f"dump 目录下未找到 Pass*.pt: {dump_dir}")
    passes = []
    for p in files:
        d = torch.load(p, map_location="cpu", weights_only=False)
        passes.append((p, d))

    prefill = None
    decodes = []
    for p, d in passes:
        n = qkv_len_of(d)
        if n is None:
            continue
        if n > 1 and prefill is None:
            prefill = (p, d)
        elif prefill is not None and n == 1:
            decodes.append((p, d))
    if prefill is None:
        sys.exit("未找到 prefill pass（qkv_proj token 数 > 1）")
    return prefill, decodes


def module_tensor(d, name):
    """按候选键名取 dump 张量（兼容 model.layers.* / model.language_model.layers.*）。"""
    for cand in (f"model.layers.{name}", f"model.language_model.layers.{name}"):
        if cand in d:
            t = d[cand]
            return t[0] if isinstance(t, list) and len(t) == 1 else t
    return None


def positions_of(d, device, fallback):
    """优先取 dump 中的 forward_batch_info.positions；缺失则用推导值。

    单请求贪心回放中 prefill positions = 0..T-1，decode step t = prompt_len+t，
    与 radix 前缀首次请求的连续布局一致。"""
    key = next((k for k in d if k.endswith("forward_batch_info.positions")), None)
    if key is not None:
        return d[key].reshape(-1).to(device)
    return torch.tensor(fallback, dtype=torch.int64, device=device)


# ---------------------------------------------------------------- 主对比
def replay_layer(layer, prefill, decodes, device, max_steps):
    pre_path, pre_d = prefill
    prompt_len = qkv_len_of(pre_d)
    pre_pos = positions_of(pre_d, device, list(range(prompt_len)))
    qkv_key = next(
        (k for k in pre_d if k.endswith(f"layers.{layer}.qkv_proj")), None
    )
    if qkv_key is None:
        sys.exit(f"dump 缺少 layers.{layer}.qkv_proj")
    attn_key = next((k for k in pre_d if k.endswith(f"layers.{layer}.attn")), None)
    if attn_key is None:
        sys.exit(f"dump 缺少 layers.{layer}.attn（RadixAttention 输出，gate 前）")

    qn_w = load_ckpt(f"model.language_model.layers.{layer}.self_attn.q_norm.weight").to(device).float()
    kn_w = load_ckpt(f"model.language_model.layers.{layer}.self_attn.k_norm.weight").to(device).float()
    k_scale_param, k_scale, v_scale_param, v_scale = load_layer_scales(layer)

    # ---- prefill 重建（历史 KV 来源）----
    qkv_pre = pre_d[qkv_key].float()
    if qkv_pre.dim() == 3:
        qkv_pre = qkv_pre[0]
    q_pre, k_pre, v_pre = reconstruct_qkv(qkv_pre, pre_pos, device, qn_w, kn_w)

    k_cache_seqs = [k_pre]
    v_cache_seqs = [v_pre]
    results = []

    for step, (dec_path, dec_d) in enumerate(decodes[:max_steps]):
        # 当前 token 的 q/k/v（qkv_proj 段 + norm+rope at its position）
        qkv_cur = dec_d[qkv_key].float()
        if qkv_cur.dim() == 3:
            qkv_cur = qkv_cur[0]
        pos_cur = positions_of(dec_d, device, [prompt_len + step])
        q_cur, k_cur, v_cur = reconstruct_qkv(qkv_cur, pos_cur, device, qn_w, kn_w)

        # 历史序列 = prefill + 之前 decode 步 + 当前 token（decode 当前 token
        # 先写入 cache 再参与 attention，与生产 forward_decode 一致）
        k_all = torch.cat(k_cache_seqs + [k_cur], dim=0)
        v_all = torch.cat(v_cache_seqs + [v_cur], dim=0)
        seq_len = k_all.shape[0]
        pos_q = int(pos_cur[0].item())

        # FP8 QDQ cache + radix gather（连续 loc 即可：数值与物理布局无关，
        # 布局正确性由单测的 shuffled/interleaved 用例覆盖）
        k_fp8 = fp8_cache_quantize_reference(k_all, k_scale)
        v_fp8 = fp8_cache_quantize_reference(v_all, v_scale)
        req_to_token = torch.arange(seq_len, device=device).view(1, -1)

        ref = torch_fp8_radix_decode_reference(
            q_cur[0].float(),
            k_fp8,
            v_fp8,
            req_to_token,
            0,
            seq_len,
            scaling=SCALING,
            k_scale=k_scale,
            v_scale=v_scale,
        )

        actual = dec_d[attn_key].float()
        if actual.dim() == 3:
            actual = actual[0]
        actual = actual.reshape(NH, HD)

        metrics = decode_output_metrics(actual, ref)
        results.append((step, pos_q, seq_len, metrics, dec_path))

        # 当前 token 进入历史，供下一步使用
        k_cache_seqs.append(k_cur)
        v_cache_seqs.append(v_cur)

    return results, (k_scale, v_scale)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--dump-dir", default="/tmp/sgl_fp8_dump")
    ap.add_argument("--layers", default="3", help="逗号分隔的 full-attention 层号")
    ap.add_argument("--max-steps", type=int, default=8, help="最多回放多少个 decode 步")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    layers = [int(x) for x in args.layers.split(",")]
    prefill, decodes = load_passes(args.dump_dir)
    print(f"prefill pass: {os.path.basename(prefill[0])}; decode steps: {len(decodes)}")
    if not decodes:
        sys.exit("未找到 decode pass（input_ids 长度 == 1）；请用 max_new_tokens>1 发请求")

    all_ok = True
    for layer in layers:
        results, (k_scale, v_scale) = replay_layer(layer, prefill, decodes, device, args.max_steps)
        print(f"\n=== layer {layer} (k_scale={k_scale:.4g}, v_scale={v_scale:.4g}) ===")
        print(f"{'step':>4} {'pos':>5} {'seq':>5} {'rel_l2':>10} {'cos':>10} {'norm_ratio':>10}  verdict")
        for step, pos_q, seq_len, m, path in results:
            ok = (
                m["rel_l2"] <= REL_L2_CAP
                and m["cosine"] >= COSINE_CAP
                and NORM_RATIO_RANGE[0] <= m["norm_ratio"] <= NORM_RATIO_RANGE[1]
            )
            all_ok &= ok
            print(
                f"{step:>4} {pos_q:>5} {seq_len:>5} {m['rel_l2']:>10.3e} "
                f"{m['cosine']:>10.6f} {m['norm_ratio']:>10.5f}  "
                f"[{'PASS' if ok else 'FAIL'}]  {format_metrics(m)}"
            )

    print(f"\n[{'PASS' if all_ok else 'FAIL'}] 全部层/步满足 "
          f"rel_l2<={REL_L2_CAP}, cos>={COSINE_CAP}, norm_ratio in {NORM_RATIO_RANGE}")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
