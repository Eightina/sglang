#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""真实 Qwen3.8 MXFP4 decode 离线回放：Torch OCP oracle vs MXFP4 PLAIN dump。

用途（qwen38_attention_survey_notes.md §11.5/§11.6，L2 验收）
----------------
用 `--kv-cache-dtype mxfp4 --attention-backend flashinfer --disable-cuda-graph`
采集一次真实请求的 prefill + decode dump 后，本脚本离线重建每个 full-attention
层每一步 decode 的：

  1. OCP MXFP4 KV cache（独立 reference codec：block-32 E2M1 + E8M0）；
  2. 纯 Torch decode attention（与 L1 FP8 golden 共享同一数学核心）；

并与 dump 中 gate 乘法前的 `model.layers.<L>.attn` 输出（FlashInfer 消费
PLAIN BF16 反量化 cache 的真实输出）逐层逐 token 对比。

采集方式
--------
```
python -m sglang.launch_server \
  --model-path /sgl-workspace/models/Qwen3.8-27B-NVFP4 \
  --kv-cache-dtype mxfp4 --attention-backend flashinfer \
  --disable-cuda-graph --mem-fraction-static 0.80 \
  --context-length 32768 --port 30001 \
  --debug-tensor-dump-output-folder /tmp/sgl_mxfp4_dump \
  --debug-tensor-dump-layers 3
# 另发一条请求（prefill 一次 + decode N 步）后停服
python3 compare_mxfp4_decode.py --dump-dir /tmp/sgl_mxfp4_req \
  [--fp8-dump-dir /tmp/sgl_fp8_req]
```

判定阈值与单测硬上限一致：rel L2 <= 2e-2、cosine >= 0.999、norm ratio
[0.98, 1.02]。`--fp8-dump-dir` 提供同一请求的 FP8 基线 dump 时，额外输出
MXFP4 相对 FP8 的逐层 attention 输出余弦（codec 质量表征，不参与判定）。
"""

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from compare_fp8_decode import (  # noqa: E402
    COSINE_CAP,
    HD,
    NH,
    NORM_RATIO_RANGE,
    NKV,
    REL_L2_CAP,
    SCALING,
    load_ckpt,
    load_passes,
    module_tensor,
    positions_of,
    qkv_len_of,
    reconstruct_qkv,
)
from sglang.test.kits.attention_unittest.attention_methods.fp8_decode_attention import (  # noqa: E402
    decode_output_metrics,
    format_metrics,
)
from sglang.test.kits.attention_unittest.attention_methods.mxfp4_decode_attention import (  # noqa: E402
    mxfp4_quantize_reference,
    torch_mxfp4_radix_decode_reference,
)


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

    qn_w = (
        load_ckpt(f"model.language_model.layers.{layer}.self_attn.q_norm.weight")
        .to(device)
        .float()
    )
    kn_w = (
        load_ckpt(f"model.language_model.layers.{layer}.self_attn.k_norm.weight")
        .to(device)
        .float()
    )

    qkv_pre = pre_d[qkv_key].float()
    if qkv_pre.dim() == 3:
        qkv_pre = qkv_pre[0]
    q_pre, k_pre, v_pre = reconstruct_qkv(qkv_pre, pre_pos, device, qn_w, kn_w)

    k_cache_seqs = [k_pre]
    v_cache_seqs = [v_pre]
    results = []
    actuals = []

    for step, (dec_path, dec_d) in enumerate(decodes[:max_steps]):
        qkv_cur = dec_d[qkv_key].float()
        if qkv_cur.dim() == 3:
            qkv_cur = qkv_cur[0]
        pos_cur = positions_of(dec_d, device, [prompt_len + step])
        q_cur, k_cur, v_cur = reconstruct_qkv(qkv_cur, pos_cur, device, qn_w, kn_w)

        # 历史 + 当前 token 全部经 OCP reference codec 编码（连续 loc 即可：
        # 数值与物理布局无关，布局正确性由单测覆盖）。
        k_all = torch.cat(k_cache_seqs + [k_cur], dim=0)
        v_all = torch.cat(v_cache_seqs + [v_cur], dim=0)
        seq_len = k_all.shape[0]
        pos_q = int(pos_cur[0].item())

        k_packed, k_scales = mxfp4_quantize_reference(k_all)
        v_packed, v_scales = mxfp4_quantize_reference(v_all)
        req_to_token = torch.arange(seq_len, device=device).view(1, -1)

        ref = torch_mxfp4_radix_decode_reference(
            q_cur[0].float(),
            k_packed,
            v_packed,
            k_scales,
            v_scales,
            req_to_token,
            0,
            seq_len,
            scaling=SCALING,
            logical_dim=HD,
        )

        actual = dec_d[attn_key].float()
        if actual.dim() == 3:
            actual = actual[0]
        actual = actual.reshape(NH, HD)

        metrics = decode_output_metrics(actual, ref)
        results.append((step, pos_q, seq_len, metrics, dec_path))
        actuals.append(actual)

        k_cache_seqs.append(k_cur)
        v_cache_seqs.append(v_cur)

    return results, actuals


def load_attn_series(dump_dir, layer, max_steps):
    """加载另一个 dump（如 FP8 基线）的同层 attention 输出序列，用于质量对比。"""
    prefill, decodes = load_passes(dump_dir)
    _, pre_d = prefill
    attn_key = next((k for k in pre_d if k.endswith(f"layers.{layer}.attn")), None)
    if attn_key is None:
        sys.exit(f"对比 dump 缺少 layers.{layer}.attn: {dump_dir}")
    series = []
    for _, dec_d in decodes[:max_steps]:
        t = dec_d[attn_key].float()
        if t.dim() == 3:
            t = t[0]
        series.append(t.reshape(NH, HD))
    return series


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--dump-dir", default="/tmp/sgl_mxfp4_req")
    ap.add_argument("--layers", default="3", help="逗号分隔的 full-attention 层号")
    ap.add_argument("--max-steps", type=int, default=8, help="最多回放多少个 decode 步")
    ap.add_argument(
        "--fp8-dump-dir",
        default=None,
        help="同一请求的 FP8 基线 dump（可选，仅做 codec 质量表征）",
    )
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    layers = [int(x) for x in args.layers.split(",")]
    prefill, decodes = load_passes(args.dump_dir)
    print(f"prefill pass: {os.path.basename(prefill[0])}; decode steps: {len(decodes)}")
    if not decodes:
        sys.exit("未找到 decode pass；请用 max_new_tokens>1 发请求")

    all_ok = True
    for layer in layers:
        results, actuals = replay_layer(layer, prefill, decodes, device, args.max_steps)
        print(f"\n=== layer {layer} (OCP MXFP4, no checkpoint scale) ===")
        print(
            f"{'step':>4} {'pos':>5} {'seq':>5} {'rel_l2':>10} {'cos':>10} "
            f"{'norm_ratio':>10}  verdict"
        )
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

        if args.fp8_dump_dir:
            fp8_series = load_attn_series(args.fp8_dump_dir, layer, args.max_steps)
            if len(fp8_series) != len(results):
                print(
                    f"--- layer {layer}: FP8 baseline step count mismatch "
                    f"({len(fp8_series)} vs {len(results)}), skip quality report"
                )
            else:
                print(
                    f"--- layer {layer}: MXFP4 vs FP8 baseline "
                    "(codec quality, informational) ---"
                )
                for (step, _pos, _seq, _m, _p), actual, fp8_out in zip(
                    results, actuals, fp8_series
                ):
                    q = decode_output_metrics(actual, fp8_out)
                    print(
                        f"{step:>4} rel_l2={q['rel_l2']:.3e} cos={q['cosine']:.6f} "
                        f"norm_ratio={q['norm_ratio']:.5f}"
                    )
    print(
        f"\n[{'PASS' if all_ok else 'FAIL'}] 全部层/步满足 "
        f"rel_l2<={REL_L2_CAP}, cos>={COSINE_CAP}, norm_ratio in {NORM_RATIO_RANGE}"
    )
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
