#!/usr/bin/env python3
"""Single-config driver for profiling the MXFP4 decode attention kernels.

Mirrors the production launch (bs=1, GQA 24Q/4KV, head_dim 256, page_size 1)
with plain launches rather than a CUDA graph, so that ncu sees one kernel
instance per launch.

    ncu --kernel-name regex:mxfp4 ... python .ncu_mxfp4_decode.py [seq_len]

Standalone it just reports wall time per launch and the analytic roofline
inputs, so the ncu-measured traffic can be checked against the model.
"""

import sys

import torch
import triton

from sglang.kernels.ops.attention import decode_attention_mxfp4_sm120 as mod
from sglang.srt.layers.quantization.kvfp4_tensor import MXFP4KVQuantizeUtil

Q_HEADS, KV_HEADS, HEAD_DIM = 24, 4, 256
BATCH, PAGE_SIZE = 1, 1
DEV = "cuda"


def make_kv(num_slots):
    gen = torch.Generator(device=DEV).manual_seed(1234)
    nblk = HEAD_DIM // mod.MXFP4_BLOCK_SIZE
    k_packed = torch.empty(
        (num_slots, KV_HEADS, HEAD_DIM // 2), dtype=torch.uint8, device=DEV
    )
    v_packed = torch.empty_like(k_packed)
    k_scales = torch.empty((num_slots, KV_HEADS, nblk), dtype=torch.uint8, device=DEV)
    v_scales = torch.empty_like(k_scales)
    chunk = 8192
    for start in range(0, num_slots, chunk):
        end = min(start + chunk, num_slots)
        for packed, scales in ((k_packed, k_scales), (v_packed, v_scales)):
            ref = torch.randn(
                (end - start, KV_HEADS, HEAD_DIM),
                dtype=torch.bfloat16,
                device=DEV,
                generator=gen,
            )
            p, s = MXFP4KVQuantizeUtil.batched_quantize(ref)
            packed[start:end], scales[start:end] = p, s
    return k_packed, v_packed, k_scales, v_scales


def analytic(seq_len, splits):
    """Useful FLOPs, issued FLOPs and compulsory bytes for one stage-1 pass."""
    valid_h = min(mod._GROUPED_BLOCK_H, Q_HEADS // KV_HEADS)
    head_tiles = triton.cdiv(Q_HEADS, valid_h)
    # QK^T and P@V each contract head_dim once per (query head, token).
    useful = 2 * 2 * Q_HEADS * HEAD_DIM * seq_len
    # The M dim of every tl.dot is padded from valid_h up to BLOCK_H.
    issued = useful * (head_tiles * mod._GROUPED_BLOCK_H) / Q_HEADS
    nblk = HEAD_DIM // mod.MXFP4_BLOCK_SIZE
    kv_bytes = 2 * seq_len * KV_HEADS * (HEAD_DIM // 2 + nblk)
    # Stage 1 writes one fp32 partial per (head, split) plus its lse.
    logit_bytes = BATCH * Q_HEADS * splits * (HEAD_DIM + 1) * 4
    return useful, issued, kv_bytes, logit_bytes


def main():
    seq_len = int(sys.argv[1]) if len(sys.argv) > 1 else 100000
    launches = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    num_warps, ctas_per_sm = mod.stage1_launch_config(0)
    sms = torch.cuda.get_device_properties(0).multi_processor_count
    valid_h = min(mod._GROUPED_BLOCK_H, Q_HEADS // KV_HEADS)
    head_tiles = triton.cdiv(Q_HEADS, valid_h)
    max_splits = max(1, ctas_per_sm * sms // head_tiles)

    kv = make_kv(seq_len + 16)
    k_packed, v_packed, k_scales, v_scales = kv
    q = torch.randn(
        (BATCH, Q_HEADS, HEAD_DIM),
        dtype=torch.bfloat16,
        device=DEV,
        generator=torch.Generator(device=DEV).manual_seed(7),
    )
    o = torch.empty((BATCH, Q_HEADS, HEAD_DIM), dtype=torch.bfloat16, device=DEV)
    kv_indptr = torch.tensor([0, seq_len], dtype=torch.int32, device=DEV)
    kv_indices = torch.arange(seq_len, dtype=torch.int32, device=DEV)
    splits = max(1, min(max_splits, triton.cdiv(seq_len, mod._MIN_BLOCK_KV)))
    num_kv_splits = torch.tensor([splits], dtype=torch.int32, device=DEV)
    attn_logits = torch.empty(
        (BATCH, Q_HEADS, max_splits, HEAD_DIM), dtype=torch.float32, device=DEV
    )
    attn_lse = torch.empty(
        (BATCH, Q_HEADS, max_splits), dtype=torch.float32, device=DEV
    )

    def run():
        mod.mxfp4_decode_attention_fwd(
            q,
            k_packed,
            v_packed,
            k_scales,
            v_scales,
            o,
            kv_indptr,
            kv_indices,
            1.0 / (HEAD_DIM**0.5),
            page_size=PAGE_SIZE,
            max_kv_splits=max_splits,
            attn_logits=attn_logits,
            attn_lse=attn_lse,
            num_kv_splits=num_kv_splits,
            block_n=mod._BLOCK_N,
            num_warps=num_warps,
        )

    for _ in range(3):
        run()
    torch.cuda.synchronize()
    for _ in range(launches):
        run()
    torch.cuda.synchronize()

    useful, issued, kv_bytes, logit_bytes = analytic(seq_len, splits)
    print(
        f"seq_len={seq_len} splits={splits} CTAs={head_tiles * splits} "
        f"warps={num_warps} ctas_per_sm={ctas_per_sm}"
    )
    print(
        f"analytic: useful={useful / 1e9:.3f} GFLOP issued={issued / 1e9:.3f} GFLOP "
        f"kv_read={kv_bytes / 2**20:.1f} MiB logits_rw={logit_bytes / 2**20:.2f} MiB "
        f"AI={useful / kv_bytes:.1f} FLOP/B"
    )


if __name__ == "__main__":
    main()
