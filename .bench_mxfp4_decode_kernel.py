#!/usr/bin/env python3
"""Config sweep for the native MXFP4 grouped decode stage-1 kernel.

Mirrors the production launch (bs=1, GQA 24Q/4KV, head_dim 256, page_size 1,
CUDA-graph replay) and sweeps BLOCK_H / num_warps / BLOCK_N / kv_splits.

    python .bench_mxfp4_decode_kernel.py [seq_len ...]
"""
import sys

import torch
import triton

from sglang.kernels.ops.attention import decode_attention_mxfp4_sm120 as mod
from sglang.srt.layers.quantization.kvfp4_tensor import MXFP4KVQuantizeUtil

Q_HEADS, KV_HEADS, HEAD_DIM = 24, 4, 256
BATCH = 1
PAGE_SIZE = 1
WARMUP, ITERS = 10, 50
DEV = "cuda"

# (BLOCK_H, num_warps, BLOCK_N, max_kv_splits); first entry is production.
# The grid should land in a whole number of waves of the 170 SMs:
# head_tiles(=cdiv(24, min(BLOCK_H, 6))) * splits vs SMs * CTAs-per-SM. See
# .sweep_cold_mxfp4.py for the per-config residency, and note that this script
# times back-to-back graph replays, so a working set near the 96 MiB L2 is
# reported L2-warm rather than at the cold-L2 state production sees.
CONFIGS = [
    (16, 8, 64, 42),  # reference: warps=8, 1 CTA/SM, one wave
    (16, 4, 64, 80),
    (16, 4, 64, 84),
    (16, 4, 64, 85),  # 340 CTAs = 170 SMs x 2 CTA/SM
    (16, 4, 64, 88),
    (16, 4, 64, 96),
    (16, 2, 64, 85),
    (16, 2, 64, 170),
    (8, 4, 64, 85),
    (8, 4, 64, 84),
]


def make_kv(num_slots):
    """Quantize random bf16 KV through the production codec."""
    gen = torch.Generator(device=DEV).manual_seed(1234)
    k_packed = torch.empty(
        (num_slots, KV_HEADS, HEAD_DIM // 2), dtype=torch.uint8, device=DEV
    )
    v_packed = torch.empty_like(k_packed)
    nblk = HEAD_DIM // mod.MXFP4_BLOCK_SIZE
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


def bench(seq_len, block_h, num_warps, block_n, max_splits, kv, q, baseline=None):
    k_packed, v_packed, k_scales, v_scales = kv
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

    saved = mod._GROUPED_BLOCK_H
    mod._GROUPED_BLOCK_H = block_h

    def run():
        mod.mxfp4_decode_attention_fwd(
            q, k_packed, v_packed, k_scales, v_scales, o,
            kv_indptr, kv_indices, 1.0 / (HEAD_DIM**0.5),
            page_size=PAGE_SIZE, max_kv_splits=max_splits,
            attn_logits=attn_logits, attn_lse=attn_lse,
            num_kv_splits=num_kv_splits, block_n=block_n, num_warps=num_warps,
        )

    try:
        for _ in range(3):
            run()
        torch.cuda.synchronize()
        out = o.clone().float()

        graph = torch.cuda.CUDAGraph()
        pool = torch.cuda.graphs.graph_pool_handle()
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            with torch.cuda.graph(graph, pool=pool):
                run()
        torch.cuda.current_stream().wait_stream(stream)

        for _ in range(WARMUP):
            graph.replay()
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        stop = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(ITERS):
            graph.replay()
        stop.record()
        torch.cuda.synchronize()
        ms = start.elapsed_time(stop) / ITERS
    finally:
        mod._GROUPED_BLOCK_H = saved

    rel = ""
    if baseline is not None:
        denom = baseline.abs().max().clamp_min(1e-6)
        rel = f"{((out - baseline).abs().max() / denom).item():.2e}"
    return ms, out, rel


def main():
    seq_lens = [int(a) for a in sys.argv[1:]] or [65536, 100000, 150000]
    kv = make_kv(max(seq_lens) + 16)
    # One fixed q for every config, so rel_err compares math, not inputs.
    q = torch.randn(
        (BATCH, Q_HEADS, HEAD_DIM),
        dtype=torch.bfloat16,
        device=DEV,
        generator=torch.Generator(device=DEV).manual_seed(7),
    )
    print(f"GPU={torch.cuda.get_device_name()} SMs={torch.cuda.get_device_properties(0).multi_processor_count}")
    for seq_len in seq_lens:
        print(f"\n=== seq_len={seq_len} (bs=1, {Q_HEADS}Q/{KV_HEADS}KV, d={HEAD_DIM}) ===")
        print(f"{'BLOCK_H':>8} {'warps':>6} {'BLOCK_N':>8} {'splits':>7} "
              f"{'CTAs':>6} {'ms':>8} {'vs prod':>9}  rel_err")
        base_ms, baseline = None, None
        for block_h, warps, block_n, max_splits in CONFIGS:
            valid_h = min(block_h, Q_HEADS // KV_HEADS)
            ctas = triton.cdiv(Q_HEADS, valid_h) * min(
                max_splits, triton.cdiv(seq_len, mod._MIN_BLOCK_KV)
            )
            try:
                ms, out, rel = bench(
                    seq_len, block_h, warps, block_n, max_splits, kv, q, baseline
                )
            except Exception as exc:  # noqa: BLE001
                print(f"{block_h:>8} {warps:>6} {block_n:>8} {max_splits:>7} "
                      f"{ctas:>6} {'FAIL':>8}   {type(exc).__name__}: {exc}"[:150])
                continue
            if base_ms is None:
                base_ms, baseline, rel = ms, out, "-"
            delta = f"{100 * (ms - base_ms) / base_ms:+.1f}%"
            print(f"{block_h:>8} {warps:>6} {block_n:>8} {max_splits:>7} "
                  f"{ctas:>6} {ms:8.3f} {delta:>9}  {rel}")


if __name__ == "__main__":
    main()
