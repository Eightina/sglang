#!/usr/bin/env python3
"""Cold-L2 sweep of the stage-2 split-loop pipeline depth, with an exactness check.

Stage 2 merges the per-split partials. Its cost tracks MAX_KV_SPLITS and
nothing else: the loop-carried e_max -> old_scale -> acc chain paid one load
latency per split because the loads sat inside a data-dependent `if`, which
blocked the pipeliner (num_stages had literally no effect). Predicating the
body instead makes num_stages work.

Two negative results worth not repeating, both reproducible by editing this
file: splitting the V dim over extra CTAs does nothing (24 -> 384 CTAs within
noise, worse below 64 dims/CTA) because it trades request size for CTA count
at constant bytes in flight; and num_stages beyond 4 regresses.

    python .sweep_cold_stage2.py [seq_len ...]
"""
import sys

import torch
import triton
from torch.profiler import ProfilerActivity, profile

from sglang.kernels.ops.attention import decode_attention_mxfp4_sm120 as mod
from sglang.srt.layers.quantization.kvfp4_tensor import MXFP4KVQuantizeUtil

Q_HEADS, KV_HEADS, HEAD_DIM = 24, 4, 256
BATCH, PAGE_SIZE = 1, 1
DEV = "cuda"
ITERS = 50
PEAK_READ_GBPS = 1686.6

STAGES = [1, 2, 3, 4, 6, 8]


def make_kv(num_slots):
    gen = torch.Generator(device=DEV).manual_seed(1234)
    nblk = HEAD_DIM // mod.MXFP4_BLOCK_SIZE
    k_packed = torch.empty(
        (num_slots, KV_HEADS, HEAD_DIM // 2), dtype=torch.uint8, device=DEV
    )
    v_packed = torch.empty_like(k_packed)
    k_scales = torch.empty((num_slots, KV_HEADS, nblk), dtype=torch.uint8, device=DEV)
    v_scales = torch.empty_like(k_scales)
    for start in range(0, num_slots, 8192):
        end = min(start + 8192, num_slots)
        for packed, scales in ((k_packed, k_scales), (v_packed, v_scales)):
            ref = torch.randn(
                (end - start, KV_HEADS, HEAD_DIM),
                dtype=torch.bfloat16, device=DEV, generator=gen,
            )
            p, s = MXFP4KVQuantizeUtil.batched_quantize(ref)
            packed[start:end], scales[start:end] = p, s
    return k_packed, v_packed, k_scales, v_scales


def main():
    seq_lens = [int(a) for a in sys.argv[1:]] or [100000]
    sms = torch.cuda.get_device_properties(0).multi_processor_count
    kv_group = Q_HEADS // KV_HEADS
    num_warps, ctas_per_sm = mod.stage1_launch_config(kv_group, 0)
    valid_h = min(mod._GROUPED_BLOCK_H, kv_group)
    head_tiles = triton.cdiv(Q_HEADS, valid_h)
    max_splits = max(1, ctas_per_sm * sms // head_tiles)

    kv = make_kv(max(seq_lens) + 16)
    k_packed, v_packed, k_scales, v_scales = kv
    q = torch.randn(
        (BATCH, Q_HEADS, HEAD_DIM), dtype=torch.bfloat16, device=DEV,
        generator=torch.Generator(device=DEV).manual_seed(7),
    )
    thrash = torch.empty(192 * 2**20, dtype=torch.uint8, device=DEV)
    print(f"grid=({BATCH}, {Q_HEADS}) = {BATCH * Q_HEADS} CTAs on {sms} SMs")

    for seq_len in seq_lens:
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
        # Bytes stage 2 must read: the fp32 partials plus their lse column.
        s2_bytes = BATCH * Q_HEADS * splits * (HEAD_DIM + 1) * 4

        print(f"\n=== seq_len={seq_len} splits={splits} cold L2 ===")
        print(f"{'stages':>7} {'fwd ms':>8} {'stage1':>8} {'stage2':>8} "
              f"{'s2 us/split':>12} {'s2 GB/s':>8} {'vs stages=2':>12}  exact")
        base_s2, ref = None, None
        for stages in STAGES:
            def run(stages=stages):
                mod.mxfp4_decode_attention_fwd(
                    q, k_packed, v_packed, k_scales, v_scales, o,
                    kv_indptr, kv_indices, 1.0 / (HEAD_DIM**0.5),
                    page_size=PAGE_SIZE, max_kv_splits=max_splits,
                    attn_logits=attn_logits, attn_lse=attn_lse,
                    num_kv_splits=num_kv_splits, block_n=mod._BLOCK_N,
                    num_warps=num_warps, stage2_num_stages=stages,
                )

            for _ in range(3):
                run()
            torch.cuda.synchronize()
            out = o.clone()

            pairs = []
            for i in range(ITERS):
                thrash.fill_(i & 0xFF)
                s = torch.cuda.Event(enable_timing=True)
                e = torch.cuda.Event(enable_timing=True)
                s.record()
                run()
                e.record()
                pairs.append((s, e))
            torch.cuda.synchronize()
            total = sum(s.elapsed_time(e) for s, e in pairs) / ITERS

            with profile(activities=[ProfilerActivity.CUDA]) as prof:
                for i in range(20):
                    thrash.fill_(i & 0xFF)
                    run()
                torch.cuda.synchronize()
            t1 = t2 = 0.0
            for ev in prof.key_averages():
                if "stage1" in ev.key:
                    t1 += ev.self_device_time_total / 20 / 1e3
                elif "stage2" in ev.key:
                    t2 += ev.self_device_time_total / 20 / 1e3

            if ref is None:
                ref = out
            exact = "yes" if torch.equal(out, ref) else "NO"
            if stages == 2:
                base_s2 = t2
            delta = f"{t2 / base_s2 - 1:+.1%}" if base_s2 else ""
            print(f"{stages:>7} {total:>8.4f} {t1:>8.4f} {t2:>8.4f} "
                  f"{t2 * 1e3 / splits:>12.2f} {s2_bytes / (t2 / 1e3) / 1e9:>8.0f} "
                  f"{delta:>12}  {exact}")


if __name__ == "__main__":
    main()
