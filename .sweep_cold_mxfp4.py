#!/usr/bin/env python3
"""Cold-L2 sweep of the launch knobs that control stage-1 occupancy.

The earlier warm-L2 sweep only ever tried BLOCK_N 64 and 128. The occupancy
table shows BLOCK_N 32 halves both the register and the shared-memory
footprint, which is the only lever that raises residency past 2 CTAs/SM, so
this re-sweeps (BLOCK_N, num_warps) with kv_splits derived from each config's
own wave size, and reports stage1/stage2 separately under a cold L2.

    python .sweep_cold_mxfp4.py [seq_len ...]
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

# (BLOCK_N, num_warps, splits_as_waves) -- splits is derived per config from
# its measured CTA residency, times this many waves.
CONFIGS = [
    (64, 4, 1),  # production
    (64, 4, 2),
    (32, 4, 1),
    (32, 4, 2),
    (32, 8, 1),
    (32, 8, 2),
    (32, 2, 1),
]


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


def residency(block_n, num_warps, kv, q):
    """Compile once at this config and return (regs, smem, ctas_per_sm)."""
    fn = mod._mxfp4_grouped_decode_stage1_kernel
    for dc in fn.device_caches.values():
        dc[0].clear()
    o = torch.empty((BATCH, Q_HEADS, HEAD_DIM), dtype=torch.bfloat16, device=DEV)
    indptr = torch.tensor([0, 4096], dtype=torch.int32, device=DEV)
    idx = torch.arange(4096, dtype=torch.int32, device=DEV)
    mod.mxfp4_decode_attention_fwd(
        q, kv[0], kv[1], kv[2], kv[3], o, indptr, idx, 1.0 / (HEAD_DIM**0.5),
        max_kv_splits=42, block_n=block_n, num_warps=num_warps,
    )
    torch.cuda.synchronize()
    p = torch.cuda.get_device_properties(0)
    for dc in fn.device_caches.values():
        for k in dc[0].values():
            by_rf = p.regs_per_multiprocessor // (k.n_regs * num_warps * 32)
            by_sm = p.shared_memory_per_multiprocessor // k.metadata.shared
            return k.n_regs, k.metadata.shared, max(1, min(by_rf, by_sm))
    raise RuntimeError("no compiled kernel found")


def run_config(seq_len, block_n, num_warps, waves, kv, q, thrash):
    k_packed, v_packed, k_scales, v_scales = kv
    sms = torch.cuda.get_device_properties(0).multi_processor_count
    valid_h = min(mod._GROUPED_BLOCK_H, Q_HEADS // KV_HEADS)
    head_tiles = triton.cdiv(Q_HEADS, valid_h)
    regs, smem, ctas_per_sm = residency(block_n, num_warps, kv, q)
    max_splits = max(1, waves * ctas_per_sm * sms // head_tiles)
    splits = max(1, min(max_splits, triton.cdiv(seq_len, mod._MIN_BLOCK_KV)))

    o = torch.empty((BATCH, Q_HEADS, HEAD_DIM), dtype=torch.bfloat16, device=DEV)
    kv_indptr = torch.tensor([0, seq_len], dtype=torch.int32, device=DEV)
    kv_indices = torch.arange(seq_len, dtype=torch.int32, device=DEV)
    num_kv_splits = torch.tensor([splits], dtype=torch.int32, device=DEV)
    attn_logits = torch.empty(
        (BATCH, Q_HEADS, max_splits, HEAD_DIM), dtype=torch.float32, device=DEV
    )
    attn_lse = torch.empty((BATCH, Q_HEADS, max_splits), dtype=torch.float32, device=DEV)

    def run():
        mod.mxfp4_decode_attention_fwd(
            q, k_packed, v_packed, k_scales, v_scales, o,
            kv_indptr, kv_indices, 1.0 / (HEAD_DIM**0.5),
            page_size=PAGE_SIZE, max_kv_splits=max_splits,
            attn_logits=attn_logits, attn_lse=attn_lse,
            num_kv_splits=num_kv_splits, block_n=block_n, num_warps=num_warps,
        )

    for _ in range(3):
        run()
    torch.cuda.synchronize()
    out = o.clone().float()

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

    return dict(
        splits=splits, ctas=head_tiles * splits, regs=regs, smem=smem,
        ctas_per_sm=ctas_per_sm, warps_per_sm=ctas_per_sm * num_warps,
        total=total, t1=t1, t2=t2, out=out,
    )


def main():
    seq_lens = [int(a) for a in sys.argv[1:]] or [100000]
    p = torch.cuda.get_device_properties(0)
    warp_cap = p.max_threads_per_multi_processor // 32
    nblk = HEAD_DIM // mod.MXFP4_BLOCK_SIZE
    kv = make_kv(max(seq_lens) + 16)
    q = torch.randn(
        (BATCH, Q_HEADS, HEAD_DIM), dtype=torch.bfloat16, device=DEV,
        generator=torch.Generator(device=DEV).manual_seed(7),
    )
    thrash = torch.empty(192 * 2**20, dtype=torch.uint8, device=DEV)

    for seq_len in seq_lens:
        kv_bytes = 2 * seq_len * KV_HEADS * (HEAD_DIM // 2 + nblk)
        print(f"\n=== seq_len={seq_len} cold L2, {ITERS} timed launches ===")
        print(f"{'BN':>4} {'w':>3} {'wv':>3} {'splits':>7} {'CTAs':>6} {'regs':>5} "
              f"{'smem':>6} {'CTA/SM':>7} {'occ':>6} {'fwd ms':>8} {'stage1':>8} "
              f"{'stage2':>8} {'MBU':>6} {'vs prod':>8}  rel")
        base, ref = None, None
        for block_n, num_warps, waves in CONFIGS:
            try:
                r = run_config(seq_len, block_n, num_warps, waves, kv, q, thrash)
            except Exception as exc:  # noqa: BLE001 - report and keep sweeping
                print(f"{block_n:>4} {num_warps:>3} {waves:>3}  FAILED: "
                      f"{type(exc).__name__}: {str(exc)[:60]}")
                continue
            if base is None:
                base, ref = r["total"], r["out"]
                rel = "ref"
            else:
                denom = ref.abs().max().clamp_min(1e-6)
                rel = f"{((r['out'] - ref).abs().max() / denom).item():.1e}"
            mbu = kv_bytes / (r["t1"] / 1e3) / 1e9 / PEAK_READ_GBPS * 100
            print(f"{block_n:>4} {num_warps:>3} {waves:>3} {r['splits']:>7} "
                  f"{r['ctas']:>6} {r['regs']:>5} {r['smem']:>6} "
                  f"{r['ctas_per_sm']:>7} {r['warps_per_sm'] / warp_cap * 100:>5.1f}% "
                  f"{r['total']:>8.4f} {r['t1']:>8.4f} {r['t2']:>8.4f} {mbu:>5.1f}% "
                  f"{r['total'] / base - 1:>+7.1%}  {rel}")


if __name__ == "__main__":
    main()
