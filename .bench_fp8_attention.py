#!/usr/bin/env python3
"""Cold-L2 kernel comparison: flashinfer FP8 KV decode vs MXFP4 stage1+stage2.

The only FP8 attention number on record (0.130 ms/layer/step) came from a
server trace, i.e. real/warm conditions and a different context length, so it
cannot be put next to the cold-L2 microbenchmarks in section 4 of the doc.
This runs both paths through the identical harness: same shapes, same bs=1,
page_size=1, L2 evicted before every timed launch, CUDA events around the
launch only.

Per token per layer the two codecs move different amounts:
    FP8    2 * kv_heads * head_dim * 1 B            = 2048 B
    MXFP4  2 * kv_heads * (head_dim/2 + head_dim/32) = 1088 B
so MXFP4 has a 1.88x traffic advantage and the interesting question is how
much of it survives as wall time.

Also compares the two MXFP4 KV layouts. "separate" is the shipped one: codes
and E8M0 scales in two tensors, so the scale load builds its own pointer tile.
"combined" puts the 8 scale bytes in the same row as the 128 code bytes, which
makes the two loads share one slot/head address expression. The row is padded to
keep every row base 16-byte aligned -- at 136 B the odd rows are only 8-byte
aligned and the packed load loses its 128-bit vectorization. Neither variant
reads the padding, so the traffic and MBU columns are the same for both.

    python .bench_fp8_attention.py [seq_len ...]
"""

import os
import sys

import torch
import triton

from sglang.kernels.ops.attention import decode_attention_mxfp4_sm120 as mod
from sglang.srt.layers.quantization.kvfp4_tensor import MXFP4KVQuantizeUtil

Q_HEADS, KV_HEADS, HEAD_DIM = 24, 4, 256
BATCH, PAGE_SIZE = 1, 1
DEV = "cuda"
ITERS = 50
PEAK_READ_GBPS = 1686.6
SM_SCALE = 1.0 / (HEAD_DIM**0.5)
# Combined-layout row width: 128 code bytes + 8 scale bytes + 8 pad, the
# narrowest width that keeps every row base 16-byte aligned.
COMB_WIDTH = 144
# Page size for the page-granular index arm. Must be a multiple of the kernel's
# BLOCK_N so a tile cannot straddle two pages.
PS_TEST = 64


def build_mapping(seq_len, shuffle, gen):
    """One slot mapping in both index forms: token-granular and page-granular.

    Both describe the same mapping, so the two kernel arms read exactly the same
    bytes in the same order and only the index form differs. ``shuffle`` permutes
    whole pages, which is what a real allocator produces; the identity mapping is
    kept as a separate row because it flatters the token-level arm (its index
    stream becomes a perfectly coalesced sequential read).
    """
    npages = triton.cdiv(seq_len, PS_TEST)
    if shuffle:
        perm = torch.randperm(npages, device=DEV, generator=gen)
    else:
        perm = torch.arange(npages, device=DEV)
    toks = perm[:, None] * PS_TEST + torch.arange(PS_TEST, device=DEV)[None, :]
    return (
        toks.reshape(-1)[:seq_len].to(torch.int32).contiguous(),
        perm.view(1, npages).to(torch.int32).contiguous(),
        npages,
    )


# The index stream is the one input a real decode loop re-reads: all layers of
# one step share it, so after the first layer it is L2-resident. Thrashing before
# every iteration charges it to DRAM every time, which overstates any saving that
# comes from index traffic rather than from index *work*. WARM=1 measures the
# other regime.
WARM = os.environ.get("WARM") == "1"
# How to evict L2 between iterations. "write" (fill_) leaves up to an L2's worth
# of dirty lines that must be written back *while the measured kernel runs*,
# competing for DRAM bandwidth -- that is bandwidth pollution, not a cold cache.
# A real decode step evicts a layer's KV with other layers' *reads*, which is a
# clean eviction, so "read" is the faithful model. Kept selectable because every
# number in sections 1-5 of the profiling doc was taken with "write".
FLUSH = os.environ.get("FLUSH", "write")


def _evict(thrash, i):
    if WARM or FLUSH == "none":
        return
    if FLUSH == "read":
        _evict.sink = thrash.sum()
    else:
        thrash.fill_(i & 0xFF)


def cold_time(run, thrash, iters=ITERS):
    for _ in range(3):
        run()
    torch.cuda.synchronize()
    pairs = []
    for i in range(iters):
        _evict(thrash, i)
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        run()
        e.record()
        pairs.append((s, e))
    torch.cuda.synchronize()
    return sum(s.elapsed_time(e) for s, e in pairs) / iters


def split_stages(run, thrash, iters=20):
    """Per-stage device time, from the profiler rather than the event pairs."""
    from torch.profiler import ProfilerActivity, profile

    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for i in range(iters):
            _evict(thrash, i)
            run()
        torch.cuda.synchronize()
    t1 = t2 = 0.0
    for ev in prof.key_averages():
        if "stage1" in ev.key:
            t1 += ev.self_device_time_total / iters / 1e3
        elif "stage2" in ev.key:
            t2 += ev.self_device_time_total / iters / 1e3
    return t1, t2


def to_combined(packed, scales, width):
    """Codes at column 0, scales at column 128, in one ``[slots, heads, width]``.

    Returns the two row-strided views the kernel is handed. Both carry the same
    slot/head strides, so their row offsets are one shared expression; the scale
    view's base pointer already includes the column offset. Nothing is copied --
    the kernel reads both through explicit strides.
    """
    slots, heads, pk = packed.shape
    nblk = scales.shape[-1]
    comb = torch.zeros(slots, heads, width, dtype=torch.uint8, device=DEV)
    comb[..., :pk] = packed
    comb[..., pk : pk + nblk] = scales
    return comb[..., :pk], comb[..., pk : pk + nblk], comb


def main():
    seq_lens = [int(a) for a in sys.argv[1:]] or [32768, 100000, 150000]
    # Rounded up to a whole number of pages: the page-granular arm addresses the
    # last partial page's slots even though the mask discards them.
    max_slots = triton.cdiv(max(seq_lens), PS_TEST) * PS_TEST + PS_TEST
    gen = torch.Generator(device=DEV).manual_seed(1234)
    ref = torch.randn(
        (max_slots, KV_HEADS, HEAD_DIM),
        dtype=torch.bfloat16,
        device=DEV,
        generator=gen,
    )
    # Independent K and V, so neither path gets V reads for free out of the L2
    # lines its K reads already pulled in.
    ref_v = torch.randn(
        (max_slots, KV_HEADS, HEAD_DIM),
        dtype=torch.bfloat16,
        device=DEV,
        generator=gen,
    )
    kp, ks = MXFP4KVQuantizeUtil.batched_quantize(ref)
    vp, vs = MXFP4KVQuantizeUtil.batched_quantize(ref_v)
    k_fp8 = ref.to(torch.float8_e4m3fn)
    v_fp8 = ref_v.to(torch.float8_e4m3fn)
    del ref, ref_v
    torch.cuda.empty_cache()
    kcv, ksv, _kc = to_combined(kp, ks, COMB_WIDTH)
    vcv, vsv, _vc = to_combined(vp, vs, COMB_WIDTH)

    q = torch.randn(
        (BATCH, Q_HEADS, HEAD_DIM),
        dtype=torch.bfloat16,
        device=DEV,
        generator=torch.Generator(device=DEV).manual_seed(7),
    )
    thrash = torch.empty(192 * 2**20, dtype=torch.uint8, device=DEV)

    import flashinfer

    workspace = torch.empty(384 * 1024 * 1024, dtype=torch.uint8, device=DEV)
    wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(workspace, "NHD")

    nblk = HEAD_DIM // mod.MXFP4_BLOCK_SIZE
    num_warps, ctas_per_sm = mod.stage1_launch_config(Q_HEADS // KV_HEADS, 0)
    sms = torch.cuda.get_device_properties(0).multi_processor_count
    valid_h = min(mod._GROUPED_BLOCK_H, Q_HEADS // KV_HEADS)
    head_tiles = triton.cdiv(Q_HEADS, valid_h)
    max_splits = max(1, ctas_per_sm * sms // head_tiles)

    fp8_bytes_per_tok = 2 * KV_HEADS * HEAD_DIM
    mx_bytes_per_tok = 2 * KV_HEADS * (HEAD_DIM // 2 + nblk)
    print(f"flashinfer {flashinfer.__version__}  peak_read={PEAK_READ_GBPS:.0f} GB/s")
    print(
        f"per token/layer: FP8 {fp8_bytes_per_tok} B | MXFP4 "
        f"{mx_bytes_per_tok} B ({fp8_bytes_per_tok / mx_bytes_per_tok:.2f}x)"
    )

    print(
        f"\n{'seq':>8} | {'map':<5} | {'layout':<17} | {'ms':>8} {'s1':>7} {'s2':>7} "
        f"{'MiB':>7} {'GB/s':>6} {'MBU':>6} | {'vs FP8':>8}"
    )
    map_gen = torch.Generator(device=DEV).manual_seed(4242)
    for seq_len in seq_lens:
        kv_indptr = torch.tensor([0, seq_len], dtype=torch.int32, device=DEV)
        last_page = torch.ones(BATCH, dtype=torch.int32, device=DEV)
        qf = q.view(-1, Q_HEADS, HEAD_DIM).contiguous()

        o = torch.empty((BATCH, Q_HEADS, HEAD_DIM), dtype=torch.bfloat16, device=DEV)
        splits = max(1, min(max_splits, triton.cdiv(seq_len, mod._MIN_BLOCK_KV)))
        num_kv_splits = torch.tensor([splits], dtype=torch.int32, device=DEV)
        attn_logits = torch.empty(
            (BATCH, Q_HEADS, max_splits, HEAD_DIM), dtype=torch.float32, device=DEV
        )
        attn_lse = torch.empty(
            (BATCH, Q_HEADS, max_splits), dtype=torch.float32, device=DEV
        )

        for map_tag, shuffle in (("ident", False), ("shuf", True)):
            kv_indices, page_table, npages = build_mapping(seq_len, shuffle, map_gen)

            wrapper.begin_forward(
                kv_indptr,
                kv_indices,
                last_page,
                Q_HEADS,
                KV_HEADS,
                HEAD_DIM,
                PAGE_SIZE,
                data_type=torch.float8_e4m3fn,
                q_data_type=torch.bfloat16,
                non_blocking=False,
            )

            def run_fp8():
                wrapper.forward(
                    qf, (k_fp8, v_fp8), sm_scale=SM_SCALE, k_scale=1.0, v_scale=1.0
                )

            t_fp8 = cold_time(run_fp8, thrash)

            def make_run(k_p, v_p, k_s, v_s, page_size, pt):
                def run():
                    mod.mxfp4_decode_attention_fwd(
                        q,
                        k_p,
                        v_p,
                        k_s,
                        v_s,
                        o,
                        kv_indptr,
                        kv_indices,
                        SM_SCALE,
                        page_size=page_size,
                        max_kv_splits=max_splits,
                        attn_logits=attn_logits,
                        attn_lse=attn_lse,
                        num_kv_splits=num_kv_splits,
                        num_warps=num_warps,
                        page_table=pt,
                    )

                return run

            fp8_b = seq_len * fp8_bytes_per_tok

            def mx_bytes(index_entries):
                # MXFP4 also streams its index once per head tile and round-trips
                # the fp32 split partials through stage 2. The combined layout's
                # row padding is never read, so both layouts move the same bytes;
                # the paged arm's index is seq_len/PS_TEST times smaller, which
                # has to be in the total or its MBU reads high.
                return (
                    seq_len * mx_bytes_per_tok
                    + head_tiles * index_entries * 4
                    + 2 * BATCH * Q_HEADS * splits * (HEAD_DIM + 1) * 4
                )

            def row(label, t, nbytes, t1=None, t2=None):
                gbps = nbytes / (t / 1e3) / 1e9
                s1 = f"{t1:>7.4f}" if t1 is not None else f"{'-':>7}"
                s2 = f"{t2:>7.4f}" if t2 is not None else f"{'-':>7}"
                vs = "--" if t is t_fp8 else f"{t / t_fp8 - 1:+.1%}"
                print(
                    f"{seq_len:>8} | {map_tag:<5} | {label:<17} | {t:>8.4f} {s1} {s2} "
                    f"{nbytes / 2**20:>7.1f} {gbps:>6.0f} "
                    f"{gbps / PEAK_READ_GBPS * 100:>5.1f}% | {vs:>8}"
                )

            row("FP8 (flashinfer)", t_fp8, fp8_b)
            arms = [
                (f"MXFP4 comb tok", (kcv, vcv, ksv, vsv), 1, None, seq_len),
                (
                    f"MXFP4 comb paged",
                    (kcv, vcv, ksv, vsv),
                    PS_TEST,
                    page_table,
                    npages,
                ),
            ]
            if not shuffle:
                # Kept only on the identity mapping, as continuity with the
                # section 2 table; the decision here is between the two index
                # forms, both on the combined layout.
                arms.insert(0, ("MXFP4 separate", (kp, vp, ks, vs), 1, None, seq_len))
            for label, bufs, ps, pt, idx_entries in arms:
                run = make_run(*bufs, ps, pt)
                t = cold_time(run, thrash)
                t1, t2 = split_stages(run, thrash)
                row(label, t, mx_bytes(idx_entries), t1, t2)
        print()


if __name__ == "__main__":
    main()
