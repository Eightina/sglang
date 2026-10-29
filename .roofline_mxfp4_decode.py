#!/usr/bin/env python3
"""Roofline / MFU / MBU analysis for the MXFP4 grouped decode attention kernel.

Three measurements, all at the production launch config:

1. Per-kernel split of ``mxfp4_decode_attention_fwd`` (stage 1 vs stage 2) via
   the torch profiler, since the existing sweep only reports their sum.
2. A memory-only twin: identical grid, tiling and global load pattern, but no
   LUT gather, no tl.dot and no softmax. This is the empirically achievable
   floor for this access pattern, which is what the roofline bound has to be
   compared against (ncu hardware counters need CAP_SYS_ADMIN, unavailable
   here).
3. The same twin plus the LUT gathers, to price the dequantization separately
   from the tensor-core work.

    python .roofline_mxfp4_decode.py [seq_len ...]
"""
import sys

import torch
import triton
import triton.language as tl
from torch.profiler import ProfilerActivity, profile

from sglang.kernels.ops.attention import decode_attention_mxfp4_sm120 as mod
from sglang.srt.layers.quantization.kvfp4_tensor import MXFP4KVQuantizeUtil

Q_HEADS, KV_HEADS, HEAD_DIM = 24, 4, 256
BATCH, PAGE_SIZE = 1, 1
DEV = "cuda"
WARMUP, ITERS = 10, 50

# Measured on this device rather than taken from the spec sheet, so that the
# utilization denominators are achievable numbers (see docstring of main).
PEAK_READ_GBPS = 1686.6
PEAK_BF16_TFLOPS = 241.8


@triton.jit
def _dequant_arith(code, sbyte):
    """E2M1 x E8M0 -> bf16 by assembling the bf16 bit pattern directly.

    Prices the arithmetic alternative to the 4096-entry LUT gather; it is not
    the production codec, which also has to define the degenerate E8M0 scales
    (0 and 0xFF) that this ignores.

    Normal E2M1 (exp != 0) is (1 + m/2) * 2^(e-1), and E8M0 sb means 2^(sb-127),
    so the bf16 exponent field is sb + e - 1 with m as the mantissa MSB. The
    exp == 0 subnormal is m * 0.5, i.e. exponent sb - 1 with zero mantissa.
    """
    s = (code >> 3) & 0x1
    e = (code >> 1) & 0x3
    m = code & 0x1
    sub = e == 0
    exp = tl.where(sub, sbyte - 1, sbyte + e - 1)
    mant = tl.where(sub, 0, m)
    bits = (s << 15) | ((exp & 0xFF) << 7) | (mant << 6)
    bits = tl.where((code & 0x7) == 0, s << 15, bits)
    return (bits & 0xFFFF).to(tl.uint16).to(tl.bfloat16, bitcast=True)


@triton.jit
def _twin_kernel(
    K_Packed,
    V_Packed,
    K_Scales,
    V_Scales,
    E2M1_LUT,
    LUT16,
    Out,
    kv_indptr,
    kv_indices,
    num_kv_splits,
    stride_kp_bs,
    stride_kp_h,
    stride_ks_bs,
    stride_ks_h,
    HALF_D: tl.constexpr,
    SCALE_BLOCKS: tl.constexpr,
    BLOCK_N: tl.constexpr,
    MIN_BLOCK_KV: tl.constexpr,
    MODE: tl.constexpr,
):
    """Load-only twin of the grouped stage-1 kernel.

    Mirrors its grid, its per-iteration tile shapes and every global load
    (kv_indices, packed K/V bytes, K/V scale bytes), then reduces whatever it
    read into a single value so nothing is dead-code eliminated.

    MODE selects how far past the loads it goes: 0 loads only, 1 the
    production 4096-entry fused LUT, 2 a 16-entry value LUT times a
    reconstructed power-of-two scale, 3 pure bit arithmetic.
    """
    cur_batch = tl.program_id(0).to(tl.int64)
    cur_kv_head = tl.program_id(1)
    split_kv_id = tl.program_id(2)

    offs_half = tl.arange(0, HALF_D)
    blk_idx = offs_half // (HALF_D // SCALE_BLOCKS)

    kv_start = tl.load(kv_indptr + cur_batch)
    seq_len = tl.load(kv_indptr + cur_batch + 1) - kv_start
    kv_splits = tl.load(num_kv_splits + cur_batch)
    per_split = tl.cdiv(tl.cdiv(seq_len, kv_splits), MIN_BLOCK_KV) * MIN_BLOCK_KV
    split_start = split_kv_id * per_split
    split_end = tl.minimum(split_start + per_split, seq_len)

    acc_i = tl.zeros([BLOCK_N, HALF_D], dtype=tl.int32)
    acc_f = tl.zeros([BLOCK_N, HALF_D], dtype=tl.float32)

    if split_kv_id < kv_splits:
        for start_n in range(split_start, split_end, BLOCK_N):
            offs_n = start_n + tl.arange(0, BLOCK_N)
            tok_mask = offs_n < split_end
            kv_loc = tl.load(
                kv_indices + kv_start + offs_n, mask=tok_mask, other=0
            )
            row = kv_loc * stride_kp_bs + cur_kv_head * stride_kp_h
            s_row = kv_loc * stride_ks_bs + cur_kv_head * stride_ks_h

            kbyte = tl.load(
                K_Packed + row[:, None] + offs_half[None, :],
                mask=tok_mask[:, None],
                other=0,
            )
            ksb = tl.load(
                K_Scales + s_row[:, None] + blk_idx[None, :],
                mask=tok_mask[:, None],
                other=0,
            )
            vbyte = tl.load(
                V_Packed + row[:, None] + offs_half[None, :],
                mask=tok_mask[:, None],
                other=0,
            )
            vsb = tl.load(
                V_Scales + s_row[:, None] + blk_idx[None, :],
                mask=tok_mask[:, None],
                other=0,
            )
            if MODE == 1:
                kbase = ksb.to(tl.int32) << 4
                vbase = vsb.to(tl.int32) << 4
                acc_f += tl.load(E2M1_LUT + (kbase | (kbyte & 0x0F)))
                acc_f += tl.load(E2M1_LUT + (kbase | ((kbyte >> 4) & 0x0F)))
                acc_f += tl.load(E2M1_LUT + (vbase | (vbyte & 0x0F)))
                acc_f += tl.load(E2M1_LUT + (vbase | ((vbyte >> 4) & 0x0F)))
            elif MODE == 2:
                ksc = ((ksb.to(tl.int32) << 7) & 0xFFFF).to(tl.uint16)
                vsc = ((vsb.to(tl.int32) << 7) & 0xFFFF).to(tl.uint16)
                ksc = ksc.to(tl.bfloat16, bitcast=True)
                vsc = vsc.to(tl.bfloat16, bitcast=True)
                acc_f += tl.load(LUT16 + (kbyte & 0x0F)) * ksc
                acc_f += tl.load(LUT16 + ((kbyte >> 4) & 0x0F)) * ksc
                acc_f += tl.load(LUT16 + (vbyte & 0x0F)) * vsc
                acc_f += tl.load(LUT16 + ((vbyte >> 4) & 0x0F)) * vsc
            elif MODE == 3:
                ki = kbyte.to(tl.int32)
                vi = vbyte.to(tl.int32)
                ks = ksb.to(tl.int32)
                vs = vsb.to(tl.int32)
                acc_f += _dequant_arith(ki & 0x0F, ks)
                acc_f += _dequant_arith((ki >> 4) & 0x0F, ks)
                acc_f += _dequant_arith(vi & 0x0F, vs)
                acc_f += _dequant_arith((vi >> 4) & 0x0F, vs)
            else:
                acc_i += (
                    kbyte.to(tl.int32)
                    + ksb.to(tl.int32)
                    + vbyte.to(tl.int32)
                    + vsb.to(tl.int32)
                )

    pid = (cur_batch * tl.num_programs(1) + cur_kv_head) * tl.num_programs(
        2
    ) + split_kv_id
    tl.store(Out + pid, tl.sum(tl.sum(acc_f, 1), 0) + tl.sum(tl.sum(acc_i, 1), 0))


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


def graph_time(run):
    """Time per launch under CUDA-graph replay, matching the existing sweep.

    Back-to-back replays leave the previous iteration's KV in L2, so for a
    working set near the 96 MiB L2 this reports an L2-warm number. Production
    never sees that reuse -- a full decode step streams ~20 GB of weights
    between two visits to the same layer -- so cold_time() below is the
    representative measurement and this one is kept only as the warm bound.
    """
    for _ in range(3):
        run()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        with torch.cuda.graph(graph, pool=torch.cuda.graphs.graph_pool_handle()):
            run()
    torch.cuda.current_stream().wait_stream(stream)
    for _ in range(WARMUP):
        graph.replay()
    torch.cuda.synchronize()
    s, e = (torch.cuda.Event(enable_timing=True) for _ in range(2))
    s.record()
    for _ in range(ITERS):
        graph.replay()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / ITERS


def cold_time(run, thrash, iters=ITERS):
    """Time per launch with L2 evicted first, which is the production state.

    The thrashing write stays outside the event window, so only the target
    launch is timed; all iterations are queued before syncing to keep the GPU
    saturated and the device-side timestamps gap-free.
    """
    for _ in range(3):
        run()
    torch.cuda.synchronize()
    pairs = []
    for i in range(iters):
        thrash.fill_(i & 0xFF)
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        run()
        e.record()
        pairs.append((s, e))
    torch.cuda.synchronize()
    return sum(s.elapsed_time(e) for s, e in pairs) / iters


def analyse(seq_len, kv, q):
    k_packed, v_packed, k_scales, v_scales = kv
    nblk = HEAD_DIM // mod.MXFP4_BLOCK_SIZE
    kv_group = Q_HEADS // KV_HEADS
    num_warps, ctas_per_sm = mod.stage1_launch_config(kv_group, 0)
    sms = torch.cuda.get_device_properties(0).multi_processor_count
    valid_h = min(mod._GROUPED_BLOCK_H, kv_group)
    head_tiles = triton.cdiv(Q_HEADS, valid_h)
    max_splits = max(1, ctas_per_sm * sms // head_tiles)
    splits = max(1, min(max_splits, triton.cdiv(seq_len, mod._MIN_BLOCK_KV)))

    o = torch.empty((BATCH, Q_HEADS, HEAD_DIM), dtype=torch.bfloat16, device=DEV)
    kv_indptr = torch.tensor([0, seq_len], dtype=torch.int32, device=DEV)
    kv_indices = torch.arange(seq_len, dtype=torch.int32, device=DEV)
    num_kv_splits = torch.tensor([splits], dtype=torch.int32, device=DEV)
    attn_logits = torch.empty(
        (BATCH, Q_HEADS, max_splits, HEAD_DIM), dtype=torch.float32, device=DEV
    )
    attn_lse = torch.empty(
        (BATCH, Q_HEADS, max_splits), dtype=torch.float32, device=DEV
    )

    def run_full():
        mod.mxfp4_decode_attention_fwd(
            q, k_packed, v_packed, k_scales, v_scales, o,
            kv_indptr, kv_indices, 1.0 / (HEAD_DIM**0.5),
            page_size=PAGE_SIZE, max_kv_splits=max_splits,
            attn_logits=attn_logits, attn_lse=attn_lse,
            num_kv_splits=num_kv_splits, block_n=mod._BLOCK_N,
            num_warps=num_warps,
        )

    twin_out = torch.empty(
        BATCH * KV_HEADS * splits, dtype=torch.float32, device=DEV
    )

    def make_twin(mode):
        lut = mod.e2m1_scale_lut(DEV, torch.bfloat16)
        # Unscaled E2M1 values, i.e. the scale==127 row of the fused table.
        lut16 = lut[127 * 16 : 127 * 16 + 16].contiguous()

        def run():
            _twin_kernel[(BATCH, KV_HEADS, splits)](
                k_packed, v_packed, k_scales, v_scales, lut, lut16, twin_out,
                kv_indptr, kv_indices, num_kv_splits,
                k_packed.stride(0), k_packed.stride(1),
                k_scales.stride(0), k_scales.stride(1),
                HALF_D=HEAD_DIM // 2, SCALE_BLOCKS=nblk,
                BLOCK_N=mod._BLOCK_N, MIN_BLOCK_KV=mod._MIN_BLOCK_KV,
                MODE=mode, num_warps=num_warps, num_stages=2,
            )

        return run

    # 2x L2, written before every timed launch to evict the previous pass.
    thrash = torch.empty(192 * 2**20, dtype=torch.uint8, device=DEV)
    t_full_warm = graph_time(run_full)
    t_full = cold_time(run_full, thrash)
    t_mem = cold_time(make_twin(0), thrash)
    t_lut = cold_time(make_twin(1), thrash)
    t_lut16 = cold_time(make_twin(2), thrash)
    t_arith = cold_time(make_twin(3), thrash)

    # Per-kernel split: the sweep only ever reported stage1 + stage2 together.
    # Profiled inside the same cold-L2 loop so the split matches t_full.
    for _ in range(5):
        run_full()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for i in range(20):
            thrash.fill_(i & 0xFF)
            run_full()
        torch.cuda.synchronize()
    stage = {}
    for ev in prof.key_averages():
        if "mxfp4" in ev.key:
            stage[ev.key] = ev.self_device_time_total / 20 / 1e3
    t_s1 = sum(v for k, v in stage.items() if "stage1" in k)
    t_s2 = sum(v for k, v in stage.items() if "stage2" in k)

    useful = 2 * 2 * Q_HEADS * HEAD_DIM * seq_len
    issued = useful * (head_tiles * mod._GROUPED_BLOCK_H) / Q_HEADS
    kv_bytes = 2 * seq_len * KV_HEADS * (HEAD_DIM // 2 + nblk)
    idx_bytes = head_tiles * seq_len * 4
    logit_bytes = BATCH * Q_HEADS * splits * (HEAD_DIM + 1) * 4
    total_bytes = kv_bytes + idx_bytes + 2 * logit_bytes
    ai = useful / kv_bytes
    floor_ms = total_bytes / (PEAK_READ_GBPS * 1e9) * 1e3

    print(f"\n=== seq_len={seq_len}  splits={splits}  CTAs={head_tiles * splits} "
          f"warps={num_warps}  {ctas_per_sm} CTA/SM ===")
    print(f"  traffic   : KV {kv_bytes / 2**20:7.1f} MiB | kv_indices "
          f"{idx_bytes / 2**20:5.2f} | logits r+w {2 * logit_bytes / 2**20:5.2f} "
          f"| total {total_bytes / 2**20:7.1f} MiB (L2 = 96 MiB)")
    print(f"  flops     : useful {useful / 1e9:6.3f} GFLOP | issued "
          f"{issued / 1e9:6.3f} GFLOP (BLOCK_H {mod._GROUPED_BLOCK_H} vs "
          f"{valid_h} valid = {issued / useful:.2f}x pad)")
    print(f"  intensity : {ai:5.1f} FLOP/B useful "
          f"(ridge {PEAK_BF16_TFLOPS * 1e12 / (PEAK_READ_GBPS * 1e9):5.1f}) "
          f"-> memory bound")
    print(f"  roofline  : {floor_ms:.4f} ms at {PEAK_READ_GBPS:.0f} GB/s")
    print(f"  measured  : fwd cold {t_full:.4f} ms (warm/graph {t_full_warm:.4f}) "
          f"= stage1 {t_s1:.4f} + stage2 {t_s2:.4f}")
    print(f"  twin cold : loads {t_mem:.4f} | +LUT4096 {t_lut:.4f} "
          f"| +LUT16*scale {t_lut16:.4f} | +arith {t_arith:.4f} ms")
    print(f"  --> MBU   : stage1 {total_bytes / (t_s1 / 1e3) / 1e9 / PEAK_READ_GBPS * 100:5.1f}% "
          f"| twin {total_bytes / (t_mem / 1e3) / 1e9 / PEAK_READ_GBPS * 100:5.1f}% "
          f"({total_bytes / (t_s1 / 1e3) / 1e9:.0f} GB/s vs twin "
          f"{total_bytes / (t_mem / 1e3) / 1e9:.0f} GB/s)")
    print(f"  --> MFU   : useful {useful / (t_s1 / 1e3) / 1e12 / PEAK_BF16_TFLOPS * 100:5.1f}% "
          f"| issued {issued / (t_s1 / 1e3) / 1e12 / PEAK_BF16_TFLOPS * 100:5.1f}% "
          f"({useful / (t_s1 / 1e3) / 1e12:.1f} / "
          f"{issued / (t_s1 / 1e3) / 1e12:.1f} TFLOP/s)")
    print(f"  --> budget: loads {t_mem:.4f} ({t_mem / t_s1 * 100:.0f}%) "
          f"| LUT gather +{t_lut - t_mem:.4f} ({(t_lut - t_mem) / t_s1 * 100:.0f}%) "
          f"| dot+softmax +{t_s1 - t_lut:.4f} ({(t_s1 - t_lut) / t_s1 * 100:.0f}%)")
    print(f"  --> dequant cost over loads: LUT4096 +{t_lut - t_mem:.4f} | "
          f"LUT16*scale +{t_lut16 - t_mem:.4f} | arith +{t_arith - t_mem:.4f} ms")
    print(f"  --> gap   : stage1/roofline {t_s1 / floor_ms:.2f}x | "
          f"stage1/twin {t_s1 / t_mem:.2f}x | twin/roofline {t_mem / floor_ms:.2f}x")


def main():
    seq_lens = [int(a) for a in sys.argv[1:]] or [65536, 100000, 150000]
    print(f"GPU={torch.cuda.get_device_name()} "
          f"peak_read={PEAK_READ_GBPS:.0f} GB/s peak_bf16={PEAK_BF16_TFLOPS:.0f} TFLOP/s")
    kv = make_kv(max(seq_lens) + 16)
    q = torch.randn(
        (BATCH, Q_HEADS, HEAD_DIM),
        dtype=torch.bfloat16,
        device=DEV,
        generator=torch.Generator(device=DEV).manual_seed(7),
    )
    for seq_len in seq_lens:
        analyse(seq_len, kv, q)


if __name__ == "__main__":
    main()
