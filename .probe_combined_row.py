#!/usr/bin/env python3
"""Feasibility probe for the combined MXFP4 KV row layout.

Direction A of ``mxfp4_decode_layout_ws_profiling.md``: put the 8 E8M0 bytes of
a (slot, head) in the same row as its 128 packed codes, so the scale load shares
the slot/head address math with the packed load instead of building a second
pointer tile. The probe answers three questions before the cross-layer refactor
is paid for:

1. Does sharing the row offset actually recover the 36% that P3 priced?
2. Does a non-128-byte row width break the ``LDGSTS.E.BYPASS.128`` async copy?
   A 136 B row is only 8-byte aligned on odd slots, so the vectorizer may fall
   back; 144 B and 160 B are 16-byte aligned on every slot.
3. What does the extra padding cost in traffic?

Correctness is checked too: the combined tensor holds the same bytes, so the
output must match the two-tensor path bit for bit.

Usage: ``python .probe_combined_row.py [seq ...]``
"""

import sys

import torch

from sglang.kernels.ops.attention.decode_attention_mxfp4_sm120 import (
    MXFP4_BLOCK_SIZE,
    mxfp4_decode_attention_fwd,
)

DEV = "cuda"
# Measured achievable read bandwidth on this card (not the spec number).
PEAK_READ_GBPS = 1686.6

H_Q, H_KV, D = 24, 4, 256
PACKED = D // 2
NBLK = D // MXFP4_BLOCK_SIZE

# Row widths to try. 128 is the packed-only baseline width; the others append
# the 8 scale bytes plus enough padding to keep each row's base address aligned.
#   136 = 128 + 8        -> 8-byte aligned only (136 * odd_slot is not 16|)
#   144 = 128 + 8 + 8    -> 16-byte aligned on every slot
#   160 = 128 + 8 + 24   -> 32-byte aligned on every slot (full sector)
#   256 = 128 + 8 + 120  -> 128-byte aligned, i.e. one whole L2 line per row,
#                           at the cost of a second line fetched for 8 bytes
WIDTHS = (136, 144, 160, 256)


def l2_flush():
    torch.empty(int(96e6), dtype=torch.int8, device=DEV).zero_()


def timeit(fn, iters=50, warmup=10):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        l2_flush()
        torch.cuda.synchronize()
        a, b = torch.cuda.Event(True), torch.cuda.Event(True)
        a.record()
        fn()
        b.record()
        torch.cuda.synchronize()
        ts.append(a.elapsed_time(b))
    ts.sort()
    return ts[len(ts) // 2]


def build(seq):
    torch.manual_seed(0)
    q = torch.randn(1, H_Q, D, dtype=torch.bfloat16, device=DEV)
    kp = torch.randint(0, 256, (seq, H_KV, PACKED), dtype=torch.uint8, device=DEV)
    vp = torch.randint(0, 256, (seq, H_KV, PACKED), dtype=torch.uint8, device=DEV)
    # Keep exponents in a sane range so the reference output is finite.
    ks = torch.randint(118, 136, (seq, H_KV, NBLK), dtype=torch.uint8, device=DEV)
    vs = torch.randint(118, 136, (seq, H_KV, NBLK), dtype=torch.uint8, device=DEV)
    indptr = torch.tensor([0, seq], dtype=torch.int32, device=DEV)
    # Sequential slots, matching ``.bench_fp8_attention.py``. A randperm page
    # table makes the packed stream a true gather and inflates every number by
    # ~50%, which is a different question than the one this probe asks.
    indices = torch.arange(seq, dtype=torch.int32, device=DEV)
    o = torch.empty(1, H_Q, D, dtype=torch.float32, device=DEV)
    return q, kp, vp, ks, vs, indptr, indices, o


def combine(packed, scales, width):
    """``[slots, heads, width]`` holding codes at col 0 and scales at col 128."""
    slots, heads, _ = packed.shape
    comb = torch.zeros(slots, heads, width, dtype=torch.uint8, device=DEV)
    comb[..., :PACKED] = packed
    comb[..., PACKED : PACKED + NBLK] = scales
    return comb


def run(seq):
    q, kp, vp, ks, vs, indptr, indices, o = build(seq)
    splits = 85
    common = dict(
        kv_indptr=indptr,
        kv_indices=indices,
        sm_scale=D**-0.5,
        max_kv_splits=splits,
        num_kv_splits=torch.tensor([splits], dtype=torch.int32, device=DEV),
        # Preallocated so the timing does not include the per-call allocation the
        # prototype path would otherwise do.
        attn_logits=torch.empty(1, H_Q, splits, D, dtype=torch.float32, device=DEV),
        attn_lse=torch.empty(1, H_Q, splits, dtype=torch.float32, device=DEV),
    )

    def call(k_p, v_p, k_s, v_s, base):
        mxfp4_decode_attention_fwd(
            q, k_p, v_p, k_s, v_s, o, scale_col_base=base, **common
        )

    # Baseline: two tensors, 128 B packed rows.
    base_t = timeit(lambda: call(kp, vp, ks, vs, 0))
    call(kp, vp, ks, vs, 0)
    ref = o.clone()
    # Bytes the kernel actually reads per token/head, both halves of KV. The row
    # padding is never touched, so this is the same for every width -- only the
    # addresses it lands on differ.
    base_bytes = seq * H_KV * (PACKED + NBLK) * 2
    print(f"\n=== seq={seq} ===")
    print(
        f"  {'layout':<22}{'ms':>9}{'GB/s':>9}{'MBU':>8}"
        f"{'row align':>11}{'vs base':>10}  exact"
    )
    print(
        f"  {'128 + separate':<22}{base_t:>9.4f}"
        f"{base_bytes / base_t / 1e6:>9.1f}"
        f"{base_bytes / base_t / 1e6 / PEAK_READ_GBPS * 100:>7.1f}%"
        f"{'128 B':>11}{'--':>10}  --"
    )

    for w in WIDTHS:
        kc = combine(kp, ks, w)
        vc = combine(vp, vs, w)
        # The codes go in as a strided view: shape keeps the 128-byte logical
        # width the kernel validates, while the row stride is the combined one,
        # so the packed and scale loads share the slot/head address expression.
        kcv, vcv = kc[..., :PACKED], vc[..., :PACKED]
        t = timeit(lambda: call(kcv, vcv, kc, vc, PACKED))
        call(kcv, vcv, kc, vc, PACKED)
        exact = torch.equal(o, ref)
        # Largest power of two that divides every row base address.
        align = w & -w
        print(
            f"  {f'{w} combined':<22}{t:>9.4f}"
            f"{base_bytes / t / 1e6:>9.1f}"
            f"{base_bytes / t / 1e6 / PEAK_READ_GBPS * 100:>7.1f}%"
            f"{f'{align} B':>11}"
            f"{(t / base_t - 1) * 100:>+9.1f}%  {'yes' if exact else 'NO'}"
        )
        del kc, vc, kcv, vcv
        torch.cuda.empty_cache()

    # P7, the single-variable form. Read the "scales" from inside the real
    # 128-byte packed row (columns 120..127, overlapping the codes). The values
    # are garbage so the output is wrong, but every other property is the one we
    # want: rows stay exactly one 128-byte L2 line, the scale load shares the
    # slot/head address expression with the packed load, and the instruction
    # count is unchanged. This separates "sharing the address helps" from "the
    # row width broke the line alignment".
    t = timeit(lambda: call(kp, vp, kp, vp, PACKED - NBLK))
    print(
        f"  {'P7 128 in-row (wrong)':<22}{t:>9.4f}"
        f"{base_bytes / t / 1e6:>9.1f}"
        f"{base_bytes / t / 1e6 / PEAK_READ_GBPS * 100:>7.1f}%"
        f"{'128 B':>11}"
        f"{(t / base_t - 1) * 100:>+9.1f}%  n/a"
    )


if __name__ == "__main__":
    seqs = [int(a) for a in sys.argv[1:]] or [100000, 150000]
    for s in seqs:
        run(s)
