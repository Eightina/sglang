#!/usr/bin/env python3
"""Is the hardware E2M1 decode bit-identical to the 4096-entry LUT?

The LUT is the codec of record: test_lut_matches_eager_codec enumerates all
4096 (E8M0 byte, E2M1 code) pairs against the eager OCP reference and demands
bit-identical results, NaN scale byte included. Swapping the gather for
cvt.rn.f16x2.e2m1x2 plus a separate scale multiply only holds up if it
reproduces that table exactly, including the awkward rows: scale byte 0
(2^-127, a bf16 subnormal), scale byte 255 (E8M0 NaN), signed zeros, and the
codes whose product overflows to inf.

Checks both nibble positions, since the low nibble and high nibble come out of
different halves of the converter's result.
"""
import torch
import triton
import triton.language as tl

from sglang.kernels.ops.attention.decode_attention_mxfp4_sm120 import (
    _cvt_e2m1_hi,
    _cvt_e2m1_lo,
    _e8m0_to_bf16,
    e2m1_scale_lut,
)


@triton.jit
def _probe(Byte, SByte, OutLo, OutHi, N: tl.constexpr):
    i = tl.arange(0, N)
    byte = tl.load(Byte + i)
    scale = _e8m0_to_bf16(tl.load(SByte + i))
    tl.store(OutLo + i, _cvt_e2m1_lo(byte).to(tl.bfloat16) * scale)
    tl.store(OutHi + i, _cvt_e2m1_hi(byte).to(tl.bfloat16) * scale)


def main():
    dev = "cuda"
    # 4096 pairs laid out so the low nibble carries the code in one pass and
    # the high nibble carries it in the other.
    codes = torch.arange(16, device=dev).repeat(256)
    sbytes = torch.arange(256, device=dev).repeat_interleave(16)
    n = codes.numel()

    lut = e2m1_scale_lut(dev, torch.bfloat16)
    want = lut[(sbytes.to(torch.int32) << 4) | codes.to(torch.int32)]

    bad = 0
    for pos in ("lo", "hi"):
        packed = (codes if pos == "lo" else codes << 4).to(torch.uint8)
        out_lo = torch.zeros(n, dtype=torch.bfloat16, device=dev)
        out_hi = torch.zeros(n, dtype=torch.bfloat16, device=dev)
        _probe[(1,)](
            packed.contiguous(),
            sbytes.to(torch.uint8).contiguous(),
            out_lo,
            out_hi,
            N=n,
            num_warps=4,
        )
        torch.cuda.synchronize()
        got = out_lo if pos == "lo" else out_hi

        want_nan = torch.isnan(want)
        got_nan = torch.isnan(got)
        nan_ok = torch.equal(want_nan, got_nan)
        bits_eq = torch.equal(
            got.view(torch.int16)[~want_nan], want.view(torch.int16)[~want_nan]
        )
        print(f"[{pos} nibble] NaN pattern match: {nan_ok} | "
              f"non-NaN bitwise equal: {bits_eq}")
        if not (nan_ok and bits_eq):
            bad += 1
            diff = (~want_nan) & (got.view(torch.int16) != want.view(torch.int16))
            idx = diff.nonzero().flatten()[:12]
            for j in idx.tolist():
                print(f"   sbyte={sbytes[j].item():3d} code={codes[j].item():2d} "
                      f"want={want[j].item()!r} (0x{want.view(torch.int16)[j] & 0xFFFF:04x}) "
                      f"got={got[j].item()!r} (0x{got.view(torch.int16)[j] & 0xFFFF:04x})")

    # Spot-check the rows that motivated the selects.
    print()
    for sb, label in ((0, "2^-127 (bf16 subnormal)"), (255, "E8M0 NaN"), (127, "2^0")):
        sel = sbytes == sb
        w = want[sel]
        print(f"  sbyte={sb:3d} {label:24s} lut[0..7] = "
              f"{[round(float(v), 6) for v in w[:8]]}")

    print("\nRESULT:", "bit-exact on both nibbles" if bad == 0 else f"{bad} position(s) MISMATCH")


if __name__ == "__main__":
    main()
