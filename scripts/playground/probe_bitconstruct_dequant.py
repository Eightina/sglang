"""Probe: integer bit-construction E2M1 x E8M0 -> bf16 vs the OCP oracle.

Verifies the shift/or dequant (no exp2, no fp32 multiply) is element-exact
with ``mxfp4_dequantize_reference`` over the full code x scale-byte matrix,
including the bf16 normal/subnormal boundary (value 2^-127), signed zeros,
and overflow-to-inf at out-of-range scales.
"""

import torch
import triton
import triton.language as tl

_DEVICE = "cuda"


@triton.jit
def _bitconstruct_dequant_kernel(
    Codes,  # (N, D) uint8, raw 4-bit codes (not packed)
    SBytes,  # (N, D//32) uint8 e8m0
    Out,  # (N, D) bf16
    stride_cn,
    stride_sn,
    stride_on,
    D: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)
    offs_n = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, D)
    # NOTE: probe launches exact multiples of BLOCK_N (256/64), no masking.

    code = tl.load(Codes + offs_n[:, None] * stride_cn + offs_d[None, :])
    sbyte = tl.load(
        SBytes + offs_n[:, None] * stride_sn + (offs_d // 32)[None, :]
    )

    e = (code >> 1) & 3
    m = (code & 1).to(tl.int32)
    sgn = (code & 8).to(tl.int32) << 12  # bit3 -> bit15
    b = sbyte.to(tl.int32)

    is_zero = (e == 0) & (m == 0)
    # Unified exponent field: normal e>=1 -> e+b-1 (mantissa m<<6);
    # e2m1-subnormal m=1 -> b-1 (mantissa 0).
    E = tl.where(e == 0, b - 1, e + b - 1)
    mant = tl.where(e == 0, 0, m << 6)

    # bf16 normal/subnormal bridge: E==0 means value 2^-127 (< smallest
    # normal 2^-126) -> subnormal encoding mant = 1 << (E+6); E < -6 under-
    # flows to zero. Shift amount clamped to keep the discarded lane defined.
    is_sub = E < 1
    sub_mant = tl.where(E >= -6, 1 << tl.minimum(E + 6, 31), 0)
    is_inf = E >= 255
    is_nan = b == 255

    normal_bits = tl.where(E > 254, 0x7F80, (E << 7) | mant)
    bits = tl.where(is_sub, sub_mant, normal_bits)
    bits = tl.where(is_inf, 0x7F80, bits)
    bits = tl.where(is_zero, 0, bits)
    bits = tl.where(is_nan, 0x7FC0, bits)  # NaN highest priority (oracle: NaN scale -> all-NaN)
    bits = bits | sgn

    tl.store(
        Out + offs_n[:, None] * stride_on + offs_d[None, :],
        (bits & 0xFFFF).to(tl.uint16).to(tl.bfloat16, bitcast=True),
    )


def main():
    from sglang.test.kits.attention_unittest.attention_methods.mxfp4_decode_attention import (
        mxfp4_dequantize_reference,
    )

    # Full matrix: all 16 codes x all 256 scale bytes, dim 256 (8 blocks).
    # Full coverage: every (code, scale-byte) pair. Column c carries code
    # c%16; row r carries scale bytes (r*8+blk) % 256 across its 8 blocks ->
    # all 256 e8m0 bytes x all 16 codes appear (4096 combinations).
    codes = (torch.arange(256, dtype=torch.int64, device=_DEVICE) % 16).to(
        torch.uint8
    ).view(1, 256).repeat(256, 1)
    scales = (
        (
            torch.arange(256, dtype=torch.int64, device=_DEVICE)[:, None] * 8
            + torch.arange(8, dtype=torch.int64, device=_DEVICE)[None, :]
        )
        % 256
    ).to(torch.uint8).view(256, 8)

    out = torch.empty((256, 256), dtype=torch.bfloat16, device=_DEVICE)
    _bitconstruct_dequant_kernel[(256 // 64,)](
        codes, scales, out,
        codes.stride(0), scales.stride(0), out.stride(0),
        D=256, BLOCK_N=64,
    )
    torch.cuda.synchronize()

    ref = mxfp4_dequantize_reference(
        (codes[..., 0::2] | (codes[..., 1::2] << 4)).unsqueeze(0),
        scales.unsqueeze(0),
        logical_dim=256,
        dtype=torch.bfloat16,
    )[0]
    same = (out == ref) | (out.isnan() & ref.isnan())
    n_bad = (~same).sum().item()
    print(f"mismatches: {n_bad} / {out.numel()}")
    if n_bad:
        bad_scales = sorted(
            set(scales.repeat_interleave(32, dim=1).flatten()[~same.flatten()].tolist())
        )
        print(f"mismatching scale bytes: {bad_scales}")
    if n_bad:
        idx = (~same).nonzero()[:8]
        for i in idx:
            r, c = i.tolist()
            print(
                f"  code={codes[r,c].item():02x} scale={scales[r,c//32].item():02x} "
                f"kernel={out[r,c].item()} oracle={ref[r,c].item()}"
            )
    else:
        print("bit-construct dequant is element-exact with the OCP oracle "
              "(incl. NaN/Inf/zero/subnormal boundaries)")


if __name__ == "__main__":
    main()
