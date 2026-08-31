"""Strict OCP MXFP4 paged/radix decode-attention reference (test-only).

The codec implementation in this module is intentionally independent from the
runtime ``MXFP4KVQuantizeUtil``. Tests compare their packed E2M1 bytes and E8M0
scale bytes bit-for-bit before this reference is used as the L2 decode oracle.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from sglang.test.kits.attention_unittest.attention_methods.fp8_decode_attention import (
    DecodeMathDiagnostics,
    torch_radix_decode_from_effective_kv,
)

MXFP4_BLOCK_SIZE = 32
E2M1_MAX_POWER_OF_TWO = 4.0
E8M0_MIN_EXP = -127
E8M0_MAX_EXP = 127
E8M0_NAN_BYTE = 0xFF

_E2M1_POSITIVE_VALUES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def _round_e2m1_rne_reference(values: torch.Tensor) -> torch.Tensor:
    """Return positive E2M1 codes using round-to-nearest, ties-to-even."""
    values = values.to(torch.float32).abs().clamp(max=6.0)
    table = values.new_tensor(_E2M1_POSITIVE_VALUES)
    distance = (values.unsqueeze(-1) - table).abs()
    minimum = distance.amin(dim=-1, keepdim=True)
    tied = distance == minimum
    code_ids = torch.arange(8, dtype=torch.int64, device=values.device)
    even_tied = tied & ((code_ids & 1) == 0)
    has_even = even_tied.any(dim=-1)
    even_code = even_tied.to(torch.uint8).argmax(dim=-1).to(torch.uint8)
    first_code = tied.to(torch.uint8).argmax(dim=-1).to(torch.uint8)
    return torch.where(has_even, even_code, first_code)


def mxfp4_quantize_reference(tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize ``[tokens, heads, dim]`` to OCP MXFP4 bytes.

    Blocks are independent along each head's final dimension. The final partial
    block and an odd high nibble are padded with positive zero.
    """
    if tensor.ndim != 3:
        raise ValueError(f"MXFP4 expects a 3-D [tokens, heads, dim] tensor, got {tensor.shape}")
    tokens, heads, logical_dim = tensor.shape
    if logical_dim <= 0:
        raise ValueError("MXFP4 logical_dim must be positive")

    num_blocks = (logical_dim + MXFP4_BLOCK_SIZE - 1) // MXFP4_BLOCK_SIZE
    padded_dim = num_blocks * MXFP4_BLOCK_SIZE
    values = tensor.to(torch.float32)
    if padded_dim != logical_dim:
        values = torch.nn.functional.pad(values, (0, padded_dim - logical_dim))
    blocks = values.reshape(tokens, heads, num_blocks, MXFP4_BLOCK_SIZE)

    nan_blocks = torch.isnan(blocks).any(dim=-1)
    inf_blocks = torch.isinf(blocks).any(dim=-1) & ~nan_blocks
    finite_abs = torch.nan_to_num(blocks.abs(), nan=0.0, posinf=0.0, neginf=0.0)
    amax = finite_abs.amax(dim=-1)

    scale_exp = torch.floor(torch.log2(amax)) - 2.0
    scale_exp = torch.where(amax == 0, scale_exp.new_full((), E8M0_MIN_EXP), scale_exp)
    scale_exp = torch.where(inf_blocks, scale_exp.new_full((), E8M0_MAX_EXP), scale_exp)
    scale_exp = scale_exp.clamp(E8M0_MIN_EXP, E8M0_MAX_EXP)
    scale_bytes = (scale_exp.to(torch.int32) + 127).to(torch.uint8)
    scale_bytes = torch.where(
        nan_blocks,
        scale_bytes.new_full((), E8M0_NAN_BYTE),
        scale_bytes,
    )

    scale = torch.exp2(scale_exp).unsqueeze(-1)
    scaled = blocks / scale
    scaled = torch.nan_to_num(scaled, nan=0.0, posinf=6.0, neginf=-6.0)
    magnitude = _round_e2m1_rne_reference(scaled)
    codes = magnitude | (torch.signbit(scaled).to(torch.uint8) << 3)
    codes = torch.where(nan_blocks.unsqueeze(-1), torch.zeros_like(codes), codes)
    codes = codes.reshape(tokens, heads, padded_dim)[..., :logical_dim]

    if logical_dim % 2:
        codes = torch.nn.functional.pad(codes, (0, 1))
    packed = codes[..., 0::2] | (codes[..., 1::2] << 4)
    return packed.contiguous(), scale_bytes.contiguous()


def mxfp4_dequantize_reference(
    packed: torch.Tensor,
    scale_bytes: torch.Tensor,
    *,
    logical_dim: int,
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Dequantize raw packed E2M1 and E8M0 bytes to ``dtype``."""
    if packed.ndim != 3 or scale_bytes.ndim != 3:
        raise ValueError("MXFP4 packed data and scales must both be 3-D")
    if logical_dim <= 0 or packed.shape[-1] != (logical_dim + 1) // 2:
        raise ValueError(
            f"packed last dim {packed.shape[-1]} does not match logical_dim {logical_dim}"
        )
    expected_blocks = (logical_dim + MXFP4_BLOCK_SIZE - 1) // MXFP4_BLOCK_SIZE
    if scale_bytes.shape[:-1] != packed.shape[:-1] or scale_bytes.shape[-1] != expected_blocks:
        raise ValueError(
            f"scale shape {scale_bytes.shape} does not match packed shape {packed.shape} "
            f"and logical_dim {logical_dim}"
        )

    packed_bytes = packed.view(torch.uint8)
    raw_codes = torch.empty(
        *packed_bytes.shape[:-1], packed_bytes.shape[-1] * 2,
        dtype=torch.uint8,
        device=packed.device,
    )
    raw_codes[..., 0::2] = packed_bytes & 0x0F
    raw_codes[..., 1::2] = (packed_bytes >> 4) & 0x0F
    raw_codes = raw_codes[..., :logical_dim]

    table = packed_bytes.new_tensor(_E2M1_POSITIVE_VALUES, dtype=torch.float32)
    magnitude = table[(raw_codes & 0x07).long()]
    elements = torch.where((raw_codes & 0x08) != 0, -magnitude, magnitude)

    num_blocks = scale_bytes.shape[-1]
    padded_dim = num_blocks * MXFP4_BLOCK_SIZE
    if padded_dim != logical_dim:
        elements = torch.nn.functional.pad(elements, (0, padded_dim - logical_dim))
    elements = elements.reshape(*elements.shape[:-1], num_blocks, MXFP4_BLOCK_SIZE)

    raw_scales = scale_bytes.view(torch.uint8)
    nan_blocks = raw_scales == E8M0_NAN_BYTE
    scale_exp = raw_scales.to(torch.int16) - 127
    scales = torch.exp2(scale_exp.to(torch.float32))
    scales = torch.where(nan_blocks, scales.new_full((), float("nan")), scales)
    output = (elements * scales.unsqueeze(-1)).flatten(-2)[..., :logical_dim]
    return output.to(dtype)


@dataclass
class Mxfp4DecodeDiagnostics:
    k_packed: torch.Tensor
    v_packed: torch.Tensor
    k_scales: torch.Tensor
    v_scales: torch.Tensor
    k_dequant: torch.Tensor
    v_dequant: torch.Tensor
    scores: torch.Tensor
    probs: torch.Tensor
    output: torch.Tensor


def torch_mxfp4_radix_decode_reference(
    q: torch.Tensor,
    k_packed_cache: torch.Tensor,
    v_packed_cache: torch.Tensor,
    k_scale_cache: torch.Tensor,
    v_scale_cache: torch.Tensor,
    req_to_token: torch.Tensor,
    req_pool_indices,
    seq_lens,
    *,
    scaling: float,
    logical_dim: int | None = None,
    return_diagnostics: bool = False,
):
    """Decode one request from a paged MXFP4 cache with shared Torch math."""
    logical_dim = q.shape[-1] if logical_dim is None else logical_dim
    seq_len = int(seq_lens)
    locs = req_to_token[int(req_pool_indices), :seq_len].long()
    k_packed = k_packed_cache[locs]
    v_packed = v_packed_cache[locs]
    k_scales = k_scale_cache[locs]
    v_scales = v_scale_cache[locs]
    k_effective = mxfp4_dequantize_reference(
        k_packed, k_scales, logical_dim=logical_dim, dtype=torch.float32
    )
    v_effective = mxfp4_dequantize_reference(
        v_packed, v_scales, logical_dim=logical_dim, dtype=torch.float32
    )
    out, math_diag = torch_radix_decode_from_effective_kv(
        q,
        k_effective,
        v_effective,
        scaling=scaling,
        return_diagnostics=True,
    )
    if not return_diagnostics:
        return out
    assert isinstance(math_diag, DecodeMathDiagnostics)
    return out, Mxfp4DecodeDiagnostics(
        k_packed=k_packed,
        v_packed=v_packed,
        k_scales=k_scales,
        v_scales=v_scales,
        k_dequant=math_diag.k_dequant,
        v_dequant=math_diag.v_dequant,
        scores=math_diag.scores,
        probs=math_diag.probs,
        output=math_diag.output,
    )
