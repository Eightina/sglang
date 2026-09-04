# SPDX-License-Identifier: Apache-2.0
"""MXFP4 native decode kernels (L3 prototype).

Two standalone Triton kernels for the OCP MXFP4 KV cache (block-32 E2M1 data
+ per-block E8M0 scale), consumed directly from the packed pool buffers --
no PLAIN materialization:

1. ``mxfp4_dequant_fwd``: dequant micro-kernel, packed + scales -> bf16/fp32.
   Validated element-exact against the independent OCP oracle
   (``mxfp4_dequantize_reference`` in the attention unittest kit) before the
   decode kernel is trusted.
2. ``mxfp4_decode_attention_fwd``: two-stage split-KV decode attention that
   mirrors the bf16/fp8 kernel structure in ``decode_attention.py``
   (``_fwd_kernel_stage1`` + ``_fwd_kernel_stage2``) but loads the packed
   bytes per token/head and dequantizes inline (fp32 math throughout).

Codec semantics (must stay bit-compatible with the production codec and the
oracle): low nibble of a packed byte = even-index element, high nibble =
odd-index element; E2M1 code magnitude table (0, .5, 1, 1.5, 2, 3, 4, 6)
with bit 3 = sign; E8M0 scale = 2^(byte - 127), applied by exact fp32 bit
construction (``_e8m0_to_f32``). A NaN scale byte (0xFF) propagates NaN
exactly like the oracle.

Intentionally NOT implemented in this prototype: lean attention, score_mod,
DCP, sliding window, logit cap, xai temperature, MLA, extend/prefill.
"""

import torch
import triton
import triton.language as tl

from sglang.kernels.ops.attention.decode_attention import _extract_kv_strides

MXFP4_BLOCK_SIZE = 32
# Kernel-side mirror of MXFP4_BLOCK_SIZE: @triton.jit functions may only read
# tl.constexpr globals.
_MXFP4_BLOCK = tl.constexpr(MXFP4_BLOCK_SIZE)
_MIN_BLOCK_KV = 32
_BLOCK_N = 64


@triton.jit
def _e8m0_to_f32(sbytes):
    """Bit-exact E8M0 byte -> fp32 scale factor 2^(byte - 127).

    Bit construction instead of ``tl.exp2``: the fp32 pattern ``byte << 23``
    is exact for bytes 1..254, byte 0 (2^-127) needs the subnormal pattern
    0x00400000, and byte 255 (E8M0 NaN) maps to a quiet NaN. The approximate
    ex2 instruction flushes subnormal results (FTZ), which silently zeroed
    the 2^-127 scale; plain fp32 multiplies downstream do not flush.
    """
    b = sbytes.to(tl.int32)
    bits = b << 23
    bits = tl.where(b == 0, 0x00400000, bits)
    bits = tl.where(b == 255, 0x7FC00000, bits)
    return bits.to(tl.float32, bitcast=True)


# ---------------------------------------------------------------------------
# 1. Dequant micro-kernel
# ---------------------------------------------------------------------------


@triton.jit
def _mxfp4_dequant_kernel(
    Packed,
    Scales,
    Out,
    stride_p_slot,
    stride_p_head,
    stride_s_slot,
    stride_s_head,
    stride_o_slot,
    stride_o_head,
    HEADS: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    PACKED_DIM: tl.constexpr,
    NUM_BLOCKS: tl.constexpr,
    Ld: tl.constexpr,
):
    # One program dequantizes one (slot, head) row.
    pid = tl.program_id(0)
    slot = pid // HEADS
    head = pid % HEADS

    offs_d = tl.arange(0, BLOCK_DMODEL)
    mask_d = offs_d < Ld
    byte_idx = offs_d // 2

    packed_base = Packed + slot * stride_p_slot + head * stride_p_head
    byte = tl.load(packed_base + byte_idx, mask=byte_idx < PACKED_DIM, other=0)
    is_odd = (offs_d % 2) == 1
    code = tl.where(is_odd, (byte >> 4) & 0x0F, byte & 0x0F)

    # E2M1 magnitude: exp = (code>>1)&3 (2 bits), mant = code&1 (1 bit).
    # exp 0: 0.5 * mant (subnormal: 0, 0.5)
    # exp e: 2^(e-1) * (1 + mant/2) (1, 1.5, 2, 3, 4, 6)
    mant = (code & 0x1).to(tl.float32)
    exp_bits = (code >> 1) & 0x3
    normal = tl.exp2(exp_bits.to(tl.float32) - 1.0) * (1.0 + 0.5 * mant)
    subnormal = 0.5 * mant
    magnitude = tl.where(exp_bits == 0, subnormal, normal)
    element = tl.where((code & 0x8) != 0, -magnitude, magnitude)

    # Per-element scale gather: block index = dim index // 32. Positions
    # past the block count are masked (their magnitudes are already 0, and
    # 0 * 2^-127 = 0 keeps them inert).
    block_idx = offs_d // _MXFP4_BLOCK
    sbytes = tl.load(
        Scales + slot * stride_s_slot + head * stride_s_head + block_idx,
        mask=block_idx < NUM_BLOCKS,
        other=0,
    )
    sf = _e8m0_to_f32(sbytes)

    out = element * sf
    tl.store(
        Out + slot * stride_o_slot + head * stride_o_head + offs_d,
        out,
        mask=mask_d,
    )


def mxfp4_dequant_fwd(
    packed: torch.Tensor,
    scales: torch.Tensor,
    logical_dim: int,
    out_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Dequantize ``(slots, heads, ceil(logical_dim/2))`` packed uint8 plus
    ``(slots, heads, num_blocks)`` E8M0 bytes to ``(slots, heads, logical_dim)``.

    Element-exact with ``mxfp4_dequantize_reference`` for every representable
    value: E2M1 magnitudes and 2^k scales are all exactly representable in
    both bf16 and fp32, and NaN-scale blocks propagate NaN identically.
    """
    if packed.ndim != 3 or scales.ndim != 3:
        raise ValueError("packed and scales must be 3-D (slots, heads, ...)")
    if packed.shape[:2] != scales.shape[:2]:
        raise ValueError(
            f"packed/scale leading dims differ: {packed.shape} vs {scales.shape}"
        )
    num_blocks = (logical_dim + MXFP4_BLOCK_SIZE - 1) // MXFP4_BLOCK_SIZE
    packed_dim = (logical_dim + 1) // 2
    if packed.shape[-1] != packed_dim or scales.shape[-1] != num_blocks:
        raise ValueError(
            f"packed/scale last dims {packed.shape[-1]}/{scales.shape[-1]} do not "
            f"match logical_dim={logical_dim}"
        )
    packed = packed.view(torch.uint8).contiguous()
    scales = scales.view(torch.uint8).contiguous()

    slots, heads, _ = packed.shape
    out = torch.empty((slots, heads, logical_dim), dtype=out_dtype, device=packed.device)
    if slots == 0:
        return out

    block_dmodel = triton.next_power_of_2(max(logical_dim, 2))
    _mxfp4_dequant_kernel[(slots * heads,)](
        packed,
        scales,
        out,
        packed.stride(0),
        packed.stride(1),
        scales.stride(0),
        scales.stride(1),
        out.stride(0),
        out.stride(1),
        HEADS=heads,
        BLOCK_DMODEL=block_dmodel,
        PACKED_DIM=packed_dim,
        NUM_BLOCKS=num_blocks,
        Ld=logical_dim,
    )
    return out


# ---------------------------------------------------------------------------
# 2. Native MXFP4 decode attention (two-stage split-KV)
# ---------------------------------------------------------------------------


@triton.jit
def _mxfp4_decode_stage1_kernel(
    Q,
    K_Packed,
    V_Packed,
    K_Scales,
    V_Scales,
    sm_scale,
    kv_indptr,
    kv_indices,
    Att_Out,
    Att_Lse,
    num_kv_splits,
    stride_qbs,
    stride_qh,
    stride_kp_bs,
    stride_kp_h,
    stride_kp_page,
    stride_kp_tok,
    stride_vp_bs,
    stride_vp_h,
    stride_vp_page,
    stride_vp_tok,
    stride_ks_bs,
    stride_ks_h,
    stride_vs_bs,
    stride_vs_h,
    stride_mid_ob,
    stride_mid_oh,
    stride_mid_os,
    kv_group_num: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_DV: tl.constexpr,
    PACKED_K_DIM: tl.constexpr,
    PACKED_V_DIM: tl.constexpr,
    NUM_K_BLOCKS: tl.constexpr,
    NUM_V_BLOCKS: tl.constexpr,
    BLOCK_N: tl.constexpr,
    MIN_BLOCK_KV: tl.constexpr,
    Lk: tl.constexpr,
    Lv: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
):
    # int64 to mirror the stock kernel's flat-offset overflow guard.
    cur_batch = tl.program_id(0).to(tl.int64)
    cur_head = tl.program_id(1)
    split_kv_id = tl.program_id(2)

    cur_kv_head = cur_head // kv_group_num

    offs_d = tl.arange(0, BLOCK_DMODEL)
    offs_dv = tl.arange(0, BLOCK_DV)
    mask_d = offs_d < Lk
    mask_dv = offs_dv < Lv
    byte_idx_k = offs_d // 2
    byte_idx_v = offs_dv // 2
    blk_idx_k = offs_d // _MXFP4_BLOCK
    blk_idx_v = offs_dv // _MXFP4_BLOCK

    cur_batch_kv_start_idx = tl.load(kv_indptr + cur_batch)
    cur_batch_seq_len = tl.load(kv_indptr + cur_batch + 1) - cur_batch_kv_start_idx
    kv_splits = tl.load(num_kv_splits + cur_batch)

    off_q = cur_batch * stride_qbs + cur_head * stride_qh + offs_d

    kv_len_per_split = (
        tl.cdiv(tl.cdiv(cur_batch_seq_len, kv_splits), MIN_BLOCK_KV) * MIN_BLOCK_KV
    )
    split_kv_start = kv_len_per_split * split_kv_id
    split_kv_end = tl.minimum(split_kv_start + kv_len_per_split, cur_batch_seq_len)

    e_max = -float("inf")
    e_sum = 0.0
    acc = tl.zeros([BLOCK_DV], dtype=tl.float32)

    if split_kv_end > split_kv_start:
        q = tl.load(Q + off_q, mask=mask_d, other=0.0).to(tl.float32)
        for start_n in range(split_kv_start, split_kv_end, BLOCK_N):
            offs_n = start_n + tl.arange(0, BLOCK_N)
            tok_mask = offs_n < split_kv_end
            kv_loc = tl.load(
                kv_indices + cur_batch_kv_start_idx + offs_n,
                mask=tok_mask,
                other=0,
            )
            # Page-aware address math, identical to the stock stage-1 kernel
            # (for these buffers it reduces to kv_loc * slot_stride).
            if PAGE_SIZE == 1:
                k_row = kv_loc * stride_kp_bs + cur_kv_head * stride_kp_h
                v_row = kv_loc * stride_vp_bs + cur_kv_head * stride_vp_h
            else:
                page_id = kv_loc // PAGE_SIZE
                tok_in_p = kv_loc % PAGE_SIZE
                k_row = (
                    page_id * stride_kp_page
                    + tok_in_p * stride_kp_tok
                    + cur_kv_head * stride_kp_h
                )
                v_row = (
                    page_id * stride_vp_page
                    + tok_in_p * stride_vp_tok
                    + cur_kv_head * stride_vp_h
                )
            ks_row = kv_loc * stride_ks_bs + cur_kv_head * stride_ks_h
            vs_row = kv_loc * stride_vs_bs + cur_kv_head * stride_vs_h

            # --- K tile: packed bytes -> E2M1 magnitudes -> E8M0 scale -> fp32
            kbyte = tl.load(
                K_Packed + k_row[:, None] + byte_idx_k[None, :],
                mask=tok_mask[:, None] & (byte_idx_k[None, :] < PACKED_K_DIM),
                other=0,
            )
            is_odd_k = (offs_d % 2) == 1
            kcode = tl.where(is_odd_k[None, :], (kbyte >> 4) & 0x0F, kbyte & 0x0F)
            kmant = (kcode & 0x1).to(tl.float32)
            kexp = (kcode >> 1) & 0x3
            knormal = tl.exp2(kexp.to(tl.float32) - 1.0) * (1.0 + 0.5 * kmant)
            ksub = 0.5 * kmant
            kmag = tl.where(kexp == 0, ksub, knormal)
            k = tl.where((kcode & 0x8) != 0, -kmag, kmag)
            k = tl.where(mask_d[None, :], k, 0.0)

            ksbytes = tl.load(
                K_Scales + ks_row[:, None] + blk_idx_k[None, :],
                mask=tok_mask[:, None] & (blk_idx_k[None, :] < NUM_K_BLOCKS),
                other=0,
            )
            ksf = _e8m0_to_f32(ksbytes)
            k = k * ksf

            qk = tl.sum(q[None, :] * k, 1)
            qk *= sm_scale
            qk = tl.where(tok_mask, qk, float("-inf"))

            # --- V tile: same inline dequant
            vbyte = tl.load(
                V_Packed + v_row[:, None] + byte_idx_v[None, :],
                mask=tok_mask[:, None] & (byte_idx_v[None, :] < PACKED_V_DIM),
                other=0,
            )
            is_odd_v = (offs_dv % 2) == 1
            vcode = tl.where(is_odd_v[None, :], (vbyte >> 4) & 0x0F, vbyte & 0x0F)
            vmant = (vcode & 0x1).to(tl.float32)
            vexp = (vcode >> 1) & 0x3
            vnormal = tl.exp2(vexp.to(tl.float32) - 1.0) * (1.0 + 0.5 * vmant)
            vsub = 0.5 * vmant
            vmag = tl.where(vexp == 0, vsub, vnormal)
            v = tl.where((vcode & 0x8) != 0, -vmag, vmag)
            v = tl.where(mask_dv[None, :], v, 0.0)

            vsbytes = tl.load(
                V_Scales + vs_row[:, None] + blk_idx_v[None, :],
                mask=tok_mask[:, None] & (blk_idx_v[None, :] < NUM_V_BLOCKS),
                other=0,
            )
            vsf = _e8m0_to_f32(vsbytes)
            v = v * vsf

            # --- online softmax, fp32 (same recurrence as the stock kernel)
            n_e_max = tl.maximum(tl.max(qk, 0), e_max)
            re_scale = tl.exp(e_max - n_e_max)
            p = tl.exp(qk - n_e_max)
            acc *= re_scale
            acc += tl.sum(p[:, None] * v, 0)

            e_sum = e_sum * re_scale + tl.sum(p, 0)
            e_max = n_e_max

        offs_mid_o = (
            cur_batch * stride_mid_ob
            + cur_head * stride_mid_oh
            + split_kv_id * stride_mid_os
            + offs_dv
        )
        tl.store(Att_Out + offs_mid_o, acc / e_sum, mask=mask_dv)

        offs_mid_o_1 = (
            cur_batch * stride_mid_ob
            + cur_head * stride_mid_oh
            + split_kv_id * stride_mid_os
        ) // Lv
        tl.store(Att_Lse + offs_mid_o_1, e_max + tl.log(e_sum))


@triton.jit
def _mxfp4_decode_stage2_kernel(
    Mid_O,
    Mid_O_1,
    O,
    kv_indptr,
    num_kv_splits,
    stride_mid_ob,
    stride_mid_oh,
    stride_mid_os,
    stride_obs,
    stride_oh,
    MAX_KV_SPLITS: tl.constexpr,
    MIN_BLOCK_KV: tl.constexpr,
    BLOCK_DV: tl.constexpr,
    Lv: tl.constexpr,
):
    # Codec-agnostic split merge; mirrors the stock ``_fwd_kernel_stage2``
    # (v_scale fixed at 1.0: MXFP4 scales are applied inline in stage 1).
    cur_batch = tl.program_id(0).to(tl.int64)
    cur_head = tl.program_id(1)

    cur_batch_seq_len = tl.load(kv_indptr + cur_batch + 1) - tl.load(
        kv_indptr + cur_batch
    )
    kv_splits = tl.load(num_kv_splits + cur_batch)

    offs_d = tl.arange(0, BLOCK_DV)
    mask_d = offs_d < Lv

    e_sum = 0.0
    e_max = -float("inf")
    acc = tl.zeros([BLOCK_DV], dtype=tl.float32)

    offs_v = cur_batch * stride_mid_ob + cur_head * stride_mid_oh + offs_d
    offs_logic = (cur_batch * stride_mid_ob + cur_head * stride_mid_oh) // Lv
    kv_len_per_split = (
        tl.cdiv(tl.cdiv(cur_batch_seq_len, kv_splits), MIN_BLOCK_KV) * MIN_BLOCK_KV
    )

    for split_kv_id in tl.range(0, MAX_KV_SPLITS, num_stages=2):
        split_kv_start = kv_len_per_split * split_kv_id
        split_kv_end = tl.minimum(split_kv_start + kv_len_per_split, cur_batch_seq_len)

        if split_kv_end > split_kv_start:
            tv = tl.load(
                Mid_O + offs_v + split_kv_id * stride_mid_os, mask=mask_d, other=0.0
            )
            tlogic = tl.load(Mid_O_1 + offs_logic + split_kv_id * stride_mid_os // Lv)
            n_e_max = tl.maximum(tlogic, e_max)

            old_scale = tl.exp(e_max - n_e_max)
            acc *= old_scale
            exp_logic = tl.exp(tlogic - n_e_max)
            acc += exp_logic * tv

            e_sum = e_sum * old_scale + exp_logic
            e_max = n_e_max

    tl.store(
        O + cur_batch * stride_obs + cur_head * stride_oh + offs_d,
        acc / e_sum,
        mask=mask_d,
    )


def _host_num_kv_splits(seq_lens_cpu, max_kv_splits):
    """Test/prototype split schedule: fill up to max_kv_splits with at least
    MIN_BLOCK_KV tokens per split. The production backend brings its own
    on-device schedule (Phase 2); correctness holds for any split count."""
    splits = []
    for seq_len in seq_lens_cpu:
        seq_len = int(seq_len)
        splits.append(max(1, min(max_kv_splits, triton.cdiv(seq_len, _MIN_BLOCK_KV))))
    return splits


def mxfp4_decode_attention_fwd(
    q: torch.Tensor,
    k_packed: torch.Tensor,
    v_packed: torch.Tensor,
    k_scales: torch.Tensor,
    v_scales: torch.Tensor,
    o: torch.Tensor,
    kv_indptr: torch.Tensor,
    kv_indices: torch.Tensor,
    sm_scale: float,
    page_size: int = 1,
    max_kv_splits: int = 8,
    attn_logits: torch.Tensor | None = None,
    attn_lse: torch.Tensor | None = None,
    num_kv_splits: torch.Tensor | None = None,
):
    """Native MXFP4 decode attention over packed paged KV.

    Args:
        q: (batch, q_heads, head_dim), bf16/fp16/fp32.
        k_packed / v_packed: (slots, kv_heads, ceil(head_dim/2)) uint8.
        k_scales / v_scales: (slots, kv_heads, head_dim//32) E8M0 bytes.
        o: (batch, q_heads, v_head_dim) output buffer; written in its dtype
            (fp32 for oracle comparison).
        kv_indptr / kv_indices: standard ragged page-table metadata (indices
            are physical slot ids; page_size > 1 uses the same page/tok
            address math as the stock triton decode kernel).
        attn_logits / attn_lse / num_kv_splits: optional preallocated
            split-KV scratch (CUDA-graph reuse in the backend integration).

    Note: ``num_kv_splits=None`` derives the schedule on the host (prototype
    convenience; involves a device->host sync). The backend passes its
    on-device schedule instead.
    """
    batch, head_num, head_dim = q.shape
    kv_head_num = k_packed.shape[1]
    if head_num % kv_head_num != 0:
        raise ValueError(
            f"GQA requires q_heads % kv_heads == 0, got {head_num} % {kv_head_num}"
        )
    kv_group_num = head_num // kv_head_num
    logical_k = head_dim
    logical_v = o.shape[-1]
    packed_k_dim = k_packed.shape[-1]
    packed_v_dim = v_packed.shape[-1]
    if packed_k_dim != (logical_k + 1) // 2 or packed_v_dim != (logical_v + 1) // 2:
        raise ValueError(
            f"packed dims {packed_k_dim}/{packed_v_dim} do not match logical "
            f"dims {logical_k}/{logical_v}"
        )
    if q.shape[0] != kv_indptr.shape[0] - 1:
        raise ValueError("kv_indptr must have batch + 1 entries")

    k_packed = k_packed.view(torch.uint8).contiguous()
    v_packed = v_packed.view(torch.uint8).contiguous()
    k_scales = k_scales.view(torch.uint8).contiguous()
    v_scales = v_scales.view(torch.uint8).contiguous()
    num_k_blocks = k_scales.shape[-1]
    num_v_blocks = v_scales.shape[-1]

    if num_kv_splits is None:
        seq_lens_cpu = (kv_indptr[1:] - kv_indptr[:-1]).cpu().tolist()
        splits = _host_num_kv_splits(seq_lens_cpu, max_kv_splits)
        num_kv_splits = torch.tensor(splits, dtype=torch.int32, device=q.device)
    if attn_logits is None:
        attn_logits = torch.empty(
            (batch, head_num, max_kv_splits, logical_v),
            dtype=torch.float32,
            device=q.device,
        )
    if attn_lse is None:
        attn_lse = torch.empty(
            (batch, head_num, max_kv_splits), dtype=torch.float32, device=q.device
        )
    if attn_logits.shape[-1] != logical_v:
        raise ValueError("attn_logits last dim must equal the V head dim")

    BLOCK_DMODEL = triton.next_power_of_2(logical_k)
    BLOCK_DV = triton.next_power_of_2(logical_v)

    (
        kp_slot_stride,
        kp_head_stride,
        kp_page_stride,
        kp_tok_stride,
    ) = _extract_kv_strides(k_packed, page_size)
    (
        vp_slot_stride,
        vp_head_stride,
        vp_page_stride,
        vp_tok_stride,
    ) = _extract_kv_strides(v_packed, page_size)
    ks_slot_stride, ks_head_stride, _, _ = _extract_kv_strides(k_scales, 1)
    vs_slot_stride, vs_head_stride, _, _ = _extract_kv_strides(v_scales, 1)

    grid = (batch, head_num, max_kv_splits)
    _mxfp4_decode_stage1_kernel[grid](
        q,
        k_packed,
        v_packed,
        k_scales,
        v_scales,
        sm_scale,
        kv_indptr,
        kv_indices,
        attn_logits,
        attn_lse,
        num_kv_splits,
        q.stride(0),
        q.stride(1),
        kp_slot_stride,
        kp_head_stride,
        kp_page_stride,
        kp_tok_stride,
        vp_slot_stride,
        vp_head_stride,
        vp_page_stride,
        vp_tok_stride,
        ks_slot_stride,
        ks_head_stride,
        vs_slot_stride,
        vs_head_stride,
        attn_logits.stride(0),
        attn_logits.stride(1),
        attn_logits.stride(2),
        kv_group_num=kv_group_num,
        BLOCK_DMODEL=BLOCK_DMODEL,
        BLOCK_DV=BLOCK_DV,
        PACKED_K_DIM=packed_k_dim,
        PACKED_V_DIM=packed_v_dim,
        NUM_K_BLOCKS=num_k_blocks,
        NUM_V_BLOCKS=num_v_blocks,
        BLOCK_N=_BLOCK_N,
        MIN_BLOCK_KV=_MIN_BLOCK_KV,
        Lk=logical_k,
        Lv=logical_v,
        PAGE_SIZE=page_size,
        num_warps=4 if kv_group_num == 1 else 2,
        num_stages=2,
    )

    _mxfp4_decode_stage2_kernel[(batch, head_num)](
        attn_logits,
        attn_lse,
        o,
        kv_indptr,
        num_kv_splits,
        attn_logits.stride(0),
        attn_logits.stride(1),
        attn_logits.stride(2),
        o.stride(0),
        o.stride(1),
        MAX_KV_SPLITS=max_kv_splits,
        MIN_BLOCK_KV=_MIN_BLOCK_KV,
        BLOCK_DV=BLOCK_DV,
        Lv=logical_v,
    )
