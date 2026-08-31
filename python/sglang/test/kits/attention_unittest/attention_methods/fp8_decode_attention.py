"""Pure-Torch FP8 (E4M3) paged/radix decode-attention reference (test-only).

This is the L1 golden of the FP8 KV decode path (see
qwen38_attention_survey_notes.md §11): it replicates

1. the production cache-write semantics — divide the model-dtype K/V by the
   checkpoint per-tensor scale, then cast to float8_e4m3fn (mirrors
   ``MHATokenToKVPool.set_kv_buffer``), and
2. the FlashInfer decode kernel's scale placement — the K descale folds into
   the QK^T boundary and the V descale into the PV boundary (equivalent in
   exact arithmetic to descaling the K/V tiles inside the kernel),

using explicit fp32 math so it is a deterministic oracle for
FlashInfer-equivalence tests today and, later, for the MXFP4 golden (L2)
built by swapping only the KV codec. Test-only: never imported by runtime code.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

FP8_DTYPE = torch.float8_e4m3fn


# ---------------------------------------------------------------------------
# FP8 cache QDQ references
# ---------------------------------------------------------------------------


def fp8_cache_quantize_reference(x: torch.Tensor, scale=None) -> torch.Tensor:
    """Replicate ``MHATokenToKVPool.set_kv_buffer``'s FP8 write semantics.

    Production divides the incoming model-dtype tensor by the checkpoint scale
    in the source dtype (in-place) and then casts to float8_e4m3fn. We clone
    first so test inputs are never mutated by the reference.
    """
    x = x.clone()
    if scale is not None:
        x.div_(scale)
    return x.to(FP8_DTYPE)


def fp8_cache_dequantize_reference(cache: torch.Tensor, scale=None) -> torch.Tensor:
    """Effective K/V as seen by attention: fp8 -> fp32, then apply the
    checkpoint descale. Diagnostic helper; not the attention op order."""
    out = cache.to(torch.float32)
    if scale is not None:
        out = out * float(scale)
    return out


# ---------------------------------------------------------------------------
# Paged gather
# ---------------------------------------------------------------------------


def gather_kv_sequences(
    req_to_token: torch.Tensor,
    req_pool_indices,
    seq_lens,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
):
    """Gather per-request logical K/V through the radix page table.

    Returns ``[(k_i, v_i), ...]`` with each element shaped
    ``(seq_len_i, num_kv_heads, head_dim)``. Physical slots may be arbitrary;
    no assumption of contiguity or monotonicity is made.
    """
    seqs = []
    for i in range(len(seq_lens)):
        seq_len = int(seq_lens[i])
        locs = req_to_token[int(req_pool_indices[i]), :seq_len].long()
        seqs.append((k_cache[locs], v_cache[locs]))
    return seqs


# ---------------------------------------------------------------------------
# Decode attention references
# ---------------------------------------------------------------------------


@dataclass
class DecodeMathDiagnostics:
    """Codec-independent fp32 decode-attention intermediates."""

    k_dequant: torch.Tensor  # (q_heads, seq, dim) fp32
    v_dequant: torch.Tensor  # (q_heads, seq, dim) fp32
    scores: torch.Tensor  # (q_heads, seq) fp32
    probs: torch.Tensor  # (q_heads, seq) fp32
    output: torch.Tensor  # (q_heads, head_dim) fp32


def torch_radix_decode_from_effective_kv(
    q: torch.Tensor,
    k_effective: torch.Tensor,
    v_effective: torch.Tensor,
    *,
    scaling: float,
    return_diagnostics: bool = False,
):
    """Run the shared decode math on already dequantized logical K/V.

    ``k_effective`` and ``v_effective`` are logical sequences shaped
    ``(seq_len, num_kv_heads, head_dim)``. This function deliberately owns all
    codec-independent behavior shared by the FP8 L1 and MXFP4 L2 references:
    GQA head expansion, fp32 QK, softmax, and fp32 PV.
    """
    q32 = q.to(torch.float32)
    k32 = k_effective.to(torch.float32).transpose(0, 1)
    v32 = v_effective.to(torch.float32).transpose(0, 1)

    num_q_heads = q32.shape[0]
    num_kv_heads = k32.shape[0]
    assert num_q_heads % num_kv_heads == 0, "GQA requires q % kv == 0"
    group = num_q_heads // num_kv_heads
    if group > 1:
        k32 = k32.repeat_interleave(group, dim=0)
        v32 = v32.repeat_interleave(group, dim=0)

    scores = torch.einsum("hd,hsd->hs", q32, k32) * scaling
    probs = torch.softmax(scores, dim=-1)
    out = torch.einsum("hs,hsd->hd", probs, v32)

    if not return_diagnostics:
        return out
    return out, DecodeMathDiagnostics(
        k_dequant=k32,
        v_dequant=v32,
        scores=scores,
        probs=probs,
        output=out,
    )


@dataclass
class Fp8DecodeDiagnostics:
    """Intermediate values for layered failure attribution (write/QDQ vs
    gather vs QK scale vs softmax vs V scale)."""

    k_fp8: torch.Tensor  # gathered fp8 K (seq, kv_heads, head_dim)
    v_fp8: torch.Tensor
    k_dequant: torch.Tensor  # descaled K (q_heads, seq, dim) fp32
    v_dequant: torch.Tensor  # descaled V (q_heads, seq, dim) fp32
    scores: torch.Tensor  # post-scale QK^T (q_heads, seq) fp32
    probs: torch.Tensor  # softmax probs (q_heads, seq) fp32
    output: torch.Tensor  # final output (q_heads, head_dim) fp32


def torch_fp8_radix_decode_reference(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    req_to_token: torch.Tensor,
    req_pool_indices,
    seq_lens,
    *,
    scaling: float,
    k_scale=None,
    v_scale=None,
    return_diagnostics: bool = False,
):
    """FlashInfer-equivalent FP8 decode attention for one request.

    Args:
        q: (num_q_heads, head_dim), model dtype (bf16 in production).
        k_cache / v_cache: paged FP8 pools, (num_slots, num_kv_heads, head_dim).
        req_to_token: (max_reqs, max_context_len) logical->physical map.
        req_pool_indices / seq_lens: scalars for the single request.

    Scale placement mirrors the production kernel: ``K descale`` multiplies
    the QK^T result and ``V descale`` multiplies the PV result. All math in
    fp32; output stays fp32 (caller casts to the model dtype if needed).
    """
    seq_len = int(seq_lens)
    locs = req_to_token[int(req_pool_indices), :seq_len].long()
    k_fp8 = k_cache[locs]
    v_fp8 = v_cache[locs]
    q32 = q.to(torch.float32)

    k_descale = 1.0 if k_scale is None else float(k_scale)
    v_descale = 1.0 if v_scale is None else float(v_scale)

    k32 = k_fp8.to(torch.float32).transpose(0, 1)  # (kv_heads, seq, dim)
    v32 = v_fp8.to(torch.float32).transpose(0, 1)

    num_q_heads = q32.shape[0]
    num_kv_heads = k32.shape[0]
    group = num_q_heads // num_kv_heads
    assert num_q_heads % num_kv_heads == 0, "GQA requires q % kv == 0"
    if group > 1:
        k32 = k32.repeat_interleave(group, dim=0)
        v32 = v32.repeat_interleave(group, dim=0)

    scores = torch.einsum("hd,hsd->hs", q32, k32) * (scaling * k_descale)
    probs = torch.softmax(scores, dim=-1)
    out = torch.einsum("hs,hsd->hd", probs, v32) * v_descale

    if not return_diagnostics:
        return out
    diag = Fp8DecodeDiagnostics(
        k_fp8=k_fp8,
        v_fp8=v_fp8,
        k_dequant=k32 * k_descale,
        v_dequant=v32 * v_descale,
        scores=scores,
        probs=probs,
        output=out,
    )
    return out, diag


def torch_fp8_decode_dequant_first_reference(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    req_to_token: torch.Tensor,
    req_pool_indices,
    seq_lens,
    *,
    scaling: float,
    k_scale=None,
    v_scale=None,
) -> torch.Tensor:
    """Dequant-K/V-first variant of the decode reference.

    Identical to ``torch_fp8_radix_decode_reference`` in exact arithmetic;
    kept to cross-check the BMM-boundary scale placement implementation in
    fp32 (any disagreement indicates an implementation bug, not numerics).
    """
    seq_len = int(seq_lens)
    locs = req_to_token[int(req_pool_indices), :seq_len].long()
    k_effective = fp8_cache_dequantize_reference(k_cache[locs], k_scale)
    v_effective = fp8_cache_dequantize_reference(v_cache[locs], v_scale)
    return torch_radix_decode_from_effective_kv(
        q,
        k_effective,
        v_effective,
        scaling=scaling,
    )


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def decode_output_metrics(actual: torch.Tensor, expected: torch.Tensor) -> dict:
    """Comparable metrics for decode outputs: max abs, mean abs, relative L2,
    cosine, norm ratio. Both inputs flattened in fp32 (device-agnostic)."""
    e = expected.detach().float().reshape(-1)
    a = actual.detach().float().reshape(-1).to(e.device)
    assert a.shape == e.shape, f"shape mismatch: {a.shape} vs {e.shape}"
    diff = a - e
    norm_a = a.norm().item()
    norm_e = e.norm().item()
    cos = torch.nn.functional.cosine_similarity(a, e, dim=0).item()
    return {
        "max_abs": diff.abs().max().item(),
        "mean_abs": diff.abs().mean().item(),
        "rel_l2": (diff.norm() / (e.norm() + 1e-12)).item(),
        "cosine": cos,
        "norm_ratio": norm_a / (norm_e + 1e-12),
    }


def format_metrics(metrics: dict) -> str:
    return (
        f"max_abs={metrics['max_abs']:.3e} mean_abs={metrics['mean_abs']:.3e} "
        f"rel_l2={metrics['rel_l2']:.3e} cos={metrics['cosine']:.6f} "
        f"norm_ratio={metrics['norm_ratio']:.5f}"
    )
