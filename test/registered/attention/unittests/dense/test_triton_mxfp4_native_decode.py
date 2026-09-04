"""L3 Phase 1: native MXFP4 Triton decode kernels vs the L2 Torch oracle.

Two independent verification layers (see qwen38_attention_survey_notes.md
§11.7 and the L3 development plan):

1. ``TestMxfp4DequantKernel`` (A step): the dequant micro-kernel must be
   element-exact with the independent OCP oracle
   (``mxfp4_dequantize_reference``) for every representable value -- all
   E2M1 codes, E8M0 exponent boundaries, zero blocks, NaN scale blocks,
   partial blocks, odd logical dims, bf16 and fp32 outputs.
2. ``TestTritonMxfp4NativeDecode`` (B step): the two-stage split-KV decode
   kernel must agree with the L2 Torch golden
   (``torch_mxfp4_radix_decode_reference``, which reads the SAME packed pool
   bytes) within frozen thresholds after 20-seed characterization. Both
   sides consume identical dequantized values (A step proves the dequant is
   exact), so the residual is kernel accumulation order only -- the same
   interpretation as the L1/L2 frozen thresholds.

Standalone by design: no production code is touched in Phase 1; the backend
integration (access registry / hook / triton_backend dispatch) lands only
after these numbers are frozen.
"""

import contextlib
import io
import os
import unittest
from itertools import accumulate

import torch

from sglang.kernels.ops.attention.mxfp4_decode_attention import (
    mxfp4_decode_attention_fwd,
    mxfp4_dequant_fwd,
)
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kits.attention_unittest.attention_methods.fp8_decode_attention import (
    decode_output_metrics,
    format_metrics,
)
from sglang.test.kits.attention_unittest.attention_methods.mxfp4_decode_attention import (
    mxfp4_dequantize_reference,
    mxfp4_quantize_reference,
    torch_mxfp4_radix_decode_reference,
)
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=30, stage="base-b", runner_config="1-gpu-large")

_HAS_REQUIREMENTS = torch.cuda.is_available()

# Same hard caps as the L1/L2 goldens: they bound IMPLEMENTATION equivalence
# (Triton native vs Torch oracle on the SAME packed cache) and must never be
# relaxed to mask a scale/page/pack bug.
REL_L2_CAP = 2e-2
COSINE_CAP = 0.999
NORM_RATIO_RANGE = (0.98, 1.02)

# Filled in by TestTritonMxfp4NativeCharacterization (worst * 1.25 rounded
# outward, 20 seeds, SM120 / torch 2.13.0+cu130). Measured worst cases:
# rel_l2 2.61e-7 (qwen_24_4_256_multisplit_long), cosine 0.99999982
# (gqa_hd128_p16), |norm_ratio - 1| 1.17e-7 (qwen_24_4_256_splits1).
# Four orders of magnitude tighter than the PLAIN thresholds (3.1e-3):
# the native kernel dequantizes inline in fp32, so -- unlike the PLAIN path,
# which materializes BF16 K/V -- both sides consume numerically identical
# values (A step proves dequant exactness) and the residual is fp32
# accumulation order only. MHA path only.
MXFP4_TRITON_FROZEN_REL_L2 = 3.5e-7
MXFP4_TRITON_FROZEN_COSINE = 0.9999997
MXFP4_TRITON_FROZEN_NORM_RATIO = (0.9999998, 1.0000002)

# Grouped path (GQA/MQA, characterized separately): the grouped kernel serves
# the whole query group with tl.dot over bf16 tiles (stock-kernel semantics,
# and the K/V unpack is shared across the group instead of re-read per query
# head). The bf16 cast is lossless for dequantized E2M1 values; the residual
# comes from p->bf16 rounding before the PV dot plus the tensor-core
# accumulation order. 20-seed worst: rel_l2 1.45e-3 (splits1), cosine
# 0.9999989, |norm_ratio - 1| 4.8e-4 (mqa_hd64) -- same magnitude as the
# FlashInfer PLAIN frozen bound (2.3e-3), as expected.
MXFP4_TRITON_GROUPED_FROZEN_REL_L2 = 1.85e-3
MXFP4_TRITON_GROUPED_FROZEN_COSINE = 0.9999986
MXFP4_TRITON_GROUPED_FROZEN_NORM_RATIO = (0.9994, 1.0006)

_CHARACTERIZE = os.environ.get("SGLANG_MXFP4_TRITON_CHARACTERIZE", "0") == "1"
_CHARACTERIZE_SEEDS = 20

_DEVICE = "cuda"


def _assert_caps(testcase, metrics, context):
    testcase.assertLessEqual(
        metrics["rel_l2"],
        REL_L2_CAP,
        f"[{context}] rel_l2 {metrics['rel_l2']:.3e} > cap {REL_L2_CAP} "
        f"({format_metrics(metrics)})",
    )
    testcase.assertGreaterEqual(
        metrics["cosine"],
        COSINE_CAP,
        f"[{context}] cosine {metrics['cosine']:.6f} < cap {COSINE_CAP} "
        f"({format_metrics(metrics)})",
    )
    lo, hi = NORM_RATIO_RANGE
    testcase.assertTrue(
        lo <= metrics["norm_ratio"] <= hi,
        f"[{context}] norm_ratio {metrics['norm_ratio']:.5f} outside "
        f"[{lo}, {hi}] ({format_metrics(metrics)})",
    )


def _assert_frozen(testcase, metrics, context, grouped=False):
    rel_cap = MXFP4_TRITON_GROUPED_FROZEN_REL_L2 if grouped else MXFP4_TRITON_FROZEN_REL_L2
    cos_cap = MXFP4_TRITON_GROUPED_FROZEN_COSINE if grouped else MXFP4_TRITON_FROZEN_COSINE
    lo, hi = (
        MXFP4_TRITON_GROUPED_FROZEN_NORM_RATIO
        if grouped
        else MXFP4_TRITON_FROZEN_NORM_RATIO
    )
    testcase.assertLessEqual(
        metrics["rel_l2"],
        rel_cap,
        f"[{context}] rel_l2 {metrics['rel_l2']:.3e} > frozen {rel_cap} "
        f"({format_metrics(metrics)})",
    )
    testcase.assertGreaterEqual(
        metrics["cosine"],
        cos_cap,
        f"[{context}] cosine {metrics['cosine']:.6f} < frozen {cos_cap} "
        f"({format_metrics(metrics)})",
    )
    testcase.assertTrue(
        lo <= metrics["norm_ratio"] <= hi,
        f"[{context}] norm_ratio {metrics['norm_ratio']:.5f} outside frozen "
        f"[{lo}, {hi}] ({format_metrics(metrics)})",
    )


# ---------------------------------------------------------------------------
# A step: dequant micro-kernel, element-exact vs the OCP oracle
# ---------------------------------------------------------------------------


@unittest.skipUnless(_HAS_REQUIREMENTS, "CUDA is required")
class TestMxfp4DequantKernel(CustomTestCase):
    def _check(self, packed, scales, logical_dim, out_dtype=torch.float32):
        kernel = mxfp4_dequant_fwd(packed, scales, logical_dim, out_dtype=out_dtype)
        oracle = mxfp4_dequantize_reference(
            packed, scales, logical_dim=logical_dim, dtype=out_dtype
        )
        self.assertEqual(kernel.shape, oracle.shape)
        # NaN-aware exact comparison (torch.equal has no equal_nan kwarg).
        same = (kernel == oracle) | (kernel.isnan() & oracle.isnan())
        self.assertTrue(
            bool(same.all()),
            f"dequant mismatch (logical_dim={logical_dim}, dtype={out_dtype}): "
            f"max diff {(kernel.float() - oracle.float()).abs().max().item()}",
        )

    def test_all_e2m1_codes_scale_one(self):
        # All 16 nibble codes (8 magnitudes x 2 signs) at scale 2^0.
        codes = torch.arange(16, dtype=torch.uint8)
        packed = (codes[0::2] | (codes[1::2] << 4)).view(1, 1, 8)
        scales = torch.full((1, 1, 1), 127, dtype=torch.uint8)
        self._check(packed.to(_DEVICE), scales.to(_DEVICE), logical_dim=16)

    def test_scale_exponent_boundaries(self):
        # E8M0 normal fp32 range: 2^-126 (byte 1) .. 2^127 (byte 254). One
        # 32-element block per exponent, dim 256. Byte 0 (2^-127) is covered
        # separately below: its scale is an fp32 SUBNORMAL, where the oracle's
        # ``torch.exp2`` result carries a 1-ULP rounding error while the
        # kernel's bit construction is exact, so element-exactness cannot
        # hold for nonzero elements. That adversarial byte pattern (scale
        # 2^-127 with NONZERO elements) is never produced by the codec,
        # which writes byte 0 only for all-zero blocks.
        g = torch.Generator().manual_seed(20260903)
        codes = torch.randint(0, 256, (8 * 32,), generator=g, dtype=torch.uint8)
        packed = (codes[0::2] | (codes[1::2] << 4)).view(1, 1, 128)
        scales = torch.tensor(
            [[1, 2, 126, 127, 128, 129, 253, 254]], dtype=torch.uint8
        ).view(1, 1, 8)
        self._check(packed.to(_DEVICE), scales.to(_DEVICE), logical_dim=256)

    def test_min_scale_with_zero_data_production_pairing(self):
        # Production pairing for scale byte 0 (amax == 0): all-zero E2M1
        # data, dequantizing to exact zeros on both sides.
        packed = torch.zeros((1, 1, 4), dtype=torch.uint8)
        scales = torch.zeros((1, 1, 1), dtype=torch.uint8)
        self._check(packed.to(_DEVICE), scales.to(_DEVICE), logical_dim=8)

    def test_nan_scale_block_propagates(self):
        # Byte 0xFF = E8M0 NaN: the whole block dequantizes to NaN in both
        # the oracle and the kernel (exp2(128) = +inf, 0 * inf = NaN).
        g = torch.Generator().manual_seed(7)
        codes = torch.randint(0, 256, (2 * 32,), generator=g, dtype=torch.uint8)
        packed = (codes[0::2] | (codes[1::2] << 4)).view(1, 1, 32)
        scales = torch.tensor([[127, 255]], dtype=torch.uint8).view(1, 1, 2)
        kernel = mxfp4_dequant_fwd(
            packed.to(_DEVICE), scales.to(_DEVICE), 64, out_dtype=torch.float32
        )
        self.assertTrue(torch.isnan(kernel[0, 0, 32:]).all())
        self._check(packed.to(_DEVICE), scales.to(_DEVICE), logical_dim=64)

    def test_zero_block(self):
        # All-zero data with an all-zero scale byte (2^-127): 0 * tiny = 0.
        packed = torch.zeros((2, 2, 16), dtype=torch.uint8)
        scales = torch.zeros((2, 2, 1), dtype=torch.uint8)
        self._check(packed.to(_DEVICE), scales.to(_DEVICE), logical_dim=32)

    def test_random_bit_exact_fp32_and_bf16(self):
        g = torch.Generator(device=_DEVICE).manual_seed(11)
        for logical_dim in (64, 128, 256):
            for heads in (1, 4):
                raw = torch.randn(
                    (9, heads, logical_dim), generator=g, device=_DEVICE
                ).to(torch.bfloat16)
                packed, scales = mxfp4_quantize_reference(raw)
                self._check(packed, scales, logical_dim, torch.float32)
                self._check(packed, scales, logical_dim, torch.bfloat16)

    def test_partial_block_and_odd_dim(self):
        g = torch.Generator(device=_DEVICE).manual_seed(13)
        for logical_dim in (33, 47, 48, 63):
            raw = torch.randn((5, 2, logical_dim), generator=g, device=_DEVICE)
            packed, scales = mxfp4_quantize_reference(raw)
            self._check(packed, scales, logical_dim, torch.float32)


# ---------------------------------------------------------------------------
# B step: decode differential harness
# ---------------------------------------------------------------------------


def _build_req_locs(layout, seq_lens, num_slots, seed):
    """Physical slot lists per request. Slot 0 stays reserved (production
    contract A1). Page-size > 1 does not constrain the layouts: the kernel's
    page/tok address math is an identity for these buffers, which is exactly
    what the page_size cases below exercise."""
    total = sum(seq_lens)
    if num_slots < total + 2:
        raise ValueError("slot pool too small")
    g = torch.Generator().manual_seed(seed)

    def _split(pool):
        reqs, cur = [], 0
        for s in seq_lens:
            reqs.append(pool[cur : cur + s])
            cur += s
        return reqs

    if layout == "contiguous":
        return _split(list(range(1, 1 + total)))
    if layout == "shuffled_pages":
        perm = torch.randperm(num_slots - 1, generator=g).tolist()
        return _split([i + 1 for i in perm[:total]])
    if layout == "interleaved_pages":
        perm = torch.randperm(num_slots - 1, generator=g).tolist()
        pool = [i + 1 for i in perm]
        reqs = [[] for _ in seq_lens]
        idx = 0
        for slot in pool:
            while idx < len(reqs) and len(reqs[idx]) >= seq_lens[idx]:
                idx += 1
            if idx >= len(reqs):
                break
            reqs[idx].append(slot)
        if any(len(r) != s for r, s in zip(reqs, seq_lens)):
            raise ValueError("interleaved layout ran out of slots")
        return reqs
    if layout == "non_monotonic":
        reqs = _split(list(range(1, 1 + total)))
        return [list(reversed(r)) for r in reqs]
    raise ValueError(f"unknown loc layout {layout!r}")


def _run_decode_case(
    testcase,
    *,
    num_q_heads,
    num_kv_heads,
    head_dim,
    page_size,
    prefix_lens,
    layout="shuffled_pages",
    seed=20260903,
    zero_first_request=False,
    max_kv_splits=8,
    assert_level="frozen",
):
    torch.manual_seed(seed)
    seq_lens = [p + 1 for p in prefix_lens]  # decode: current token included
    batch = len(seq_lens)
    total = sum(seq_lens)
    num_slots = total + page_size + 16

    packed_dim = head_dim // 2
    num_blocks = head_dim // 32
    scaling = head_dim**-0.5

    k_pool = torch.zeros((num_slots, num_kv_heads, packed_dim), dtype=torch.uint8, device=_DEVICE)
    v_pool = torch.zeros((num_slots, num_kv_heads, packed_dim), dtype=torch.uint8, device=_DEVICE)
    ks_pool = torch.zeros((num_slots, num_kv_heads, num_blocks), dtype=torch.uint8, device=_DEVICE)
    vs_pool = torch.zeros((num_slots, num_kv_heads, num_blocks), dtype=torch.uint8, device=_DEVICE)

    req_locs = _build_req_locs(layout, seq_lens, num_slots, seed)
    req_to_token = torch.zeros((batch, max(seq_lens)), dtype=torch.int64, device=_DEVICE)
    for i, locs in enumerate(req_locs):
        locs_t = torch.tensor(locs, dtype=torch.int64, device=_DEVICE)
        req_to_token[i, : len(locs)] = locs_t
        k_logical = torch.randn(
            (len(locs), num_kv_heads, head_dim), device=_DEVICE
        ).to(torch.bfloat16)
        v_logical = torch.randn(
            (len(locs), num_kv_heads, head_dim), device=_DEVICE
        ).to(torch.bfloat16)
        if zero_first_request and i == 0:
            # amax == 0 blocks: scale byte 0 (2^-127), all-zero E2M1 data.
            k_logical = torch.zeros_like(k_logical)
            v_logical = torch.zeros_like(v_logical)
        k_packed, k_scales = mxfp4_quantize_reference(k_logical)
        v_packed, v_scales = mxfp4_quantize_reference(v_logical)
        k_pool[locs_t] = k_packed
        ks_pool[locs_t] = k_scales
        v_pool[locs_t] = v_packed
        vs_pool[locs_t] = v_scales

    q = torch.randn((batch, num_q_heads, head_dim), device=_DEVICE).to(torch.bfloat16)
    o = torch.zeros((batch, num_q_heads, head_dim), dtype=torch.float32, device=_DEVICE)

    kv_indptr = torch.tensor(
        [0] + list(accumulate(seq_lens)), dtype=torch.int32, device=_DEVICE
    )
    kv_indices = torch.tensor(
        [loc for locs in req_locs for loc in locs], dtype=torch.int32, device=_DEVICE
    )
    mxfp4_decode_attention_fwd(
        q,
        k_pool,
        v_pool,
        ks_pool,
        vs_pool,
        o,
        kv_indptr,
        kv_indices,
        scaling,
        page_size=page_size,
        max_kv_splits=max_kv_splits,
    )

    refs = [
        torch_mxfp4_radix_decode_reference(
            q[i],
            k_pool,
            v_pool,
            ks_pool,
            vs_pool,
            req_to_token,
            i,
            seq_lens[i],
            scaling=scaling,
            logical_dim=head_dim,
        )
        for i in range(batch)
    ]
    ref = torch.stack(refs, dim=0)

    metrics = decode_output_metrics(o, ref)
    grouped = num_q_heads != num_kv_heads
    if assert_level == "frozen":
        _assert_frozen(testcase, metrics, f"{layout}_p{page_size}_hd{head_dim}", grouped)
    else:
        _assert_caps(testcase, metrics, f"{layout}_p{page_size}_hd{head_dim}")
    return metrics


# (name, q_heads, kv_heads, head_dim, page_size, prefix_lens, layout, kwargs)
_DECODE_CASES = (
    ("mha_hd64_p1_contiguous", 4, 4, 64, 1, (31,), "contiguous", {}),
    ("mha_hd64_p16_boundary", 4, 4, 64, 16, (14, 15, 16), "shuffled_pages", {}),
    ("gqa_hd64_p16_boundary", 4, 2, 64, 16, (14, 15, 16), "shuffled_pages", {}),
    ("mqa_hd64_p16_bsz1", 4, 1, 64, 16, (7,), "shuffled_pages", {}),
    ("gqa_hd128_p16", 8, 4, 128, 16, (15, 16, 17), "shuffled_pages", {}),
    ("qwen_24_4_256_p16", 24, 4, 256, 16, (31,), "shuffled_pages", {}),
    ("qwen_24_4_256_p1", 24, 4, 256, 1, (31,), "contiguous", {}),
    ("qwen_24_4_256_p32_interleaved", 24, 4, 256, 32, (31, 33), "interleaved_pages", {}),
    ("qwen_24_4_256_p64_nonmono", 24, 4, 256, 64, (31, 33), "non_monotonic", {}),
    ("qwen_24_4_256_p128_boundary", 24, 4, 256, 128, (127,), "shuffled_pages", {}),
    ("qwen_24_4_256_seq1", 24, 4, 256, 16, (0,), "contiguous", {}),
    ("qwen_24_4_256_multisplit_long", 24, 4, 256, 16, (255,), "shuffled_pages", {}),
    ("qwen_24_4_256_mixed_batch", 24, 4, 256, 16, (14, 15, 16), "shuffled_pages", {}),
    ("qwen_24_4_256_zero_req", 24, 4, 256, 16, (15, 16), "shuffled_pages", {"zero_first_request": True}),
    ("qwen_24_4_256_splits1", 24, 4, 256, 16, (255,), "shuffled_pages", {"max_kv_splits": 1}),
)


@unittest.skipUnless(_HAS_REQUIREMENTS, "CUDA is required")
class TestTritonMxfp4NativeDecode(CustomTestCase):
    def test_triton_mxfp4_native_decode_cases(self):
        for (
            name,
            q_heads,
            kv_heads,
            head_dim,
            page_size,
            prefix_lens,
            layout,
            kwargs,
        ) in _DECODE_CASES:
            with self.subTest(case=name):
                metrics = _run_decode_case(
                    self,
                    num_q_heads=q_heads,
                    num_kv_heads=kv_heads,
                    head_dim=head_dim,
                    page_size=page_size,
                    prefix_lens=prefix_lens,
                    layout=layout,
                    **kwargs,
                )
                print(f"[mxfp4-triton-decode] {name}: {format_metrics(metrics)}")


@unittest.skipUnless(
    _HAS_REQUIREMENTS and _CHARACTERIZE,
    "CUDA required; set SGLANG_MXFP4_TRITON_CHARACTERIZE=1",
)
class TestTritonMxfp4NativeCharacterization(CustomTestCase):
    """20 fixed seeds over the decode matrix; prints worst-case metrics so the
    MXFP4_TRITON_FROZEN_* thresholds stay evidence-based (worst * 1.25)."""

    SEEDS = list(range(20260903, 20260903 + _CHARACTERIZE_SEEDS))

    def test_characterize_20_seeds(self):
        worst = {}
        for (
            name,
            q_heads,
            kv_heads,
            head_dim,
            page_size,
            prefix_lens,
            layout,
            kwargs,
        ) in _DECODE_CASES:
            w = {"rel_l2": 0.0, "max_abs": 0.0, "cosine": 1.0, "norm_ratio": 1.0}
            for seed in self.SEEDS:
                with contextlib.redirect_stdout(io.StringIO()):
                    metrics = _run_decode_case(
                        self,
                        num_q_heads=q_heads,
                        num_kv_heads=kv_heads,
                        head_dim=head_dim,
                        page_size=page_size,
                        prefix_lens=prefix_lens,
                        layout=layout,
                        seed=seed,
                        assert_level="caps",
                        **kwargs,
                    )
                w["rel_l2"] = max(w["rel_l2"], metrics["rel_l2"])
                w["max_abs"] = max(w["max_abs"], metrics["max_abs"])
                w["cosine"] = min(w["cosine"], metrics["cosine"])
                ratio = metrics["norm_ratio"]
                w["norm_ratio"] = (
                    ratio
                    if abs(ratio - 1.0) > abs(w["norm_ratio"] - 1.0)
                    else w["norm_ratio"]
                )
            worst[name] = w
            print(f"[mxfp4-triton-characterize] {name}: worst {w}")
        print("\n[mxfp4-triton-characterize] frozen-threshold proposal (worst * 1.25):")
        for name, w in worst.items():
            print(
                f"  {name}: rel_l2<={w['rel_l2'] * 1.25:.2e} "
                f"cos>={1 - (1 - w['cosine']) * 1.25:.6f} "
                f"norm_ratio={w['norm_ratio']:.5f}"
            )


@unittest.skipUnless(_HAS_REQUIREMENTS, "CUDA is required")
class TestMxfp4TritonCudaGraphSafety(CustomTestCase):
    """Both L3 kernels must capture into a CUDA graph and replay
    deterministically: outputs byte-identical across replays AND identical to
    the eager run (same deterministic kernel, same static buffers). This is
    the kernel-level half of the --disable-cuda-graph lift; the service-level
    half is the server smoke test in the L3 acceptance record."""

    def _capture(self, fn):
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            fn()  # warmup: triton JIT must not compile during capture
        torch.cuda.current_stream().wait_stream(s)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            fn()
        return g

    def test_decode_kernel_capture_replay(self):
        torch.manual_seed(20260905)
        batch, q_heads, kv_heads, head_dim, page_size = 2, 8, 4, 64, 16
        seq_lens = (37, 96)
        slots = sum(seq_lens) + page_size + 8
        scaling = head_dim**-0.5

        packed_dim, num_blocks = head_dim // 2, head_dim // 32
        k_pool = torch.randint(0, 256, (slots, kv_heads, packed_dim), dtype=torch.uint8, device=_DEVICE)
        v_pool = torch.randint(0, 256, (slots, kv_heads, packed_dim), dtype=torch.uint8, device=_DEVICE)
        ks_pool = torch.randint(120, 135, (slots, kv_heads, num_blocks), dtype=torch.uint8, device=_DEVICE)
        vs_pool = torch.randint(120, 135, (slots, kv_heads, num_blocks), dtype=torch.uint8, device=_DEVICE)
        q = torch.randn((batch, q_heads, head_dim), device=_DEVICE).to(torch.bfloat16)

        max_kv_splits = 8
        splits = [
            max(1, min(max_kv_splits, (s + 31) // 32)) for s in seq_lens
        ]
        num_kv_splits = torch.tensor(splits, dtype=torch.int32, device=_DEVICE)
        attn_logits = torch.empty((batch, q_heads, max_kv_splits, head_dim), dtype=torch.float32, device=_DEVICE)
        attn_lse = torch.empty((batch, q_heads, max_kv_splits), dtype=torch.float32, device=_DEVICE)

        kv_indptr = torch.tensor(
            [0] + list(accumulate(seq_lens)), dtype=torch.int32, device=_DEVICE
        )
        total = sum(seq_lens)
        kv_indices = torch.tensor(
            list(range(1, 1 + total)), dtype=torch.int32, device=_DEVICE
        )

        def run(out):
            mxfp4_decode_attention_fwd(
                q,
                k_pool,
                v_pool,
                ks_pool,
                vs_pool,
                out,
                kv_indptr,
                kv_indices,
                scaling,
                page_size=page_size,
                max_kv_splits=max_kv_splits,
                attn_logits=attn_logits,
                attn_lse=attn_lse,
                num_kv_splits=num_kv_splits,
            )

        o_eager = torch.zeros((batch, q_heads, head_dim), dtype=torch.float32, device=_DEVICE)
        run(o_eager)

        o_graph = torch.zeros_like(o_eager)
        graph = self._capture(lambda: run(o_graph))
        graph.replay()
        out1 = o_graph.clone()
        graph.replay()
        out2 = o_graph.clone()

        self.assertTrue(torch.equal(out1, out2), "replay is not deterministic")
        self.assertTrue(
            torch.equal(out1, o_eager),
            f"graph output differs from eager: max diff "
            f"{(out1 - o_eager).abs().max().item()}",
        )

    def test_fused_write_kernel_capture_replay(self):
        from sglang.kernels.ops.quantization.mxfp4_quant import quant_store_kv_mxfp4

        torch.manual_seed(20260906)
        tokens, kv_heads, head_dim = 5, 4, 128
        slots = tokens + 8
        k = (torch.randn((tokens, kv_heads, head_dim), device=_DEVICE) * 1.5).to(torch.bfloat16)
        v = (torch.randn((tokens, kv_heads, head_dim), device=_DEVICE) * 1.5).to(torch.bfloat16)
        loc = torch.arange(1, tokens + 1, dtype=torch.int64, device=_DEVICE)

        def fresh_pools():
            return (
                torch.zeros((slots, kv_heads, head_dim // 2), dtype=torch.uint8, device=_DEVICE),
                torch.zeros((slots, kv_heads, head_dim // 2), dtype=torch.uint8, device=_DEVICE),
                torch.zeros((slots, kv_heads, head_dim // 32), dtype=torch.uint8, device=_DEVICE),
                torch.zeros((slots, kv_heads, head_dim // 32), dtype=torch.uint8, device=_DEVICE),
            )

        kd_e, vd_e, ks_e, vs_e = fresh_pools()
        quant_store_kv_mxfp4(k, v, loc, kd_e, vd_e, ks_e, vs_e)

        kd_g, vd_g, ks_g, vs_g = fresh_pools()
        graph = self._capture(
            lambda: quant_store_kv_mxfp4(k, v, loc, kd_g, vd_g, ks_g, vs_g)
        )
        graph.replay()
        snap1 = (kd_g.clone(), vd_g.clone(), ks_g.clone(), vs_g.clone())
        graph.replay()
        snap2 = (kd_g.clone(), vd_g.clone(), ks_g.clone(), vs_g.clone())

        for a, b in zip(snap1, snap2):
            self.assertTrue(torch.equal(a, b), "fused write replay is not deterministic")
        for got, want in zip(snap1, (kd_e, vd_e, ks_e, vs_e)):
            self.assertTrue(
                torch.equal(got, want),
                "fused write graph output differs from eager bytes",
            )


if __name__ == "__main__":
    unittest.main()
