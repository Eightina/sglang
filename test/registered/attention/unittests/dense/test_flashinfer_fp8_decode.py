"""L1 golden: FlashInfer FP8 (E4M3) decode vs pure-Torch radix decode reference.

Two layers of verification (see qwen38_attention_survey_notes.md §11):

1. Cache-write contract: the real ``MHATokenToKVPool.set_kv_buffer`` with
   checkpoint-style per-tensor k/v scales must produce bit-exact FP8 bytes
   versus the Torch QDQ reference.
2. Single-layer differential: the FlashInfer decode kernel and the Torch
   reference consume the SAME final FP8 cache (history written via the pool
   with scales, current token written by the backend itself) and must agree
   within frozen thresholds.

Decode only — prefill stays on the existing FlashInfer path by design.
"""

import os
import unittest
from types import SimpleNamespace

import torch

from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.utils import is_flashinfer_available
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kits.attention_unittest.attention_methods.dense_attention import (
    DenseAttentionCase,
    build_dense_attention_fixture,
)
from sglang.test.kits.attention_unittest.attention_methods.fp8_decode_attention import (
    FP8_DTYPE,
    decode_output_metrics,
    format_metrics,
    fp8_cache_quantize_reference,
    torch_fp8_decode_dequant_first_reference,
    torch_fp8_radix_decode_reference,
)
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=30, stage="base-b", runner_config="1-gpu-large")

_HAS_REQUIREMENTS = torch.cuda.is_available() and is_flashinfer_available()

# ---------------------------------------------------------------------------
# Thresholds.
#
# Hard caps below are the plan-level upper bounds (rel L2 <= 2e-2, cosine >=
# 0.999, norm ratio in [0.98, 1.02]); they must never be relaxed to mask a
# scale-placement / GQA / page-mapping bug. FROZEN_* values are the 20-seed
# characterization worst case * 1.25 on SM120 (RTX 5090, torch 2.13.0+cu130,
# flashinfer 0.6.18; see TestFp8DecodeCharacterization):
#   worst rel_l2  = 3.08e-3 (mqa_hd64_p16_bsz1)  -> frozen 3.9e-3
#   worst cosine  = 0.99999523                   -> frozen 0.9999940
#   worst |norm_ratio - 1| = 9.4e-4              -> frozen 1.2e-3
# ---------------------------------------------------------------------------

REL_L2_CAP = 2e-2
COSINE_CAP = 0.999
NORM_RATIO_RANGE = (0.98, 1.02)

FROZEN_REL_L2 = 3.9e-3
FROZEN_COSINE = 0.9999940
FROZEN_NORM_RATIO = (0.9988, 1.0012)

_CHARACTERIZE = os.environ.get("SGLANG_FP8_DECODE_CHARACTERIZE", "0") == "1"
_CHARACTERIZE_SEEDS = 20


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_pool(num_slots, heads, head_dim, device="cuda"):
    return MHATokenToKVPool(
        size=num_slots,
        page_size=1,
        dtype=FP8_DTYPE,
        head_num=heads,
        head_dim=head_dim,
        layer_num=1,
        device=device,
        enable_memory_saver=False,
        enable_alt_stream=False,
    )


def _assert_within_caps(testcase, metrics, context):
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


def _assert_frozen(testcase, metrics, context):
    testcase.assertLessEqual(
        metrics["rel_l2"],
        FROZEN_REL_L2,
        f"[{context}] rel_l2 {metrics['rel_l2']:.3e} > frozen {FROZEN_REL_L2} "
        f"({format_metrics(metrics)})",
    )
    testcase.assertGreaterEqual(
        metrics["cosine"],
        FROZEN_COSINE,
        f"[{context}] cosine {metrics['cosine']:.6f} < frozen {FROZEN_COSINE} "
        f"({format_metrics(metrics)})",
    )
    lo, hi = FROZEN_NORM_RATIO
    testcase.assertTrue(
        lo <= metrics["norm_ratio"] <= hi,
        f"[{context}] norm_ratio {metrics['norm_ratio']:.5f} outside frozen "
        f"[{lo}, {hi}] ({format_metrics(metrics)})",
    )


def _run_flashinfer_decode_attn_core(fixture):
    """Run the real FlashInfer backend decode for the fixture's RadixAttention
    module; returns (attn_out, q, k, v) where q/k/v are the pre-attention
    projections (k/v in model dtype, pre-QDQ)."""
    from sglang.srt.model_executor.forward_context import (
        ForwardContext,
        forward_context,
    )

    module = fixture.actual_module
    forward_batch = fixture.forward_batch
    with torch.no_grad(), forward_context(
        ForwardContext(attn_backend=fixture.backend)
    ):
        fixture.backend.init_forward_metadata(forward_batch)
        q, k, v = module.project_qkv(fixture.input_hidden)
        # The backend's write path divides k/v IN PLACE; snapshot the
        # pre-write projections for the current-token QDQ check.
        k_pre, v_pre = k.clone(), v.clone()
        attn_out = module.attn(q, k, v, forward_batch)
    return attn_out, q, k_pre, v_pre


def _torch_reference_outputs(fixture, q, k_cache, v_cache):
    """Run both Torch references per request; returns (main, cross_check).

    Main output is stacked over requests in fixture request order, shaped
    (num_tokens, num_q_heads * head_dim), fp32.
    """
    fb = fixture.forward_batch
    req_to_token = fixture.runner.req_to_token_pool.req_to_token
    module = fixture.actual_module
    q3 = q.view(-1, module.num_heads, module.head_dim)
    num_tokens = q.shape[0]
    outs_main = []
    outs_cross = []
    for i in range(num_tokens):
        seq_len = int(fb.seq_lens[i])
        main = torch_fp8_radix_decode_reference(
            q3[i],
            k_cache,
            v_cache,
            req_to_token,
            int(fb.req_pool_indices[i]),
            seq_len,
            scaling=module.attn.scaling,
            k_scale=fixture.k_scale,
            v_scale=fixture.v_scale,
        )
        cross = torch_fp8_decode_dequant_first_reference(
            q3[i],
            k_cache,
            v_cache,
            req_to_token,
            int(fb.req_pool_indices[i]),
            seq_len,
            scaling=module.attn.scaling,
            k_scale=fixture.k_scale,
            v_scale=fixture.v_scale,
        )
        outs_main.append(main.reshape(-1))
        outs_cross.append(cross.reshape(-1))
    return torch.stack(outs_main, dim=0), torch.stack(outs_cross, dim=0)


def _run_decode_case(
    testcase,
    case,
    *,
    head_dim,
    hidden_size,
    max_context_len,
    k_scale,
    v_scale,
    seed=None,
    loc_layout="shuffled_pages",
    assert_level="frozen",
):
    """Build an FP8 fixture, run FlashInfer decode, compare against the Torch
    reference on the same final cache. Returns the metrics dict."""
    fixture = build_dense_attention_fixture(
        testcase,
        case,
        head_dim=head_dim,
        hidden_size=hidden_size,
        max_context_len=max_context_len,
        dtype=torch.bfloat16,
        kv_cache_dtype=torch.float8_e4m3fn,
        kv_cache_dtype_str="fp8_e4m3",
        k_scale=k_scale,
        v_scale=v_scale,
        seed=seed,
        loc_layout=loc_layout,
    )
    attn_out, q, k, _v = _run_flashinfer_decode_attn_core(fixture)

    pool = fixture.runner.token_to_kv_pool
    k_cache = pool.get_key_buffer(0)
    v_cache = pool.get_value_buffer(0)

    # Current token must have been written through the same QDQ as history:
    # its slot bytes must match the reference quantize of the bf16 projection.
    fb = fixture.forward_batch
    k3 = k.view(-1, case.num_kv_heads, head_dim)
    for i in range(case.batch_size):
        loc = int(fb.out_cache_loc[i])
        expected_bits = fp8_cache_quantize_reference(
            k3[i], fixture.k_scale
        ).view(torch.uint8)
        actual_bits = k_cache[loc].view(torch.uint8)
        testcase.assertTrue(
            torch.equal(actual_bits, expected_bits),
            f"[{case.name}] current-token K slot {loc} bytes differ from QDQ ref",
        )

    ref_main, ref_cross = _torch_reference_outputs(fixture, q, k_cache, v_cache)

    # The two Torch implementations are the same math in exact arithmetic.
    torch.testing.assert_close(ref_main, ref_cross, atol=1e-5, rtol=1e-5)

    metrics = decode_output_metrics(attn_out, ref_main)
    if assert_level == "frozen":
        _assert_frozen(testcase, metrics, case.name)
    else:
        _assert_within_caps(testcase, metrics, case.name)
    return metrics


# ---------------------------------------------------------------------------
# 1) FP8 cache-write contract (bit-exact, no tolerance)
# ---------------------------------------------------------------------------


@unittest.skipUnless(_HAS_REQUIREMENTS, "CUDA + flashinfer are required")
class TestFp8KvWriteContract(CustomTestCase):
    HEADS = 4
    HEAD_DIM = 64
    NUM_TOKENS = 8
    TOTAL_SLOTS = 64

    def _pool_and_layer(self):
        pool = _make_pool(self.TOTAL_SLOTS, self.HEADS, self.HEAD_DIM)
        layer = SimpleNamespace(layer_id=0)
        return pool, layer

    def _write(self, pool, layer, k, v, loc, k_scale, v_scale):
        pool.set_kv_buffer(layer, loc, k, v, k_scale, v_scale)

    def _nonzero_locs(self, device, n):
        """Slot 0 is the reserved CUDA-graph padding slot (store_cache's
        reserved_skip_index=0 skips writes to it), so real locs start at 1
        exactly like the allocator/fixture `_token_loc` (>= page_size)."""
        return torch.randperm(self.TOTAL_SLOTS - 1, device=device)[:n] + 1

    def _assert_slot_bits(self, pool, loc, expected_fp8, tag):
        buffer = pool.k_buffer[0] if tag == "K" else pool.v_buffer[0]
        actual_bits = buffer[loc].view(torch.uint8)
        expected_bits = expected_fp8.view(torch.uint8)
        mismatch = (actual_bits != expected_bits).float().mean().item()
        self.assertEqual(
            mismatch,
            0.0,
            f"{tag}: {mismatch:.4%} of FP8 bytes differ from QDQ reference",
        )

    def test_random_bf16_write_bit_exact_with_distinct_scales(self):
        torch.manual_seed(7)
        dev = "cuda"
        k = torch.randn(
            (self.NUM_TOKENS, self.HEADS, self.HEAD_DIM),
            dtype=torch.bfloat16,
            device=dev,
        )
        v = torch.randn(
            (self.NUM_TOKENS, self.HEADS, self.HEAD_DIM),
            dtype=torch.bfloat16,
            device=dev,
        )
        # Deliberately very different scales: a swap or double-apply shows up
        # as different FP8 codes for most elements.
        k_scale, v_scale = 0.05, 0.5
        loc = self._nonzero_locs(dev, self.NUM_TOKENS)
        pool, layer = self._pool_and_layer()
        # NOTE: set_kv_buffer divides cache_k/cache_v IN PLACE (production
        # semantics), so snapshot the pre-write inputs for the reference.
        k0, v0 = k.clone(), v.clone()
        self._write(pool, layer, k, v, loc, k_scale, v_scale)
        self._assert_slot_bits(
            pool, loc, fp8_cache_quantize_reference(k0, k_scale), "K"
        )
        self._assert_slot_bits(
            pool, loc, fp8_cache_quantize_reference(v0, v_scale), "V"
        )
        # The in-place divide is itself production semantics: lock it in.
        self.assertFalse(
            torch.equal(k, k0), "set_kv_buffer must divide cache_k in place"
        )

    def test_zero_write_is_all_zero_bytes(self):
        dev = "cuda"
        k = torch.zeros(
            (self.NUM_TOKENS, self.HEADS, self.HEAD_DIM),
            dtype=torch.bfloat16,
            device=dev,
        )
        v = torch.zeros_like(k)
        loc = torch.arange(1, self.NUM_TOKENS + 1, device=dev, dtype=torch.int64)
        pool, layer = self._pool_and_layer()
        self._write(pool, layer, k, v, loc, 0.0275, 0.0245)
        self.assertEqual(pool.k_buffer[0][loc].abs().sum().item(), 0.0)
        self.assertEqual(pool.v_buffer[0][loc].abs().sum().item(), 0.0)

    def test_near_e4m3_max_finite(self):
        torch.manual_seed(11)
        dev = "cuda"
        e4m3_max = torch.finfo(FP8_DTYPE).max  # 448.0
        k = torch.randn(
            (self.NUM_TOKENS, self.HEADS, self.HEAD_DIM),
            dtype=torch.bfloat16,
            device=dev,
        )
        # Values at exactly the representable maximum after descale.
        k[0, 0, 0] = e4m3_max * 0.0275
        k[1, 0, 0] = -e4m3_max * 0.0275
        v = torch.randn_like(k)
        k0, v0 = k.clone(), v.clone()
        loc = torch.arange(1, self.NUM_TOKENS + 1, device=dev, dtype=torch.int64)
        pool, layer = self._pool_and_layer()
        self._write(pool, layer, k, v, loc, 0.0275, 0.0245)
        self._assert_slot_bits(
            pool, loc, fp8_cache_quantize_reference(k0, 0.0275), "K"
        )
        self._assert_slot_bits(
            pool, loc, fp8_cache_quantize_reference(v0, 0.0245), "V"
        )

    def test_unwritten_slots_untouched(self):
        torch.manual_seed(13)
        dev = "cuda"
        k = torch.randn(
            (2, self.HEADS, self.HEAD_DIM), dtype=torch.bfloat16, device=dev
        )
        v = torch.randn_like(k)
        k0, v0 = k.clone(), v.clone()
        loc = torch.tensor([3, 40], device=dev, dtype=torch.int64)
        pool, layer = self._pool_and_layer()
        self._write(pool, layer, k, v, loc, 0.0275, 0.0245)
        self._assert_slot_bits(
            pool, loc, fp8_cache_quantize_reference(k0, 0.0275), "K"
        )
        self._assert_slot_bits(
            pool, loc, fp8_cache_quantize_reference(v0, 0.0245), "V"
        )
        # Buffers carry size + page_size rows (row 0 = reserved padding).
        untouched = torch.ones(
            pool.k_buffer[0].shape[0], dtype=torch.bool, device=dev
        )
        untouched[loc] = False
        self.assertEqual(pool.k_buffer[0][untouched].abs().sum().item(), 0.0)
        self.assertEqual(pool.v_buffer[0][untouched].abs().sum().item(), 0.0)

    def test_no_scale_means_plain_cast(self):
        torch.manual_seed(17)
        dev = "cuda"
        k = torch.randn(
            (2, self.HEADS, self.HEAD_DIM), dtype=torch.bfloat16, device=dev
        )
        v = torch.randn_like(k)
        k0, v0 = k.clone(), v.clone()
        loc = torch.tensor([5, 9], device=dev, dtype=torch.int64)
        pool, layer = self._pool_and_layer()
        self._write(pool, layer, k, v, loc, None, None)
        self._assert_slot_bits(pool, loc, fp8_cache_quantize_reference(k0), "K")
        self._assert_slot_bits(pool, loc, fp8_cache_quantize_reference(v0), "V")

    def test_reserved_padding_slot_zero_write_is_skipped(self):
        """Slot 0 is the reserved CUDA-graph padding slot; store_cache's
        reserved_skip_index=0 skips writes targeting it (production locs are
        never 0). Lock the semantics so a kernel change cannot silently
        start writing into the padding slot."""
        torch.manual_seed(19)
        dev = "cuda"
        k = torch.randn(
            (1, self.HEADS, self.HEAD_DIM), dtype=torch.bfloat16, device=dev
        )
        v = torch.randn_like(k)
        loc = torch.tensor([0], device=dev, dtype=torch.int64)
        pool, layer = self._pool_and_layer()
        self._write(pool, layer, k, v, loc, 0.0275, 0.0245)
        self.assertEqual(pool.k_buffer[0][0].abs().sum().item(), 0.0)
        self.assertEqual(pool.v_buffer[0][0].abs().sum().item(), 0.0)


# ---------------------------------------------------------------------------
# 2) Single-layer FlashInfer FP8 decode differential
# ---------------------------------------------------------------------------


@unittest.skipUnless(_HAS_REQUIREMENTS, "CUDA + flashinfer are required")
class TestFlashInferFp8DecodeGolden(CustomTestCase):
    # (name, num_heads, num_kv_heads, head_dim, hidden, page_size,
    #  prefix_lens, max_context_len, k_scale, v_scale, loc_layout)
    CASES = (
        ("mha_hd64_p16_boundary", 4, 4, 64, 256, 16, (14, 15, 16), 64, 0.0275, 0.0245, "shuffled_pages"),
        ("gqa_hd64_p16_boundary", 4, 2, 64, 256, 16, (14, 15, 16), 64, 0.0275, 0.0245, "shuffled_pages"),
        ("mqa_hd64_p16_bsz1", 4, 1, 64, 256, 16, (7,), 64, 0.0275, 0.0245, "shuffled_pages"),
        ("qwen_24_4_256_short", 24, 4, 256, 512, 16, (31,), 64, 0.0275, 0.0245, "shuffled_pages"),
        ("qwen_24_4_256_page128_boundary", 24, 4, 256, 512, 128, (127,), 256, 0.0275, 0.0245, "shuffled_pages"),
        ("qwen_24_4_256_page1", 24, 4, 256, 512, 1, (31,), 64, 0.0275, 0.0245, "shuffled_pages"),
        ("qwen_24_4_256_scale1", 24, 4, 256, 512, 16, (15, 16, 17), 64, 1.0, 1.0, "shuffled_pages"),
        ("qwen_24_4_256_contiguous", 24, 4, 256, 512, 16, (31,), 64, 0.0275, 0.0245, "contiguous"),
        ("qwen_24_4_256_interleaved", 24, 4, 256, 512, 16, (31, 33), 64, 0.0275, 0.0245, "interleaved_pages"),
        ("qwen_24_4_256_mixed_batch", 24, 4, 256, 512, 16, (14, 15, 16), 64, 0.0275, 0.0245, "shuffled_pages"),
    )

    def _run(self, spec, **overrides):
        (
            name,
            num_heads,
            num_kv_heads,
            head_dim,
            hidden,
            page_size,
            prefix_lens,
            max_context_len,
            k_scale,
            v_scale,
            loc_layout,
        ) = spec
        case = DenseAttentionCase(
            name=name,
            backend="flashinfer",
            forward_mode=ForwardMode.DECODE,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            page_size=page_size,
            prefix_lens=prefix_lens,
        )
        kwargs = dict(
            head_dim=head_dim,
            hidden_size=hidden,
            max_context_len=max_context_len,
            k_scale=k_scale,
            v_scale=v_scale,
            loc_layout=loc_layout,
        )
        kwargs.update(overrides)
        return _run_decode_case(self, case, **kwargs)

    def test_flashinfer_fp8_decode_cases(self):
        for spec in self.CASES:
            with self.subTest(case=spec[0]):
                metrics = self._run(spec)
                print(f"[fp8-decode-golden] {spec[0]}: {format_metrics(metrics)}")

    def test_torch_reference_variants_agree(self):
        # Covered inside _run_decode_case (assert_close atol=1e-5); kept as an
        # explicit named test for failure attribution.
        spec = self.CASES[3]
        self._run(spec)


# ---------------------------------------------------------------------------
# 3) Characterization (manual/slow): freeze thresholds from measurement
# ---------------------------------------------------------------------------


@unittest.skipUnless(
    _HAS_REQUIREMENTS and _CHARACTERIZE,
    "CUDA + flashinfer required; set SGLANG_FP8_DECODE_CHARACTERIZE=1",
)
class TestFp8DecodeCharacterization(CustomTestCase):
    """Runs the differential across 20 fixed seeds and prints the worst-case
    metric per case so the FROZEN_* thresholds stay evidence-based."""

    SEEDS = list(range(20260801, 20260801 + 20))

    def test_characterize_20_seeds(self):
        worst = {}
        for spec in TestFlashInferFp8DecodeGolden.CASES:
            w = {"rel_l2": 0.0, "max_abs": 0.0, "cosine": 1.0, "norm_ratio": 1.0}
            for seed in self.SEEDS:
                metrics = self._run_quiet(spec, seed)
                w["rel_l2"] = max(w["rel_l2"], metrics["rel_l2"])
                w["max_abs"] = max(w["max_abs"], metrics["max_abs"])
                w["cosine"] = min(w["cosine"], metrics["cosine"])
                ratio = metrics["norm_ratio"]
                w["norm_ratio"] = (
                    ratio
                    if abs(ratio - 1.0) > abs(w["norm_ratio"] - 1.0)
                    else w["norm_ratio"]
                )
            worst[spec[0]] = w
            print(f"[characterize] {spec[0]}: worst {w}")
        print("\n[characterize] frozen-threshold proposal (worst * 1.25):")
        for name, w in worst.items():
            print(
                f"  {name}: rel_l2<={w['rel_l2'] * 1.25:.2e} "
                f"cos>={1 - (1 - w['cosine']) * 1.25:.6f}"
            )

    def _run_quiet(self, spec, seed):
        import contextlib
        import io

        (
            name,
            num_heads,
            num_kv_heads,
            head_dim,
            hidden,
            page_size,
            prefix_lens,
            max_context_len,
            k_scale,
            v_scale,
            loc_layout,
        ) = spec
        case = DenseAttentionCase(
            name=f"{name}_s{seed}",
            backend="flashinfer",
            forward_mode=ForwardMode.DECODE,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            page_size=page_size,
            prefix_lens=prefix_lens,
        )
        with contextlib.redirect_stdout(io.StringIO()):
            return _run_decode_case(
                self,
                case,
                head_dim=head_dim,
                hidden_size=hidden,
                max_context_len=max_context_len,
                k_scale=k_scale,
                v_scale=v_scale,
                seed=seed,
                loc_layout=loc_layout,
                assert_level="caps",
            )


if __name__ == "__main__":
    unittest.main()
