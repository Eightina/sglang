"""L2: FlashInfer MXFP4 (block-32 E2M1 + E8M0) PLAIN path vs Torch oracle.

Three layers of verification (see qwen38_attention_survey_notes.md §11.5/§11.6):

1. Cache-write contract: the real ``MHATokenToKVPool.set_kv_buffer`` with the
   ``MXFP4KVCacheMethod`` must produce bit-exact packed E2M1 data and E8M0
   scale bytes versus the independent OCP reference. The checkpoint FP8
   k/v scales must be IGNORED (MXFP4 is self-scaling).
2. Decode differential: FlashInfer decode consumes the BF16 PLAIN materialized
   from the same packed cache that the Torch oracle reads as raw bytes, and
   both must agree within hard caps (frozen thresholds after characterization).
3. Prefill functional: the existing FlashInfer prefill consumes the same
   PLAIN BF16 cache for zero-prefix and prefix-reuse extends.

The PLAIN read materializes a whole layer's K/V as BF16; it is a correctness
path only (no CUDA graph), not a performance claim.
"""

import os
import unittest
from types import SimpleNamespace

import torch

from sglang.srt.layers.quantization.fp4_kv_cache_quant_method import MXFP4KVCacheMethod
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.utils import is_flashinfer_available
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kits.attention_unittest.attention_methods.dense_attention import (
    DenseAttentionCase,
    build_dense_attention_fixture,
)
from sglang.test.kits.attention_unittest.attention_methods.mxfp4_decode_attention import (
    torch_mxfp4_radix_decode_reference,
)
from sglang.test.kits.attention_unittest.attention_methods.fp8_decode_attention import (
    decode_output_metrics,
    format_metrics,
)
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=30, stage="base-b", runner_config="1-gpu-large")

_HAS_REQUIREMENTS = torch.cuda.is_available() and is_flashinfer_available()

# Same hard caps as the L1 FP8 golden: they bound IMPLEMENTATION equivalence
# (FlashInfer PLAIN vs Torch oracle on the SAME packed cache) and must never be
# relaxed to mask a scale/page/pack bug. Codec quality versus FP8 is reported
# separately and is not held to these numbers.
REL_L2_CAP = 2e-2
COSINE_CAP = 0.999
NORM_RATIO_RANGE = (0.98, 1.02)

# Filled in by TestMxfp4PlainCharacterization (worst * 1.25, 20 seeds,
# SM120 / torch 2.13.0+cu130 / flashinfer 0.6.18). The worst case came from
# mqa_hd64_p16_bsz1: rel_l2 2.46e-3, cosine 0.9999960, |norm_ratio-1| 4.0e-4.
# These bounds implementation equivalence only: after the BF16 PLAIN
# materialization both sides consume identical values, so the residual is the
# FlashInfer kernel's accumulation order, matching the L1 FP8 magnitude.
MXFP4_PLAIN_FROZEN_REL_L2 = 3.1e-3
MXFP4_PLAIN_FROZEN_COSINE = 0.9999950
MXFP4_PLAIN_FROZEN_NORM_RATIO = (0.99950, 1.00050)

_CHARACTERIZE = os.environ.get("SGLANG_MXFP4_PLAIN_CHARACTERIZE", "0") == "1"
_CHARACTERIZE_SEEDS = 20


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


def _assert_frozen(testcase, metrics, context):
    testcase.assertLessEqual(metrics["rel_l2"], MXFP4_PLAIN_FROZEN_REL_L2)
    testcase.assertGreaterEqual(metrics["cosine"], MXFP4_PLAIN_FROZEN_COSINE)
    lo, hi = MXFP4_PLAIN_FROZEN_NORM_RATIO
    testcase.assertTrue(
        lo <= metrics["norm_ratio"] <= hi,
        f"[{context}] norm_ratio {metrics['norm_ratio']:.5f} outside frozen "
        f"[{lo}, {hi}] ({format_metrics(metrics)})",
    )


def _run_flashinfer_attn_core(fixture):
    """Run the real FlashInfer backend for the fixture's RadixAttention module;
    returns (attn_out, q, k_pre, v_pre) with k/v the pre-write projections."""
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
        # The write path may mutate k/v in place depending on the pool; keep
        # pre-write projections for the current-token write contract check.
        k_pre, v_pre = k.clone(), v.clone()
        attn_out = module.attn(q, k, v, forward_batch)
    return attn_out, q, k_pre, v_pre


def _assert_current_token_write(testcase, fixture, case, head_dim, k_pre, v_pre):
    """Every input token's slot must equal the OCP reference bytes, with the
    checkpoint FP8 scales ignored entirely."""
    pool = fixture.runner.token_to_kv_pool
    fb = fixture.forward_batch
    from sglang.test.kits.attention_unittest.attention_methods.mxfp4_decode_attention import (
        mxfp4_quantize_reference,
    )

    k3 = k_pre.view(-1, case.num_kv_heads, head_dim)
    v3 = v_pre.view(-1, case.num_kv_heads, head_dim)
    for i in range(k3.shape[0]):
        loc = int(fb.out_cache_loc[i])
        exp_k, exp_ks = mxfp4_quantize_reference(k3[i : i + 1])
        exp_v, exp_vs = mxfp4_quantize_reference(v3[i : i + 1])
        testcase.assertTrue(
            torch.equal(pool.k_buffer[0][loc], exp_k[0]),
            f"[{case.name}] token {i} K packed bytes differ from OCP reference",
        )
        testcase.assertTrue(
            torch.equal(pool.k_scale_buffer[0][loc], exp_ks[0]),
            f"[{case.name}] token {i} K E8M0 bytes differ from OCP reference",
        )
        testcase.assertTrue(
            torch.equal(pool.v_buffer[0][loc], exp_v[0]),
            f"[{case.name}] token {i} V packed bytes differ from OCP reference",
        )
        testcase.assertTrue(
            torch.equal(pool.v_scale_buffer[0][loc], exp_vs[0]),
            f"[{case.name}] token {i} V E8M0 bytes differ from OCP reference",
        )


def _torch_reference_outputs(fixture, case, head_dim, q, k_raw=None, v_raw=None):
    """Per-token oracle matching what the backend actually consumed.

    Decode reads every token (including the current one) from the packed
    cache. FlashInfer ragged prefill, however, feeds the current extend chunk
    as RAW bf16 K/V and only reads the cached prefix through the PLAIN BF16
    materialization, so the extend oracle mixes both sources exactly like the
    backend.
    """
    from sglang.test.kits.attention_unittest.attention_methods.fp8_decode_attention import (
        torch_radix_decode_from_effective_kv,
    )
    from sglang.test.kits.attention_unittest.attention_methods.mxfp4_decode_attention import (
        mxfp4_dequantize_reference,
    )

    pool = fixture.runner.token_to_kv_pool
    fb = fixture.forward_batch
    req_to_token = fixture.runner.req_to_token_pool.req_to_token
    module = fixture.actual_module
    q3 = q.view(-1, module.num_heads, module.head_dim)
    ragged_extend = not case.forward_mode.is_decode() and k_raw is not None
    k_raw3 = (
        k_raw.view(-1, case.num_kv_heads, head_dim).to(torch.float32)
        if ragged_extend
        else None
    )
    v_raw3 = (
        v_raw.view(-1, case.num_kv_heads, head_dim).to(torch.float32)
        if ragged_extend
        else None
    )
    outs = []
    token = 0
    for i in range(case.batch_size):
        prefix_len = case.prefix_lens[i]
        extend_len = case.input_lens[i]
        if not ragged_extend:
            for offset in range(extend_len):
                seq_len = prefix_len + offset + 1
                outs.append(
                    torch_mxfp4_radix_decode_reference(
                        q3[token],
                        pool.k_buffer[0],
                        pool.v_buffer[0],
                        pool.k_scale_buffer[0],
                        pool.v_scale_buffer[0],
                        req_to_token,
                        int(fb.req_pool_indices[i]),
                        seq_len,
                        scaling=module.attn.scaling,
                        logical_dim=head_dim,
                    ).reshape(-1)
                )
                token += 1
            continue

        base = token
        if prefix_len:
            locs = req_to_token[int(fb.req_pool_indices[i]), :prefix_len].long()
            k_prefix = mxfp4_dequantize_reference(
                pool.k_buffer[0][locs],
                pool.k_scale_buffer[0][locs],
                logical_dim=head_dim,
                dtype=torch.float32,
            )
            v_prefix = mxfp4_dequantize_reference(
                pool.v_buffer[0][locs],
                pool.v_scale_buffer[0][locs],
                logical_dim=head_dim,
                dtype=torch.float32,
            )
        else:
            k_prefix = None
            v_prefix = None
        for offset in range(extend_len):
            k_parts = ([k_prefix] if prefix_len else []) + [
                k_raw3[base : base + offset + 1]
            ]
            v_parts = ([v_prefix] if prefix_len else []) + [
                v_raw3[base : base + offset + 1]
            ]
            out = torch_radix_decode_from_effective_kv(
                q3[base + offset],
                torch.cat(k_parts, dim=0),
                torch.cat(v_parts, dim=0),
                scaling=module.attn.scaling,
            )
            outs.append(out.reshape(-1))
        token += extend_len
    return torch.stack(outs, dim=0)


def _run_case(
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
    fixture = build_dense_attention_fixture(
        testcase,
        case,
        head_dim=head_dim,
        hidden_size=hidden_size,
        max_context_len=max_context_len,
        dtype=torch.bfloat16,
        kv_cache_dtype=torch.float4_e2m1fn_x2,
        kv_cache_dtype_str="mxfp4",
        kv_cache_quant_method=MXFP4KVCacheMethod(),
        k_scale=k_scale,
        v_scale=v_scale,
        seed=seed,
        loc_layout=loc_layout,
    )
    attn_out, q, k_pre, v_pre = _run_flashinfer_attn_core(fixture)
    _assert_current_token_write(testcase, fixture, case, head_dim, k_pre, v_pre)
    ref = _torch_reference_outputs(fixture, case, head_dim, q, k_pre, v_pre)
    metrics = decode_output_metrics(attn_out, ref)
    if assert_level == "frozen":
        _assert_frozen(testcase, metrics, case.name)
    else:
        _assert_caps(testcase, metrics, case.name)
    return metrics


@unittest.skipUnless(_HAS_REQUIREMENTS, "CUDA + flashinfer are required")
class TestFlashInferMxfp4PlainDecode(CustomTestCase):
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
        return _run_case(self, case, **kwargs)

    def test_flashinfer_mxfp4_plain_decode_cases(self):
        for spec in self.CASES:
            with self.subTest(case=spec[0]):
                metrics = self._run(spec)
                print(f"[mxfp4-plain-decode] {spec[0]}: {format_metrics(metrics)}")


@unittest.skipUnless(_HAS_REQUIREMENTS, "CUDA + flashinfer are required")
class TestFlashInferMxfp4PlainPrefill(CustomTestCase):
    """Functional: the existing FlashInfer prefill consumes the PLAIN BF16
    cache; every extend token is checked against the per-token oracle."""

    CASES = (
        ("mxfp4_extend_zero_prefix_page_edge", 24, 4, 256, 512, 16, (0,), (16,), 64, "shuffled_pages"),
        ("mxfp4_extend_prefix_reuse_ragged", 24, 4, 256, 512, 16, (31, 33), (1, 3), 64, "shuffled_pages"),
        ("mxfp4_extend_page128", 24, 4, 256, 512, 128, (0,), (17,), 256, "shuffled_pages"),
    )

    def test_flashinfer_mxfp4_plain_prefill_cases(self):
        for (
            name,
            num_heads,
            num_kv_heads,
            head_dim,
            hidden,
            page_size,
            prefix_lens,
            extend_lens,
            max_context_len,
            loc_layout,
        ) in self.CASES:
            with self.subTest(case=name):
                case = DenseAttentionCase(
                    name=name,
                    backend="flashinfer",
                    forward_mode=ForwardMode.EXTEND,
                    num_heads=num_heads,
                    num_kv_heads=num_kv_heads,
                    page_size=page_size,
                    prefix_lens=prefix_lens,
                    extend_lens=extend_lens,
                )
                metrics = _run_case(
                    self,
                    case,
                    head_dim=head_dim,
                    hidden_size=hidden,
                    max_context_len=max_context_len,
                    k_scale=0.0275,
                    v_scale=0.0245,
                    loc_layout=loc_layout,
                )
                print(f"[mxfp4-plain-prefill] {name}: {format_metrics(metrics)}")


@unittest.skipUnless(
    _HAS_REQUIREMENTS and _CHARACTERIZE,
    "CUDA + flashinfer required; set SGLANG_MXFP4_PLAIN_CHARACTERIZE=1",
)
class TestMxfp4PlainCharacterization(CustomTestCase):
    """20 fixed seeds over the decode matrix; prints worst-case metrics so the
    MXFP4_PLAIN_FROZEN_* thresholds stay evidence-based."""

    SEEDS = list(range(20260901, 20260901 + _CHARACTERIZE_SEEDS))

    def test_characterize_20_seeds(self):
        import contextlib
        import io

        worst = {}
        for spec in TestFlashInferMxfp4PlainDecode.CASES:
            w = {"rel_l2": 0.0, "max_abs": 0.0, "cosine": 1.0, "norm_ratio": 1.0}
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
            for seed in self.SEEDS:
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
                    metrics = _run_case(
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
            print(f"[mxfp4-characterize] {spec[0]}: worst {w}")
        print("\n[mxfp4-characterize] frozen-threshold proposal (worst * 1.25):")
        for name, w in worst.items():
            print(
                f"  {name}: rel_l2<={w['rel_l2'] * 1.25:.2e} "
                f"cos>={1 - (1 - w['cosine']) * 1.25:.6f} "
                f"norm_ratio={w['norm_ratio']:.5f}"
            )


if __name__ == "__main__":
    unittest.main()
