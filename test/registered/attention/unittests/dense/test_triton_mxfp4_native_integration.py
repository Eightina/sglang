"""L3 Phase 2: native MXFP4 Triton decode through the real backend stack.

Where ``test_triton_mxfp4_native_decode.py`` drives the kernel directly with
hand-built metadata, this module exercises the FULL integration path:

    RadixAttention -> HybridLinear-free dense fixture -> TritonAttnBackend
      -> access registry (NATIVE_FP4 decode for triton)
      -> pool.get_raw_kv_buffer (packed uint8 + E8M0 scales)
      -> mxfp4_decode_attention_fwd

Two contract layers, mirroring the L1/L2 acceptance structure:

1. Cache-write contract: the write path (pool.set_kv_buffer -> eager
   MXFP4KVQuantizeUtil, replaced by the fused kernel in Phase 3) must produce
   bit-exact OCP bytes at every written slot, with the checkpoint FP8 k/v
   scales ignored (MXFP4 is self-scaling). This pins the semantics the native
   kernel reads back.
2. Decode differential: TritonAttnBackend decode output vs the L2 Torch
   oracle reading the SAME packed cache bytes, within the frozen thresholds
   characterized in Phase 1.

Decode-only by design: the production pairing (enforced by kv_cache_hook) is
prefill=flashinfer (PLAIN) + decode=triton (native); triton extend over mxfp4
is out of scope.
"""

import contextlib
import io
import os
import unittest

import torch

from sglang.srt.layers.quantization.fp4_kv_cache_quant_method import MXFP4KVCacheMethod
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kits.attention_unittest.attention_methods.dense_attention import (
    DenseAttentionCase,
    build_dense_attention_fixture,
)
from sglang.test.kits.attention_unittest.attention_methods.fp8_decode_attention import (
    decode_output_metrics,
    format_metrics,
)
from sglang.test.kits.attention_unittest.attention_methods.mxfp4_decode_attention import (
    mxfp4_quantize_reference,
    torch_mxfp4_radix_decode_reference,
)
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=30, stage="base-b", runner_config="1-gpu-large")

_HAS_REQUIREMENTS = torch.cuda.is_available()

# Same hard caps as L1/L2: implementation equivalence on identical cache
# bytes; never relaxed to mask a scale/page/pack bug.
REL_L2_CAP = 2e-2
COSINE_CAP = 0.999
NORM_RATIO_RANGE = (0.98, 1.02)

# Integration thresholds are characterized SEPARATELY from the standalone
# Phase 1 ones (20 seeds, SM120 / torch 2.13.0+cu130): RadixAttention returns
# bf16 outputs (production reality) while the oracle is fp32, so bf16 output
# rounding dominates the residual. Measured worst cases: rel_l2 1.83e-3
# (mqa_hd64_p16_bsz1), cosine 0.9999983 (same case), |norm_ratio - 1| 2.63e-4
# (same case). Tighter than the L2 PLAIN frozen bound (rel_l2 3.1e-3), which
# additionally carried the bf16 K/V materialization loss that the native
# inline fp32 dequant avoids.
MXFP4_TRITON_INTEG_FROZEN_REL_L2 = 2.3e-3
MXFP4_TRITON_INTEG_FROZEN_COSINE = 0.9999978
MXFP4_TRITON_INTEG_FROZEN_NORM_RATIO = (0.99965, 1.00035)

_CHARACTERIZE = os.environ.get("SGLANG_MXFP4_TRITON_INTEGRATION_CHARACTERIZE", "0") == "1"
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
    testcase.assertLessEqual(
        metrics["rel_l2"],
        MXFP4_TRITON_INTEG_FROZEN_REL_L2,
        f"[{context}] rel_l2 {metrics['rel_l2']:.3e} > frozen "
        f"{MXFP4_TRITON_INTEG_FROZEN_REL_L2} ({format_metrics(metrics)})",
    )
    testcase.assertGreaterEqual(
        metrics["cosine"],
        MXFP4_TRITON_INTEG_FROZEN_COSINE,
        f"[{context}] cosine {metrics['cosine']:.6f} < frozen "
        f"{MXFP4_TRITON_INTEG_FROZEN_COSINE} ({format_metrics(metrics)})",
    )
    lo, hi = MXFP4_TRITON_INTEG_FROZEN_NORM_RATIO
    testcase.assertTrue(
        lo <= metrics["norm_ratio"] <= hi,
        f"[{context}] norm_ratio {metrics['norm_ratio']:.5f} outside frozen "
        f"[{lo}, {hi}] ({format_metrics(metrics)})",
    )


def _run_triton_attn_core(fixture):
    """Run the real TritonAttnBackend for the fixture's RadixAttention module;
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
    """Every input token's slot must hold OCP-reference bytes, with the
    checkpoint FP8 scales ignored entirely (same contract as L2)."""
    pool = fixture.runner.token_to_kv_pool
    fb = fixture.forward_batch

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


def _torch_reference_outputs(fixture, case, head_dim, q):
    """Decode reads every token (including the current one) from the packed
    cache, so the oracle is the pure-cache MXFP4 radix decode reference."""
    pool = fixture.runner.token_to_kv_pool
    fb = fixture.forward_batch
    req_to_token = fixture.runner.req_to_token_pool.req_to_token
    module = fixture.actual_module
    q3 = q.view(-1, module.num_heads, module.head_dim)
    outs = []
    for i in range(case.batch_size):
        prefix_len = case.prefix_lens[i]
        seq_len = prefix_len + 1  # decode: one query per request
        outs.append(
            torch_mxfp4_radix_decode_reference(
                q3[i],
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
    attn_out, q, k_pre, v_pre = _run_triton_attn_core(fixture)
    _assert_current_token_write(testcase, fixture, case, head_dim, k_pre, v_pre)
    ref = _torch_reference_outputs(fixture, case, head_dim, q)
    metrics = decode_output_metrics(attn_out, ref)
    if assert_level == "frozen":
        _assert_frozen(testcase, metrics, case.name)
    else:
        _assert_caps(testcase, metrics, case.name)
    return metrics


@unittest.skipUnless(_HAS_REQUIREMENTS, "CUDA is required")
class TestTritonMxfp4NativeIntegrationDecode(CustomTestCase):
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
            backend="triton",
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

    def test_triton_mxfp4_native_integration_decode_cases(self):
        for spec in self.CASES:
            with self.subTest(case=spec[0]):
                metrics = self._run(spec)
                print(
                    f"[mxfp4-triton-integration] {spec[0]}: "
                    f"{format_metrics(metrics)}"
                )


@unittest.skipUnless(
    _HAS_REQUIREMENTS and _CHARACTERIZE,
    "CUDA required; set SGLANG_MXFP4_TRITON_INTEGRATION_CHARACTERIZE=1",
)
class TestTritonMxfp4NativeIntegrationCharacterization(CustomTestCase):
    """20 fixed seeds over the integration decode matrix; prints worst-case
    metrics so the MXFP4_TRITON_INTEG_FROZEN_* thresholds stay
    evidence-based (worst * 1.25)."""

    SEEDS = list(range(20260904, 20260904 + _CHARACTERIZE_SEEDS))

    def test_characterize_20_seeds(self):
        worst = {}
        for spec in TestTritonMxfp4NativeIntegrationDecode.CASES:
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
                    backend="triton",
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
            print(f"[mxfp4-triton-integ-characterize] {spec[0]}: worst {w}")
        print(
            "\n[mxfp4-triton-integ-characterize] frozen-threshold proposal "
            "(worst * 1.25):"
        )
        for name, w in worst.items():
            print(
                f"  {name}: rel_l2<={w['rel_l2'] * 1.25:.2e} "
                f"cos>={1 - (1 - w['cosine']) * 1.25:.6f} "
                f"norm_ratio={w['norm_ratio']:.5f}"
            )


if __name__ == "__main__":
    unittest.main()
