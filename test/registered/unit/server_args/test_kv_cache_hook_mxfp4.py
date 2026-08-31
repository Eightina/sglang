"""Unit tests for --kv-cache-dtype mxfp4 compatibility gating (kv_cache_hook)."""

from types import SimpleNamespace
from unittest.mock import patch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _mxfp4_args(**overrides):
    kwargs = dict(
        kv_cache_dtype="mxfp4",
        disable_cuda_graph=True,
        attention_backend="flashinfer",
        prefill_attention_backend=None,
        decode_attention_backend=None,
    )
    kwargs.update(overrides)
    return SimpleNamespace(**kwargs)


def _patched_platform(is_cuda=True):
    platform = SimpleNamespace(is_cuda=is_cuda)
    return patch(
        "sglang.srt.arg_groups.kv_cache_hook.get_platform", return_value=platform
    )


# use_mla_backend would build a real ModelConfig; the hook tests only need its
# boolean outcome.
def _patched_mla(uses_mla=False):
    return patch(
        "sglang.srt.arg_groups.kv_cache_hook.use_mla_backend",
        return_value=uses_mla,
    )


class TestMxfp4KvCacheHook(CustomTestCase):
    def test_accepts_flashinfer_with_cuda_graph_disabled(self):
        from sglang.srt.arg_groups.kv_cache_hook import handle_kv4_compatibility

        with _patched_platform(), _patched_mla():
            handle_kv4_compatibility(_mxfp4_args())

    def test_rejects_non_flashinfer_backend(self):
        from sglang.srt.arg_groups.kv_cache_hook import handle_kv4_compatibility

        with _patched_platform(), _patched_mla():
            with self.assertRaisesRegex(ValueError, "FlashInfer"):
                handle_kv4_compatibility(_mxfp4_args(attention_backend="triton"))

    def test_rejects_cuda_graph_enabled(self):
        from sglang.srt.arg_groups.kv_cache_hook import handle_kv4_compatibility

        with _patched_platform(), _patched_mla():
            with self.assertRaisesRegex(ValueError, "--disable-cuda-graph"):
                handle_kv4_compatibility(_mxfp4_args(disable_cuda_graph=False))

    def test_rejects_mla(self):
        from sglang.srt.arg_groups.kv_cache_hook import handle_kv4_compatibility

        with _patched_platform(), _patched_mla(uses_mla=True):
            with self.assertRaisesRegex(ValueError, "MHA only"):
                handle_kv4_compatibility(_mxfp4_args())

    def test_rejects_non_cuda(self):
        from sglang.srt.arg_groups.kv_cache_hook import handle_kv4_compatibility

        with _patched_platform(is_cuda=False), _patched_mla():
            with self.assertRaisesRegex(RuntimeError, "CUDA-only"):
                handle_kv4_compatibility(_mxfp4_args())

    def test_non_mxfp4_dtypes_are_untouched(self):
        from sglang.srt.arg_groups.kv_cache_hook import handle_kv4_compatibility

        for dtype in ("auto", "fp8_e4m3", "bf16"):
            with self.subTest(kv_cache_dtype=dtype):
                with _patched_platform(is_cuda=False), _patched_mla():
                    # Non-FP4 dtypes return before any platform check.
                    handle_kv4_compatibility(
                        SimpleNamespace(
                            kv_cache_dtype=dtype,
                            disable_cuda_graph=False,
                            attention_backend="triton",
                            prefill_attention_backend=None,
                            decode_attention_backend=None,
                        )
                    )


if __name__ == "__main__":
    unittest.main()
