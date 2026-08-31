"""Conformance tests for strict OCP MXFP4 KV-cache encoding."""

import unittest

import torch

from sglang.srt.layers.quantization.kvfp4_tensor import MXFP4KVQuantizeUtil
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.kits.attention_unittest.attention_methods.mxfp4_decode_attention import (
    E8M0_NAN_BYTE,
    mxfp4_dequantize_reference,
    mxfp4_quantize_reference,
)
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestMXFP4CodecConformance(CustomTestCase):
    def _assert_production_matches_oracle(self, x):
        actual_data, actual_scales = MXFP4KVQuantizeUtil.batched_quantize(x)
        expected_data, expected_scales = mxfp4_quantize_reference(x)
        self.assertTrue(torch.equal(actual_data, expected_data))
        self.assertTrue(torch.equal(actual_scales, expected_scales))

        actual_dq = MXFP4KVQuantizeUtil.batched_dequantize(
            actual_data,
            actual_scales,
            logical_dim=x.shape[-1],
            dtype=torch.float32,
        )
        expected_dq = mxfp4_dequantize_reference(
            expected_data,
            expected_scales,
            logical_dim=x.shape[-1],
            dtype=torch.float32,
        )
        torch.testing.assert_close(actual_dq, expected_dq, rtol=0, atol=0, equal_nan=True)
        return actual_data, actual_scales, actual_dq

    def test_all_e2m1_codes_and_nibble_order(self):
        values = torch.tensor(
            [
                0.0,
                0.5,
                1.0,
                1.5,
                2.0,
                3.0,
                4.0,
                6.0,
                -0.0,
                -0.5,
                -1.0,
                -1.5,
                -2.0,
                -3.0,
                -4.0,
                -6.0,
            ],
            dtype=torch.float32,
        ).repeat(2).view(1, 1, 32)
        packed, scales, reconstructed = self._assert_production_matches_oracle(values)
        expected_codes = torch.arange(16, dtype=torch.uint8).repeat(2)
        expected_packed = expected_codes[0::2] | (expected_codes[1::2] << 4)
        self.assertTrue(torch.equal(packed.flatten(), expected_packed))
        self.assertEqual(scales.item(), 127)  # scale = 1.0
        torch.testing.assert_close(reconstructed, values, rtol=0, atol=0)

    def test_round_ties_to_even_and_saturation(self):
        values = torch.tensor(
            [0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0, 7.0],
            dtype=torch.float32,
        )
        values = torch.cat([values, -values, torch.tensor([6.0] * 16)]).view(1, 1, 32)
        packed, scales, reconstructed = self._assert_production_matches_oracle(values)
        self.assertEqual(scales.item(), 127)
        expected = torch.tensor(
            [0.0, 1.0, 1.0, 2.0, 2.0, 4.0, 4.0, 6.0],
            dtype=torch.float32,
        )
        torch.testing.assert_close(reconstructed[0, 0, :8], expected, rtol=0, atol=0)
        torch.testing.assert_close(reconstructed[0, 0, 8:16], -expected, rtol=0, atol=0)
        self.assertEqual(packed.dtype, torch.uint8)

    def test_scale_exponent_boundaries_and_zero_block(self):
        x = torch.zeros(1, 3, 32, dtype=torch.float32)
        x[0, 0, 0] = 3.9  # floor_pow2=2, scale=0.5, byte=126
        x[0, 1, 0] = 8.0  # floor_pow2=8, scale=2, byte=128
        _, scales, reconstructed = self._assert_production_matches_oracle(x)
        self.assertEqual(scales.flatten().tolist(), [126, 128, 0])
        self.assertEqual(reconstructed[0, 2].abs().sum().item(), 0.0)

    def test_nan_and_infinity_policy(self):
        x = torch.zeros(1, 2, 32, dtype=torch.float32)
        x[0, 0, 3] = float("nan")
        x[0, 1, 4] = float("inf")
        x[0, 1, 5] = -float("inf")
        _, scales, reconstructed = self._assert_production_matches_oracle(x)
        self.assertEqual(scales[0, 0, 0].item(), E8M0_NAN_BYTE)
        self.assertTrue(torch.isnan(reconstructed[0, 0]).all())
        self.assertEqual(scales[0, 1, 0].item(), 254)
        self.assertTrue(torch.isinf(reconstructed[0, 1, 4]))
        self.assertTrue(torch.isinf(reconstructed[0, 1, 5]))

    def test_head_isolation_and_partial_block(self):
        x = torch.zeros(2, 2, 33, dtype=torch.float32)
        x[0, 0, 0] = 4.0
        x[0, 0, 32] = 0.5
        x[0, 1, 0] = 16.0
        packed, scales, reconstructed = self._assert_production_matches_oracle(x)
        self.assertEqual(packed.shape, (2, 2, 17))
        self.assertEqual(scales.shape, (2, 2, 2))
        self.assertEqual(scales[0, 0].tolist(), [127, 124])
        self.assertEqual(scales[0, 1].tolist(), [129, 0])
        self.assertEqual(reconstructed.shape, x.shape)

    def test_random_cpu_parity(self):
        torch.manual_seed(20260831)
        for head_dim in (1, 2, 31, 32, 33, 64, 256):
            with self.subTest(head_dim=head_dim):
                x = torch.randn(3, 4, head_dim, dtype=torch.bfloat16)
                self._assert_production_matches_oracle(x)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_fixed_vector_cpu_cuda_byte_parity(self):
        torch.manual_seed(20260901)
        x_cpu = torch.randn(3, 4, 65, dtype=torch.bfloat16)
        cpu_data, cpu_scales = MXFP4KVQuantizeUtil.batched_quantize(x_cpu)
        cuda_data, cuda_scales = MXFP4KVQuantizeUtil.batched_quantize(x_cpu.cuda())
        self.assertTrue(torch.equal(cpu_data, cuda_data.cpu()))
        self.assertTrue(torch.equal(cpu_scales, cuda_scales.cpu()))


if __name__ == "__main__":
    unittest.main()
