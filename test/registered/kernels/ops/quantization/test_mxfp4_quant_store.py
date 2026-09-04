"""L3 Phase 3: fused MXFP4 quantize+store kernel write contract.

The fused triton kernel (``quant_store_kv_mxfp4``) replaces the eager
``MXFP4KVQuantizeUtil.batched_quantize`` + indexed-scatter write path inside
``MXFP4KVCacheMethod.quantize_and_store``. This module locks the contract in
the L1 A-step pattern:

1. Bit-exactness: fused kernel output bytes == eager production codec ==
   independent OCP oracle (``mxfp4_quantize_reference``), over random
   bf16/fp16 inputs, special values (NaN / +-Inf / signed zeros / subnormal
   bf16 / extremes), and partial blocks (head_dim not a multiple of 32).
2. Reserved slot: writes with loc == 0 leave slot 0 bytes untouched
   (CUDA-graph padding contract, ``reserved_skip_index=0``).
3. Gate: fp32 / CPU / non-contiguous-last-dim inputs must be reported
   unsupported so ``quantize_and_store`` falls back to the eager codec.

The bit-exactness scope is bf16/fp16 (production K/V dtypes); see the
``mxfp4_quant`` module docstring for the exponent-extraction argument.
"""

import unittest

import torch

from sglang.kernels.ops.quantization.mxfp4_quant import (
    mxfp4_fused_store_supported,
    quant_store_kv_mxfp4,
)
from sglang.srt.layers.quantization.kvfp4_tensor import MXFP4KVQuantizeUtil
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kits.attention_unittest.attention_methods.mxfp4_decode_attention import (
    mxfp4_quantize_reference,
)
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=15, stage="base-b", runner_config="1-gpu-large")

_HAS_REQUIREMENTS = torch.cuda.is_available()
_DEVICE = "cuda"


def _make_buffers(slots, heads, dim, device=_DEVICE):
    packed_dim = (dim + 1) // 2
    num_blocks = (dim + 31) // 32
    data = torch.zeros((slots, heads, packed_dim), dtype=torch.uint8, device=device)
    scales = torch.zeros((slots, heads, num_blocks), dtype=torch.uint8, device=device)
    return data, scales


def _fused_write(k, v, loc):
    """Run the fused kernel into fresh pool-shaped buffers; returns
    (k_data, v_data, k_sf, v_sf) at the written slots."""
    slots = int(loc.max().item()) + 2
    heads, dim = k.shape[1], k.shape[2]
    kd, ks = _make_buffers(slots, heads, dim, device=k.device)
    vd, vs = _make_buffers(slots, heads, dim, device=v.device)
    quant_store_kv_mxfp4(k, v, loc, kd, vd, ks, vs)
    return kd, vd, ks, vs


@unittest.skipUnless(_HAS_REQUIREMENTS, "CUDA is required")
class TestMxfp4FusedStoreBitExact(CustomTestCase):
    def _check_bit_exact(self, k, v, loc, context):
        kd, vd, ks, vs = _fused_write(k, v, loc)
        valid = loc != 0
        locs_valid = loc[valid]

        eager_k, eager_ks = MXFP4KVQuantizeUtil.batched_quantize(k)
        eager_v, eager_vs = MXFP4KVQuantizeUtil.batched_quantize(v)
        oracle_k, oracle_ks = mxfp4_quantize_reference(k)
        oracle_v, oracle_vs = mxfp4_quantize_reference(v)

        # The eager codec and the oracle must agree first (L2 invariant);
        # otherwise this test would chase a moving target.
        self.assertTrue(
            torch.equal(eager_k, oracle_k) and torch.equal(eager_ks, oracle_ks),
            f"[{context}] eager codec drifted from the OCP oracle (K scales)",
        )
        self.assertTrue(
            torch.equal(eager_v, oracle_v) and torch.equal(eager_vs, oracle_vs),
            f"[{context}] eager codec drifted from the OCP oracle (V scales)",
        )

        self.assertTrue(
            torch.equal(kd[locs_valid], eager_k[valid]),
            f"[{context}] fused K packed bytes differ from eager codec",
        )
        self.assertTrue(
            torch.equal(ks[locs_valid], eager_ks[valid]),
            f"[{context}] fused K E8M0 bytes differ from eager codec",
        )
        self.assertTrue(
            torch.equal(vd[locs_valid], eager_v[valid]),
            f"[{context}] fused V packed bytes differ from eager codec",
        )
        self.assertTrue(
            torch.equal(vs[locs_valid], eager_vs[valid]),
            f"[{context}] fused V E8M0 bytes differ from eager codec",
        )

    def test_random_bf16_dims(self):
        g = torch.Generator(device=_DEVICE).manual_seed(20260903)
        for dim in (32, 48, 64, 128, 256):
            for heads in (1, 4):
                with self.subTest(dim=dim, heads=heads):
                    k = (torch.randn((7, heads, dim), generator=g, device=_DEVICE) * 2).to(
                        torch.bfloat16
                    )
                    v = (torch.randn((7, heads, dim), generator=g, device=_DEVICE) * 2).to(
                        torch.bfloat16
                    )
                    loc = torch.arange(1, 8, dtype=torch.int64, device=_DEVICE)
                    self._check_bit_exact(k, v, loc, f"random_bf16_d{dim}_h{heads}")

    def test_random_fp16(self):
        g = torch.Generator(device=_DEVICE).manual_seed(5)
        k = torch.randn((5, 2, 128), generator=g, device=_DEVICE).to(torch.float16)
        v = torch.randn((5, 2, 128), generator=g, device=_DEVICE).to(torch.float16)
        loc = torch.arange(1, 6, dtype=torch.int64, device=_DEVICE)
        self._check_bit_exact(k, v, loc, "random_fp16")

    def test_special_values(self):
        # NaN blocks, +-Inf blocks, signed zeros, bf16 subnormals and
        # extremes; one (heads=4, dim=64) row per scenario so each 32-block
        # isolates a case.
        row = torch.zeros((4, 64), dtype=torch.float32)
        row[0] = float("nan")  # NaN block -> scale 0xFF, zero codes
        row[1, :16] = float("inf")
        row[1, 16:] = -float("inf")  # mixed +-Inf block -> scale 2^127
        row[2, ::2] = 0.0
        row[2, 1::2] = -0.0  # signed zeros -> codes 0x0 / 0x8
        row[3, 0] = 3.3895e38  # bf16 max
        row[3, 1] = -3.3895e38
        row[3, 2] = 1e-40  # fp32 subnormal -> rounds to 0 in bf16
        row[3, 3] = 2.0**-130  # bf16 subnormal
        k = row.unsqueeze(0).to(torch.bfloat16).expand(3, 4, 64).contiguous().to(_DEVICE)
        v = (-k).clone()
        loc = torch.arange(1, 4, dtype=torch.int64, device=_DEVICE)
        self._check_bit_exact(k, v, loc, "special_values")

    def test_all_zero_block(self):
        k = torch.zeros((2, 2, 64), dtype=torch.bfloat16, device=_DEVICE)
        v = torch.zeros((2, 2, 64), dtype=torch.bfloat16, device=_DEVICE)
        loc = torch.arange(1, 3, dtype=torch.int64, device=_DEVICE)
        self._check_bit_exact(k, v, loc, "all_zero")
        # amax == 0 -> scale byte 0 (2^-127) with zero codes.
        _, _, ks, _ = _fused_write(k, v, loc)
        self.assertTrue((ks[loc] == 0).all())

    def test_tie_midpoints(self):
        # Exact RNE midpoints after scaling. amax = 6.0 pins the block scale
        # to 2^(floor(log2(6))-2) = 1, so the bf16-exact tie values below hit
        # the distance-table midpoints (.25/.75/1.25/1.75/2.5/3.5/5) bit for
        # bit; ties-to-even must pick codes 0/2/2/4/4/6/6.
        base = torch.tensor(
            [0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0, 6.0], dtype=torch.float32
        )
        row = base.repeat(8)[:64]  # dim 64 = two 32-blocks, amax 6.0
        k = row.unsqueeze(0).unsqueeze(0).to(torch.bfloat16)
        v = (-row).unsqueeze(0).unsqueeze(0).to(torch.bfloat16)
        loc = torch.tensor([1], dtype=torch.int64, device=_DEVICE)
        self._check_bit_exact(
            k.contiguous().to(_DEVICE), v.contiguous().to(_DEVICE), loc, "ties"
        )


@unittest.skipUnless(_HAS_REQUIREMENTS, "CUDA is required")
class TestMxfp4FusedStoreReservedSlot(CustomTestCase):
    def test_slot0_untouched(self):
        g = torch.Generator(device=_DEVICE).manual_seed(9)
        k = torch.randn((4, 2, 64), generator=g, device=_DEVICE).to(torch.bfloat16)
        v = torch.randn((4, 2, 64), generator=g, device=_DEVICE).to(torch.bfloat16)
        # Token 1 targets the reserved slot 0; its bytes must not change.
        loc = torch.tensor([1, 0, 2, 3], dtype=torch.int64, device=_DEVICE)

        kd, ks = _make_buffers(8, 2, 64)
        vd, vs = _make_buffers(8, 2, 64)
        sentinel = torch.tensor(0xA5, dtype=torch.uint8, device=_DEVICE)
        kd[0].fill_(sentinel)
        ks[0].fill_(sentinel)
        vd[0].fill_(sentinel)
        vs[0].fill_(sentinel)
        quant_store_kv_mxfp4(k, v, loc, kd, vd, ks, vs)

        self.assertTrue((kd[0] == sentinel).all(), "slot 0 K data was written")
        self.assertTrue((ks[0] == sentinel).all(), "slot 0 K scale was written")
        self.assertTrue((vd[0] == sentinel).all(), "slot 0 V data was written")
        self.assertTrue((vs[0] == sentinel).all(), "slot 0 V scale was written")

        # The non-reserved rows must match the eager codec byte-for-byte:
        # token 0 -> slot 1, token 1 -> slot 0 (skipped), token 2 -> slot 2,
        # token 3 -> slot 3.
        eager_k, eager_ks = MXFP4KVQuantizeUtil.batched_quantize(k)
        for token, slot in ((0, 1), (2, 2), (3, 3)):
            self.assertTrue(torch.equal(kd[slot], eager_k[token]))
            self.assertTrue(torch.equal(ks[slot], eager_ks[token]))


@unittest.skipUnless(_HAS_REQUIREMENTS, "CUDA is required")
class TestMxfp4FusedStoreGate(CustomTestCase):
    def test_supported_shapes(self):
        k = torch.randn((3, 2, 64), device=_DEVICE).to(torch.bfloat16)
        loc = torch.arange(1, 4, dtype=torch.int64, device=_DEVICE)
        self.assertTrue(mxfp4_fused_store_supported(k, k.clone(), loc))

    def test_rejects_fp32(self):
        k = torch.randn((3, 2, 64), device=_DEVICE)
        loc = torch.arange(1, 4, dtype=torch.int64, device=_DEVICE)
        self.assertFalse(mxfp4_fused_store_supported(k, k.clone(), loc))

    def test_rejects_non_contiguous_last_dim(self):
        base = torch.randn((3, 2, 128), device=_DEVICE).to(torch.bfloat16)
        k = base[..., ::2]  # last-dim stride 2
        loc = torch.arange(1, 4, dtype=torch.int64, device=_DEVICE)
        self.assertFalse(mxfp4_fused_store_supported(k, k.clone(), loc))

    def test_rejects_cpu(self):
        k = torch.randn((3, 2, 64)).to(torch.bfloat16)
        loc = torch.arange(1, 4, dtype=torch.int64)
        self.assertFalse(mxfp4_fused_store_supported(k, k.clone(), loc))

    def test_rejects_shape_mismatch(self):
        k = torch.randn((3, 2, 64), device=_DEVICE).to(torch.bfloat16)
        v = torch.randn((3, 2, 32), device=_DEVICE).to(torch.bfloat16)
        loc = torch.arange(1, 4, dtype=torch.int64, device=_DEVICE)
        self.assertFalse(mxfp4_fused_store_supported(k, v, loc))


if __name__ == "__main__":
    unittest.main()
