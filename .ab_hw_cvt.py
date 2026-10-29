#!/usr/bin/env python3
"""New-vs-old A/B for the hardware E2M1 decode, plus a reachability check.

Two questions.

1. Does real quantized data ever hit the one domain where the hardware path
   and the LUT disagree -- E8M0 byte 0 (scale 2^-127) paired with a nonzero
   code? The LUT's subnormal branch writes 1 << (E+6), dropping the E2M1
   mantissa bit, so it returns 1.0*2^-127 where the hardware path returns the
   arithmetically correct 1.5*2^-127. That is 2 of 4096 table rows (codes 3
   and 11). _e2m1_scale_to_bf16's own docstring claims production pairs byte 0
   only with all-zero codes; this counts it instead of trusting it.

2. Is the kernel output bitwise identical to the previous build on realistic
   inputs? Run this, stash the kernel, run it again, diff the saved tensors.

    python .ab_hw_cvt.py new            # writes /tmp/ab_hw_cvt_new.pt
    git stash push python/sglang/kernels/ops/attention/decode_attention_mxfp4_sm120.py
    python .ab_hw_cvt.py old            # writes /tmp/ab_hw_cvt_old.pt
    git stash pop
    python .ab_hw_cvt.py cmp
"""
import sys

import torch
import triton

from sglang.kernels.ops.attention import decode_attention_mxfp4_sm120 as mod
from sglang.srt.layers.quantization.kvfp4_tensor import MXFP4KVQuantizeUtil

Q_HEADS, KV_HEADS, HEAD_DIM = 24, 4, 256
DEV = "cuda"
SM_SCALE = 1.0 / (HEAD_DIM**0.5)
CASES = [
    (1, 64, 8),
    (1, 1000, 8),
    (1, 4096, 85),
    (1, 32768, 85),
    (1, 100000, 85),
    (2, 777, 16),
]


def reachability():
    """Count (scale byte == 0, code != 0) pairs over several input regimes."""
    print("=== does scale byte 0 ever pair with a nonzero code? ===")
    gen = torch.Generator(device=DEV).manual_seed(99)
    regimes = {
        "N(0,1)": lambda n: torch.randn(n, dtype=torch.bfloat16, device=DEV, generator=gen),
        "N(0,1)*2^-60": lambda n: torch.randn(n, dtype=torch.bfloat16, device=DEV, generator=gen) * 2.0**-60,
        "all zeros": lambda n: torch.zeros(n, dtype=torch.bfloat16, device=DEV),
        "bf16 subnormals": lambda n: (
            torch.randint(1, 128, (n,), device=DEV, generator=gen)
            .to(torch.int16)
            .view(torch.bfloat16)
        ),
    }
    for name, make in regimes.items():
        ref = make(8192 * KV_HEADS * HEAD_DIM).view(8192, KV_HEADS, HEAD_DIM)
        packed, scales = MXFP4KVQuantizeUtil.batched_quantize(ref)
        sb = scales.reshape(-1)
        # one scale byte covers 32 elements = 16 packed bytes
        codes = packed.reshape(sb.numel(), 16)
        nz = (codes != 0).any(dim=1)
        hits = int(((sb == 0) & nz).sum())
        print(f"  {name:18s} scale-byte-0 blocks {int((sb == 0).sum()):7d} / "
              f"{sb.numel():7d} | of those with a nonzero code: {hits}")


def run_all():
    out = {}
    for bs, seq, msp in CASES:
        gen = torch.Generator(device=DEV).manual_seed(1000 + seq)
        slots = bs * seq + 16
        k = torch.randn((slots, KV_HEADS, HEAD_DIM), dtype=torch.bfloat16,
                        device=DEV, generator=gen)
        v = torch.randn((slots, KV_HEADS, HEAD_DIM), dtype=torch.bfloat16,
                        device=DEV, generator=gen)
        kp, ks = MXFP4KVQuantizeUtil.batched_quantize(k)
        vp, vs = MXFP4KVQuantizeUtil.batched_quantize(v)
        q = torch.randn((bs, Q_HEADS, HEAD_DIM), dtype=torch.bfloat16,
                        device=DEV, generator=gen)
        o = torch.empty((bs, Q_HEADS, HEAD_DIM), dtype=torch.bfloat16, device=DEV)
        kv_indptr = torch.tensor([i * seq for i in range(bs + 1)],
                                 dtype=torch.int32, device=DEV)
        kv_indices = torch.arange(bs * seq, dtype=torch.int32, device=DEV)
        splits = max(1, min(msp, triton.cdiv(seq, mod._MIN_BLOCK_KV)))
        nks = torch.full((bs,), splits, dtype=torch.int32, device=DEV)
        logits = torch.empty((bs, Q_HEADS, msp, HEAD_DIM), dtype=torch.float32, device=DEV)
        lse = torch.empty((bs, Q_HEADS, msp), dtype=torch.float32, device=DEV)
        mod.mxfp4_decode_attention_fwd(
            q, kp, vp, ks, vs, o, kv_indptr, kv_indices, SM_SCALE,
            page_size=1, max_kv_splits=msp, attn_logits=logits, attn_lse=lse,
            num_kv_splits=nks,
        )
        torch.cuda.synchronize()
        out[f"bs{bs}_s{seq}_sp{msp}"] = o.cpu()
        print(f"  ran bs={bs} seq={seq} splits={splits}/{msp} "
              f"finite={bool(o.isfinite().all())} absmax={o.abs().max().item():.5f}")
    return out


def main():
    what = sys.argv[1] if len(sys.argv) > 1 else "new"
    if what == "cmp":
        a = torch.load("/tmp/ab_hw_cvt_new.pt")
        b = torch.load("/tmp/ab_hw_cvt_old.pt")
        allsame = True
        for key in sorted(a):
            x, y = a[key], b[key]
            same = torch.equal(x.view(torch.int16), y.view(torch.int16))
            if not same:
                d = (x.float() - y.float()).abs()
                rel = d.norm() / y.float().norm().clamp_min(1e-30)
                print(f"  {key:22s} bitwise=NO  maxabs={d.max():.3e} rel_l2={rel:.3e}")
                allsame = False
            else:
                print(f"  {key:22s} bitwise=YES")
        print("\nRESULT:", "all cases bit-identical" if allsame
              else "differences present (see rel_l2 above)")
        return
    reachability()
    print(f"\n=== running cases ({what}) ===")
    torch.save(run_all(), f"/tmp/ab_hw_cvt_{what}.pt")
    print(f"saved /tmp/ab_hw_cvt_{what}.pt")


if __name__ == "__main__":
    main()
