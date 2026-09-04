"""Probe tl.dot_scaled (block-scaled FP4 MMA) on SM120 for the MXFP4 decode
kernel: verifies (1) it compiles on sm_120, (2) the e2m1/e8m0 packing
convention matches our pool layout (low nibble = even element, one e8m0 byte
per 32 elements), (3) speed vs the manual unpack + tl.dot path.
"""

import torch
import triton
import triton.language as tl

_DEVICE = "cuda"


@triton.jit
def _probe_dot_scaled_kernel(
    K_Packed,  # (N, D//2) uint8, low nibble = even element
    K_Scale,   # (N, D//32) uint8 e8m0
    Q,         # (D, H) bf16
    Out,       # (N, H) fp32
    stride_kn,
    stride_ksn,
    stride_qd,
    stride_on,
    N: tl.constexpr,
    D: tl.constexpr,
    H: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)
    offs_n = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_dp = tl.arange(0, D // 2)  # packed byte columns (2 e2m1 per byte)
    offs_dq = tl.arange(0, D)       # logical K dim for rhs (bf16, unpacked)
    mask_n = offs_n < N

    k = tl.load(
        K_Packed + offs_n[:, None] * stride_kn + offs_dp[None, :],
        mask=mask_n[:, None],
        other=0,
    )
    ks = tl.load(
        K_Scale + offs_n[:, None] * stride_ksn + tl.arange(0, D // 32)[None, :],
        mask=mask_n[:, None],
        other=127,
    )
    q = tl.load(Q + offs_dq[:, None] * stride_qd + tl.arange(0, H)[None, :])

    qk = tl.dot_scaled(k, ks, "e2m1", q, None, "bf16")
    tl.store(Out + offs_n[:, None] * stride_on + tl.arange(0, H)[None, :], qk, mask=mask_n[:, None])


@triton.jit
def _probe_manual_kernel(
    K_Packed,
    K_Scale,
    Q,
    Out,
    stride_kn,
    stride_ksn,
    stride_qd,
    stride_on,
    N: tl.constexpr,
    D: tl.constexpr,
    H: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)
    offs_n = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, D)
    mask_n = offs_n < N

    kbyte = tl.load(
        K_Packed + offs_n[:, None] * stride_kn + (offs_d // 2)[None, :],
        mask=mask_n[:, None],
        other=0,
    )
    is_odd = (offs_d % 2) == 1
    code = tl.where(is_odd[None, :], (kbyte >> 4) & 0x0F, kbyte & 0x0F)
    mant = (code & 0x1).to(tl.float32)
    exp_bits = (code >> 1) & 0x3
    normal = tl.exp2(exp_bits.to(tl.float32) - 1.0) * (1.0 + 0.5 * mant)
    sub = 0.5 * mant
    mag = tl.where(exp_bits == 0, sub, normal)
    k = tl.where((code & 0x8) != 0, -mag, mag)

    sbyte = tl.load(
        K_Scale + offs_n[:, None] * stride_ksn + (offs_d // 32)[None, :],
        mask=mask_n[:, None],
        other=127,
    )
    sf = tl.exp2(sbyte.to(tl.float32) - 127.0)
    k = (k * sf).to(tl.bfloat16)

    q = tl.load(Q + offs_d[:, None] * stride_qd + tl.arange(0, H)[None, :])
    qk = tl.dot(k, q)
    tl.store(Out + offs_n[:, None] * stride_on + tl.arange(0, H)[None, :], qk, mask=mask_n[:, None])


def _make(n, d, seed=0):
    g = torch.Generator(device=_DEVICE).manual_seed(seed)
    packed = torch.randint(0, 256, (n, d // 2), generator=g, dtype=torch.uint8, device=_DEVICE)
    scale = torch.randint(120, 135, (n, d // 32), generator=g, dtype=torch.uint8, device=_DEVICE)
    q = torch.randn((d, 16), generator=g, device=_DEVICE).to(torch.bfloat16)
    return packed, scale, q


def _torch_ref(packed, scale, q):
    n, d2 = packed.shape
    d = d2 * 2
    codes = torch.empty((n, d), dtype=torch.uint8, device=_DEVICE)
    codes[:, 0::2] = packed & 0x0F
    codes[:, 1::2] = (packed >> 4) & 0x0F
    table = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], device=_DEVICE)
    mag = table[(codes & 0x7).long()]
    vals = torch.where((codes & 0x8) != 0, -mag, mag)
    sf = torch.exp2(scale.to(torch.float32) - 127.0).repeat_interleave(32, dim=1)
    k = (vals * sf).to(torch.bfloat16)
    return (k.float() @ q.float())


def _bench(fn, iters=50):
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    import statistics

    ts = []
    for _ in range(iters):
        s.record()
        fn()
        e.record()
        torch.cuda.synchronize()
        ts.append(s.elapsed_time(e))
    return statistics.median(ts) * 1e3


def main():
    n, d = 4096, 256
    packed, scale, q = _make(n, d)
    ref = _torch_ref(packed, scale, q)

    out_ds = torch.empty((n, 16), dtype=torch.float32, device=_DEVICE)
    try:
        _probe_dot_scaled_kernel[(triton.cdiv(n, 64),)](
            packed, scale, q, out_ds,
            packed.stride(0), scale.stride(0), q.stride(0), out_ds.stride(0),
            N=n, D=d, H=16, BLOCK_N=64,
        )
        torch.cuda.synchronize()
        err = (out_ds - ref).abs().max().item()
        rel = ((out_ds - ref).norm() / ref.norm()).item()
        print(f"dot_scaled compiled OK on this device; max_abs={err:.3e} rel_l2={rel:.3e}")
        t_ds = _bench(
            lambda: _probe_dot_scaled_kernel[(triton.cdiv(n, 64),)](
                packed, scale, q, out_ds,
                packed.stride(0), scale.stride(0), q.stride(0), out_ds.stride(0),
                N=n, D=d, H=16, BLOCK_N=64,
            )
        )
        print(f"dot_scaled kernel time: {t_ds:.1f} us")
    except Exception as ex:
        print(f"dot_scaled FAILED: {type(ex).__name__}: {str(ex)[:300]}")

    out_man = torch.empty((n, 16), dtype=torch.float32, device=_DEVICE)
    _probe_manual_kernel[(triton.cdiv(n, 64),)](
        packed, scale, q, out_man,
        packed.stride(0), scale.stride(0), q.stride(0), out_man.stride(0),
        N=n, D=d, H=16, BLOCK_N=64,
    )
    torch.cuda.synchronize()
    err = (out_man - ref).abs().max().item()
    rel = ((out_man - ref).norm() / ref.norm()).item()
    print(f"manual unpack+dot: max_abs={err:.3e} rel_l2={rel:.3e}")
    t_man = _bench(
        lambda: _probe_manual_kernel[(triton.cdiv(n, 64),)](
            packed, scale, q, out_man,
            packed.stride(0), scale.stride(0), q.stride(0), out_man.stride(0),
            N=n, D=d, H=16, BLOCK_N=64,
        )
    )
    print(f"manual kernel time: {t_man:.1f} us")


if __name__ == "__main__":
    main()
