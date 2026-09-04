"""Micro-bench for the MXFP4 native decode kernel (L3 perf investigation).

Kernel-only timing (stage1 + stage2) at the Qwen3.8-27B full-attention shape
(batch=1, 24 q heads, 4 kv heads, head_dim=256, page_size=16), sweeping
sequence length, split count, BLOCK_N and num_warps. Prints achieved DRAM
traffic vs the RTX 5090 roofline (~1.79 TB/s GDDR7, 170 SMs) so the gap to
memory-bound is visible.

Usage:
    python scripts/playground/bench_mxfp4_decode_kernel.py [--seqs 512,4096]
        [--splits 8,32] [--block-n 64,128] [--warps 4,8] [--iters 100]
"""

import argparse
import statistics

import torch

from sglang.kernels.ops.attention.mxfp4_decode_attention import (
    mxfp4_decode_attention_fwd,
)

_DEVICE = "cuda"
# RTX 5090 (SM120): 170 SMs, GDDR7 512-bit @ 28 Gbps -> ~1.79 TB/s.
_HBM_BW_TBPS = 1.79


def _build_case(batch, q_heads, kv_heads, head_dim, seq_len, page_size, seed):
    torch.manual_seed(seed)
    slots = seq_len * batch + page_size + 16
    packed_dim = head_dim // 2
    num_blocks = head_dim // 32
    k_pool = torch.randint(
        0, 256, (slots, kv_heads, packed_dim), dtype=torch.uint8, device=_DEVICE
    )
    v_pool = torch.randint(
        0, 256, (slots, kv_heads, packed_dim), dtype=torch.uint8, device=_DEVICE
    )
    ks_pool = torch.randint(
        120, 135, (slots, kv_heads, num_blocks), dtype=torch.uint8, device=_DEVICE
    )
    vs_pool = torch.randint(
        120, 135, (slots, kv_heads, num_blocks), dtype=torch.uint8, device=_DEVICE
    )
    q = torch.randn((batch, q_heads, head_dim), device=_DEVICE).to(torch.bfloat16)

    kv_indptr = torch.tensor(
        [0] + [seq_len * (i + 1) for i in range(batch)],
        dtype=torch.int32,
        device=_DEVICE,
    )
    kv_indices = torch.arange(1, 1 + seq_len * batch, dtype=torch.int32, device=_DEVICE)
    return (
        q,
        k_pool,
        v_pool,
        ks_pool,
        vs_pool,
        kv_indptr,
        kv_indices,
    )


def _bench_one(
    q,
    k_pool,
    v_pool,
    ks_pool,
    vs_pool,
    kv_indptr,
    kv_indices,
    *,
    head_dim,
    max_kv_splits,
    block_n,
    num_warps,
    iters,
    use_dot_scaled=True,
):
    batch, q_heads = q.shape[0], q.shape[1]
    o = torch.zeros((batch, q_heads, head_dim), dtype=torch.float32, device=_DEVICE)
    attn_logits = torch.empty(
        (batch, q_heads, max_kv_splits, head_dim), dtype=torch.float32, device=_DEVICE
    )
    attn_lse = torch.empty(
        (batch, q_heads, max_kv_splits), dtype=torch.float32, device=_DEVICE
    )
    seq_lens = [int(kv_indptr[i + 1] - kv_indptr[i]) for i in range(batch)]
    num_kv_splits = torch.tensor(
        [
            max(1, min(max_kv_splits, (s + 31) // 32))
            for s in seq_lens
        ],
        dtype=torch.int32,
        device=_DEVICE,
    )

    def run():
        mxfp4_decode_attention_fwd(
            q,
            k_pool,
            v_pool,
            ks_pool,
            vs_pool,
            o,
            kv_indptr,
            kv_indices,
            head_dim**-0.5,
            page_size=16,
            max_kv_splits=max_kv_splits,
            attn_logits=attn_logits,
            attn_lse=attn_lse,
            num_kv_splits=num_kv_splits,
            block_n=block_n,
            num_warps=num_warps,
            use_dot_scaled=use_dot_scaled,
        )

    # warmup (compile + cache)
    for _ in range(5):
        run()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    times = []
    for _ in range(iters):
        start.record()
        run()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    return statistics.median(times) * 1e3  # us


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seqs", default="512,1024,2048,4096,8192")
    parser.add_argument("--splits", default="8,16,32,64")
    parser.add_argument("--block-n", default="64,128")
    parser.add_argument("--warps", default="4,8")
    parser.add_argument("--dot-scaled", default="1,0")
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--q-heads", type=int, default=24)
    parser.add_argument("--kv-heads", type=int, default=4)
    parser.add_argument("--head-dim", type=int, default=256)
    args = parser.parse_args()

    seqs = [int(x) for x in args.seqs.split(",")]
    splits = [int(x) for x in args.splits.split(",")]
    block_ns = [int(x) for x in args.block_n.split(",")]
    warps_list = [int(x) for x in args.warps.split(",")]
    ds_list = [bool(int(x)) for x in args.dot_scaled.split(",")]

    print(
        f"shape: batch={args.batch} q_heads={args.q_heads} kv_heads={args.kv_heads} "
        f"head_dim={args.head_dim} page_size=16 | HBM roofline {_HBM_BW_TBPS} TB/s"
    )
    print(
        f"{'seq':>6} {'splits':>6} {'BN':>4} {'warps':>5} {'ds':>2} | {'time_us':>9} "
        f"{'kv_MB':>7} {'GB/s':>8} {'%roof':>6}"
    )
    for seq_len in seqs:
        built = _build_case(
            args.batch,
            args.q_heads,
            args.kv_heads,
            args.head_dim,
            seq_len,
            16,
            seed=20260904,
        )
        # KV traffic per step: seq * kv_heads * (packed + scale) * 2(K+V) * batch
        kv_bytes = (
            seq_len
            * args.kv_heads
            * (args.head_dim // 2 + args.head_dim // 32)
            * 2
            * args.batch
        )
        for s in splits:
            for bn in block_ns:
                for w in warps_list:
                    for ds in ds_list:
                        t_us = _bench_one(
                            *built,
                            head_dim=args.head_dim,
                            max_kv_splits=s,
                            block_n=bn,
                            num_warps=w,
                            iters=args.iters,
                            use_dot_scaled=ds,
                        )
                        gbps = kv_bytes / (t_us * 1e-6) / 1e9
                        pct = gbps / (_HBM_BW_TBPS * 1000) * 100
                        print(
                            f"{seq_len:>6} {s:>6} {bn:>4} {w:>5} {int(ds):>2} | {t_us:>9.1f} "
                            f"{kv_bytes / 1e6:>7.1f} {gbps:>8.0f} {pct:>5.1f}%"
                        )


if __name__ == "__main__":
    main()
