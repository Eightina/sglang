"""Exactness check for the page-granular index path of the MXFP4 decode kernel.

The paged path reads one page id per BLOCK_N tile and derives the row offsets
affinely. That is only equivalent to the token-level path if a page's slots are
``PAGE_SIZE`` consecutive ids, which is the contract the token-level index
builder itself relies on (``token = entry * ps + pos % ps``).

So the two arms are fed mappings built from one shared permutation and must
agree bit for bit. A *shuffled* permutation is the point: under the identity the
paged path would look correct even if it ignored the page table entirely.
"""

import torch

import sglang.kernels.ops.attention.decode_attention_mxfp4_sm120 as mod
from sglang.srt.layers.quantization.kvfp4_tensor import MXFP4KVQuantizeUtil

DEV = "cuda"
PS = 64  # must be a multiple of mod._BLOCK_N for the paged path to engage


def build_mapping(seq_lens, npages_pool, shuffle, seed):
    """(kv_indptr, kv_indices, page_table) for one shared slot mapping.

    ``kv_indices`` is token-granular (one slot id per token), ``page_table`` is
    page-granular (one page id per page). Both describe the same mapping, so the
    two kernel paths must produce identical output.
    """
    g = torch.Generator(device=DEV).manual_seed(seed)
    max_pages = max(-(-s // PS) for s in seq_lens)
    rows, tok_chunks, indptr = [], [], [0]
    for s in seq_lens:
        np_req = -(-s // PS)
        if shuffle:
            perm = torch.randperm(npages_pool, device=DEV, generator=g)[:np_req]
        else:
            perm = torch.arange(np_req, device=DEV)
        row = torch.zeros(max_pages, dtype=torch.int32, device=DEV)
        row[:np_req] = perm.to(torch.int32)
        rows.append(row)
        toks = (
            perm.to(torch.int64)[:, None] * PS + torch.arange(PS, device=DEV)[None, :]
        ).reshape(-1)[:s]
        tok_chunks.append(toks.to(torch.int32))
        indptr.append(indptr[-1] + s)
    return (
        torch.tensor(indptr, dtype=torch.int32, device=DEV),
        torch.cat(tok_chunks).contiguous(),
        torch.stack(rows).contiguous(),
    )


def run(case, seq_lens, q_heads, kv_heads, head_dim, shuffle):
    batch = len(seq_lens)
    npages_pool = max(8, 2 * max(-(-s // PS) for s in seq_lens))
    slots = npages_pool * PS
    g = torch.Generator(device=DEV).manual_seed(99)
    ref = torch.randn(
        (slots, kv_heads, head_dim), dtype=torch.bfloat16, device=DEV, generator=g
    )
    ref_v = torch.randn(
        (slots, kv_heads, head_dim), dtype=torch.bfloat16, device=DEV, generator=g
    )
    kp, ks = MXFP4KVQuantizeUtil.batched_quantize(ref)
    vp, vs = MXFP4KVQuantizeUtil.batched_quantize(ref_v)
    q = torch.randn(
        (batch, q_heads, head_dim), dtype=torch.bfloat16, device=DEV, generator=g
    )
    kv_indptr, kv_indices, page_table = build_mapping(
        seq_lens, npages_pool, shuffle, seed=5
    )

    def fwd(page_size, pt):
        o = torch.zeros((batch, q_heads, head_dim), dtype=torch.float32, device=DEV)
        mod.mxfp4_decode_attention_fwd(
            q,
            kp,
            vp,
            ks,
            vs,
            o,
            kv_indptr,
            kv_indices,
            1.0 / head_dim**0.5,
            page_size=page_size,
            page_table=pt,
        )
        return o

    o_tok = fwd(1, None)
    o_pag = fwd(PS, page_table)
    same = torch.equal(o_tok, o_pag)
    mad = (o_tok - o_pag).abs().max().item()

    # Negative control. Agreement alone proves nothing if the paged path never
    # engaged -- it would just be the token path twice. Pointing every entry at a
    # different (still in-bounds) page must change the output of a path that
    # actually reads the table; if it does not, the arm silently fell back.
    # Rolling would not do: a single-page sequence rolls to itself.
    engaged = not torch.equal(o_tok, fwd(PS, (page_table + 1) % npages_pool))
    # Head shape no longer gates this: the per-head kernel that had no page-table
    # argument is gone, so kv_group_num == 1 runs the grouped kernel too.
    expect_engaged = PS % mod._BLOCK_N == 0

    tag = "shuffled" if shuffle else "identity"
    ok = same and engaged == expect_engaged
    print(
        f"{case:<34} {tag:<9} bitwise={'OK ' if same else 'DIFF'} "
        f"max_abs_diff={mad:.3e} paged_engaged={str(engaged):<5} "
        f"(expected {expect_engaged})"
    )
    return ok


CASES = [
    # (name, seq_lens, q_heads, kv_heads, head_dim)
    ("qwen3 24/4 hd256 one page", [64], 24, 4, 256),
    ("qwen3 24/4 hd256 partial page", [100], 24, 4, 256),
    ("qwen3 24/4 hd256 multi page", [577], 24, 4, 256),
    ("qwen3 24/4 hd256 batch4 ragged", [64, 191, 512, 1000], 24, 4, 256),
    ("gqa 16/2 hd128", [333], 16, 2, 128),
    ("gqa 8/1 hd64", [777], 8, 1, 64),
    # kv_group_num == 1 is the degenerate grid: VALID_BLOCK_H == 1, so one row
    # of the BLOCK_H tile is valid. Narrowest masking path, and the one the
    # since-removed per-head kernel used to serve.
    ("mha 4/4 hd128 (kv_group_num 1)", [300], 4, 4, 128),
]

if __name__ == "__main__":
    ok = True
    for shuffle in (False, True):
        for name, sl, qh, kh, hd in CASES:
            ok &= run(name, sl, qh, kh, hd, shuffle)
        print()
    print("ALL BITWISE EQUAL" if ok else "MISMATCH")
