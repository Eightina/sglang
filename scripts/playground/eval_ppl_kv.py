# SPDX-License-Identifier: Apache-2.0
"""Teacher-forcing PPL evaluation over /generate for KV-cache quality.

Measures how the served KV cache (fp8_e4m3 vs mxfp4 ...) affects the model's
next-token prediction quality over long contexts. For each text segment the
script sends ``input_ids + return_logprob + logprob_start_len=0`` and reads
``meta_info.input_token_logprobs`` — the teacher-forcing NLL of every prompt
token given its KV state — then reports overall PPL plus per-position-bucket
NLL (the KV-quantization signature: error grows with history length).

Output: eval_results_l3/ppl_<tag>_<ts>.jsonl (one line per segment, includes
per-token NLL) + .summary.json. Never overwrites (counter suffix).
"""

import argparse
import glob
import json
import math
import os
import time
import urllib.request

import torch
from transformers import AutoTokenizer

_BUCKET = 2048


def post_generate(host, port, input_ids, timeout=600):
    payload = {
        "input_ids": input_ids,
        "sampling_params": {
            "temperature": 0.0,
            "max_new_tokens": 1,
            "ignore_eos": True,
        },
        "return_logprob": True,
        "logprob_start_len": 0,
    }
    req = urllib.request.Request(
        f"http://{host}:{port}/generate",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def collect_texts(tokenizer, model_path, total_tokens, seg_len):
    """Concatenate repository markdown documents and tokenize up to
    total_tokens, returning `_SEG_COUNT` disjoint segments."""
    candidates = sorted(
        glob.glob("/sgl-workspace/sglang/docs/**/*.md", recursive=True)
        + glob.glob("/sgl-workspace/sglang/*.md")
    )
    ids = []
    used = []
    for path in candidates:
        if len(ids) >= total_tokens:
            break
        try:
            content = open(path, encoding="utf-8", errors="ignore").read()
        except OSError:
            continue
        if len(content) < 2000:
            continue
        seg_ids = tokenizer(content, add_special_tokens=False)["input_ids"]
        ids.extend(seg_ids)
        used.append((path, len(seg_ids)))
    if len(ids) < total_tokens:
        raise RuntimeError(
            f"only {len(ids)} tokens collected from {len(used)} files; need {total_tokens}"
        )
    segments = [ids[i * seg_len : (i + 1) * seg_len] for i in range(total_tokens // seg_len)]
    return segments, used


def unique_path(path):
    if not os.path.exists(path):
        return path
    stem, ext = os.path.splitext(path)
    for i in range(1, 1000):
        cand = f"{stem}_{i}{ext}"
        if not os.path.exists(cand):
            return cand
    raise RuntimeError("no free output path")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=30001)
    ap.add_argument("--model-path", default="/sgl-workspace/models/Qwen3.8-27B-NVFP4")
    ap.add_argument("--seg-len", type=int, default=4096)
    ap.add_argument("--seg-count", type=int, default=8)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out-dir", default="/sgl-workspace/sglang/eval_results_l3")
    args = ap.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    total = args.seg_len * args.seg_count
    segments, used = collect_texts(tokenizer, args.model_path, total, args.seg_len)
    print(f"collected {total} tokens from {len(used)} files; "
          f"{len(segments)} segments x {args.seg_len} tokens")

    ts = time.strftime("%Y%m%d-%H%M%S")
    out_path = unique_path(os.path.join(args.out_dir, f"ppl_{args.tag}_{ts}.jsonl"))
    summary_path = unique_path(
        os.path.join(args.out_dir, f"ppl_{args.tag}_{ts}.summary.json")
    )
    print(f"output: {out_path}")

    summary = {"tag": args.tag, "segments": []}
    with open(out_path, "w") as fout:
        for si, seg_ids in enumerate(segments):
            res = post_generate(args.host, args.port, seg_ids)
            meta = res["meta_info"]
            lp_entries = meta["input_token_logprobs"]
            # each entry: [logprob, token_id]; skip the first token (no context)
            nlls = [-e[0] for e in lp_entries[1:]]
            mean_nll = sum(nlls) / len(nlls)
            ppl = math.exp(mean_nll)
            buckets = []
            for start in range(0, len(nlls), _BUCKET):
                chunk = nlls[start : start + _BUCKET]
                buckets.append(
                    {
                        "pos_range": [start + 1, start + 1 + len(chunk)],
                        "mean_nll": sum(chunk) / len(chunk),
                    }
                )
            record = {
                "tag": args.tag,
                "seg_idx": si,
                "n_scored": len(nlls),
                "mean_nll": mean_nll,
                "ppl": ppl,
                "buckets": buckets,
                "per_token_nll": nlls,
            }
            fout.write(json.dumps(record) + "\n")
            fout.flush()
            summary["segments"].append(
                {
                    "seg_idx": si,
                    "n_scored": len(nlls),
                    "mean_nll": mean_nll,
                    "ppl": ppl,
                    "buckets": buckets,
                }
            )
            print(
                f"seg {si}: n={len(nlls)} mean_nll={mean_nll:.4f} ppl={ppl:.3f} "
                + " ".join(f"[{b['pos_range'][0]}-{b['pos_range'][1]}): {b['mean_nll']:.4f}" for b in buckets)
            )

    overall_nll = sum(s["mean_nll"] * s["n_scored"] for s in summary["segments"]) / sum(
        s["n_scored"] for s in summary["segments"]
    )
    summary["overall"] = {
        "mean_nll": overall_nll,
        "ppl": math.exp(overall_nll),
        "n_tokens": sum(s["n_scored"] for s in summary["segments"]),
    }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(
        f"\nFINAL {args.tag}: overall PPL = {summary['overall']['ppl']:.3f} "
        f"(mean_nll {overall_nll:.4f} over {summary['overall']['n_tokens']} tokens)"
    )
    print(f"summary: {summary_path}")


if __name__ == "__main__":
    main()
