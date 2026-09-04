# SPDX-License-Identifier: Apache-2.0
"""Standalone 5-sample HumanEval evaluator with per-sample persistence.

Created because ``run_eval``'s report files use a fixed name
(``humaneval__<model>.json/html``) and silently overwrite each other across
runs/configurations, which destroyed an earlier baseline run's artifacts.

Properties:
- Same 60-problem sample as ``run_eval --eval-name humaneval`` (random.Random(0)
  over ``read_problems()``).
- ``--num-samples`` completions per problem through the same
  ChatCompletionSampler path (temperature 0; the sample spread comes from the
  server-side batch-state non-determinism, same as the 5-sample runs via
  run_eval).
- Extraction is think-block aware (strip <think>..</think>, matching the fixed
  ``simple_eval_humaneval.find_code``).
- Scoring: direct exec of prompt + completion + test (same semantics as
  human_eval's check_correctness without the subprocess sandbox).
- Output: ``eval_results_l3/humaneval_<tag>_<ts>.jsonl`` (one line per sample:
  task_id, sample_idx, passed, error, completion) plus a ``.summary.json``.
  NEVER overwrites: an existing path gets a counter suffix.
"""

import argparse
import json
import os
import random
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from human_eval.data import read_problems

from sglang.test.simple_eval_common import ChatCompletionSampler

INSTR = (
    "Read the following function signature and docstring, and fully implement "
    "the function described. Your response should only contain the code for "
    "this function.\n"
)


def find_code(completion: str) -> str:
    """Think-block aware extraction (mirror of fixed simple_eval_humaneval)."""
    completion = completion or ""
    if "</think>" in completion:
        completion = completion.rsplit("</think>", 1)[-1]
    completion = completion.split("<think>")[0]
    matches = re.findall(r"```python\n(.*?)```", completion, re.DOTALL)
    extracted = matches[0] if len(matches) >= 1 else completion
    extracted = extracted[extracted.find(":\n    ") + 2 :]
    return extracted


def check(problem: dict, completion: str):
    try:
        ns = {}
        exec(problem["prompt"] + "\n" + completion, ns)
        exec(problem["test"], ns)
        return True, ""
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


def unique_path(path: str) -> str:
    if not os.path.exists(path):
        return path
    stem, ext = os.path.splitext(path)
    for i in range(1, 1000):
        cand = f"{stem}_{i}{ext}"
        if not os.path.exists(cand):
            return cand
    raise RuntimeError("cannot find a free output path")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=30001)
    ap.add_argument("--model", required=True)
    ap.add_argument("--num-examples", type=int, default=60)
    ap.add_argument("--num-samples", type=int, default=5)
    ap.add_argument("--num-threads", type=int, default=4)
    ap.add_argument("--tag", required=True, help="config tag, e.g. C_mxfp4_native")
    ap.add_argument("--out-dir", default="/sgl-workspace/sglang/eval_results_l3")
    args = ap.parse_args()

    problems = read_problems()
    examples = random.Random(0).sample(list(problems.values()), args.num_examples)

    sampler = ChatCompletionSampler(
        base_url=f"http://{args.host}:{args.port}/v1",
        model=args.model,
        max_tokens=2048,
        temperature=0.0,
    )

    ts = time.strftime("%Y%m%d-%H%M%S")
    out_path = unique_path(
        os.path.join(args.out_dir, f"humaneval_{args.tag}_{ts}.jsonl")
    )
    summary_path = unique_path(
        os.path.join(args.out_dir, f"humaneval_{args.tag}_{ts}.summary.json")
    )
    print(f"output: {out_path}")

    jobs = [(ex, k) for ex in examples for k in range(args.num_samples)]
    n_pass = 0
    per_task = {}
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.num_threads) as pool:
        futures = {
            pool.submit(
                sampler, [{"role": "user", "content": INSTR + ex["prompt"]}]
            ): (ex, k)
            for ex, k in jobs
        }
        with open(out_path, "w") as fout:
            for fut in as_completed(futures):
                ex, k = futures[fut]
                task_id = ex["task_id"]
                try:
                    response = fut.result()
                except Exception as e:
                    response = ""
                    err = f"API {type(e).__name__}: {e}"
                else:
                    err = ""
                completion = find_code(response)
                passed, cerr = check(ex, completion)
                if not err:
                    err = cerr
                n_pass += int(passed)
                per_task.setdefault(task_id, []).append(int(passed))
                fout.write(
                    json.dumps(
                        {
                            "task_id": task_id,
                            "sample_idx": k,
                            "passed": passed,
                            "error": err,
                            "completion": completion,
                            "raw_response": response,
                        }
                    )
                    + "\n"
                )
                fout.flush()
                done = sum(len(v) for v in per_task.values())
                print(
                    f"[{done}/{len(jobs)}] {task_id}#{k}: "
                    f"{'PASS' if passed else 'FAIL'} {err[:60]}",
                    flush=True,
                )

    elapsed = time.time() - t0
    summary = {
        "tag": args.tag,
        "num_examples": args.num_examples,
        "num_samples_per_task": args.num_samples,
        "pass@1": n_pass / len(jobs),
        "n_pass": n_pass,
        "n_total": len(jobs),
        "elapsed_s": elapsed,
        "model": args.model,
        "per_task_pass": per_task,
    }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(
        f"\nFINAL {args.tag}: pass@1 = {n_pass}/{len(jobs)} = "
        f"{summary['pass@1']:.4f}  ({elapsed:.0f}s)"
    )
    print(f"summary: {summary_path}")


if __name__ == "__main__":
    main()
