"""Offline HumanEval rescorer: re-extract completions from the report HTML
with the fixed ``find_code`` (think-block aware) and re-run the test suites.

The original runs scored 0.017 (mxfp4 native) / 0.233 (fp8 baseline) because
``find_code`` did not strip the model's <think> block: its signature-based
extraction keyed on the first ":\\n    " occurrence, which the reasoning text
routinely contains, so most completions were extracted from the middle of the
reasoning and failed exec. The completions themselves are correct (verified by
replaying HumanEval/10 locally: TESTS PASSED).

Usage: python scripts/playground/rescore_humaneval_html.py <report.html>
"""

import re
import sys

from human_eval.data import read_problems


def find_code(completion: str) -> str:
    """Mirror of the fixed simple_eval_humaneval.find_code."""
    completion = completion or ""
    if "</think>" in completion:
        completion = completion.rsplit("</think>", 1)[-1]
    completion = completion.split("<think>")[0]
    matches = re.findall(r"```python\n(.*?)```", completion, re.DOTALL)
    extracted = matches[0] if len(matches) >= 1 else completion
    extracted = extracted[extracted.find(":\n    ") + 2 :]
    return extracted


def main(html_path: str):
    html = open(html_path).read()
    text = re.sub(r"<[^>]+>", "\n", html)
    for a, b in (
        ("&#34;", '"'),
        ("&#39;", "'"),
        ("&gt;", ">"),
        ("&lt;", "<"),
        ("&amp;", "&"),
    ):
        text = text.replace(a, b)
    # NOTE: no whitespace normalization — dataset prompts contain trailing
    # spaces that must survive for exact prompt matching.

    problems = read_problems()
    # dataset prompts carry leading newlines ("\ndef ..."); the html-extracted
    # prompt is stripped, so index by the stripped prompt for exact matching.
    by_prompt = {
        p["prompt"].strip(): (tid, p) for tid, p in problems.items()
    }
    # The report embeds each problem's prompt after the fixed instruction.
    INSTR = (
        "Read the following function signature and docstring, and fully "
        "implement the function described. Your response should only contain "
        "the code for this function.\n"
    )

    segments = text.split("Sampled message")[1:]
    n_pass = 0
    n_total = 0
    missing_prompt = 0
    for seg in segments:
        # response = from segment start to its "Results" line
        r_idx = seg.find("Results\n")
        if r_idx < 0:
            continue
        response = seg[:r_idx].strip()
        # prompt = the dataset prompt embedded after the fixed instruction
        i_inst = seg.find(INSTR)
        prompt = seg[i_inst + len(INSTR) :].strip() if i_inst >= 0 else ""
        hit = by_prompt.get(prompt)
        if hit is None:
            # tolerate trailing whitespace differences: match by prefix
            for p_prompt, item in by_prompt.items():
                if prompt and prompt.startswith(p_prompt[:200]):
                    hit = item
                    break
        if hit is None:
            missing_prompt += 1
            continue
        task_id, problem = hit
        completion = find_code(response)
        full = problem["prompt"] + "\n" + completion
        ok = False
        try:
            ns = {}
            exec(full, ns)
            exec(problem["test"], ns)
            ok = True
        except Exception:
            ok = False
        n_total += 1
        n_pass += int(ok)
        print(f"{task_id}: {'PASS' if ok else 'FAIL'}")

    print(f"\nrescored {n_total} samples (prompt-unmatched: {missing_prompt})")
    if n_total:
        print(f"pass@1 = {n_pass}/{n_total} = {n_pass / n_total:.4f}")


if __name__ == "__main__":
    main(sys.argv[1])
