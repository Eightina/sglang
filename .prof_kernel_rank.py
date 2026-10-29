#!/usr/bin/env python3
"""Server-side kernel profiling for one long-context request.

Drives /start_profile (profile_by_stage) + /generate, then ranks CUDA kernels by
total device time from the emitted DECODE / EXTEND traces.

    python .prof_kernel_rank.py <port> <input_len> <out_dir> [new_tokens]
"""
import glob
import gzip
import json
import os
import sys
import time
import urllib.request

PORT = sys.argv[1] if len(sys.argv) > 1 else "30000"
INPUT_LEN = int(sys.argv[2]) if len(sys.argv) > 2 else 98000
OUT = sys.argv[3] if len(sys.argv) > 3 else "/tmp/prof/mxfp4"
NEW_TOKENS = int(sys.argv[4]) if len(sys.argv) > 4 else 48
DECODE_STEPS = 32

BASE = f"http://localhost:{PORT}"


def post(path, payload, timeout=1200):
    req = urllib.request.Request(
        BASE + path, json.dumps(payload).encode(), {"Content-Type": "application/json"}
    )
    body = urllib.request.urlopen(req, timeout=timeout).read()
    try:
        return json.loads(body) if body.strip() else {}
    except Exception:
        return {"_text": body.decode(errors="replace")}


# This filler tokenizes at ~1.0 token/word, so INPUT_LEN words ~= INPUT_LEN tokens.
words = (
    "the quick brown fox jumps over lazy dog while sun sets behind "
    "mountain river flows through valley".split()
)
text = " ".join(words[i % len(words)] for i in range(INPUT_LEN))

os.makedirs(OUT, exist_ok=True)
# A request that 400s/500s mid-profile leaves the profiler armed, and the next
# /start_profile then fails with "Profiling is already in progress".
try:
    post("/stop_profile", {})
except Exception:
    pass

post(
    "/start_profile",
    {
        "output_dir": OUT,
        "activities": ["CPU", "GPU"],
        "profile_by_stage": True,
        "num_steps": DECODE_STEPS,
    },
)
t0 = time.time()
r = post(
    "/generate",
    {"text": text, "sampling_params": {"max_new_tokens": NEW_TOKENS, "temperature": 0}},
)
wall = time.time() - t0
mi = r["meta_info"]
print(
    f"req done: wall={wall:.2f}s prompt={mi.get('prompt_tokens')} "
    f"new={mi.get('completion_tokens')}"
)


def load_trace(path, retries=12, delay=5):
    """Traces are flushed asynchronously after the request returns."""
    opener = gzip.open if path.endswith(".gz") else open
    for _ in range(retries):
        try:
            with opener(path) as f:
                return json.load(f)
        except (EOFError, json.JSONDecodeError):
            time.sleep(delay)
    return None


for path in sorted(
    glob.glob(OUT + "/*.trace.json.gz") + glob.glob(OUT + "/*.trace.json")
):
    trace = load_trace(path)
    name = os.path.basename(path)
    if trace is None:
        print(f"\n== {name}: trace never finished writing ==")
        continue
    kernels = {}
    for e in trace.get("traceEvents", []):
        if e.get("cat") == "kernel":
            k = kernels.setdefault(e["name"], [0, 0.0])
            k[0] += 1
            k[1] += e.get("dur", 0.0)
    if not kernels:
        print(f"\n== {name}: no kernel events ==")
        continue
    total = sum(v[1] for v in kernels.values())
    launches = sum(v[0] for v in kernels.values())
    print(f"\n== {name} | kernel total {total / 1000:.1f} ms | {launches} launches ==")
    for kname, (cnt, dur) in sorted(kernels.items(), key=lambda x: -x[1][1])[:22]:
        print(
            f"{dur / 1000:10.2f} ms  {cnt:7d}x  {100 * dur / total:5.1f}%  {kname[:110]}"
        )
