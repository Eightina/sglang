# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Language

All communication with the user must be in **Chinese (中文)**, while keeping technical terms, code, commands, file paths, and identifiers in English. Thinking/reasoning may be in either Chinese or English.

## Repository Layout

SGLang is a high-performance LLM serving framework. The repo has two main code trees:

- **`python/sglang/`** — Python package (`sglang`). The runtime core lives in `python/sglang/srt/` ("SGLang Runtime"):
  - `srt/managers/` — Multi-process orchestration: `TokenizerManager` (HTTP + tokenization), `Scheduler` (request scheduling + batching), `TpWorker`
  - `srt/model_executor/` — `ModelRunner` and CUDA graph runners (breakable, piecewise, CPU)
  - `srt/models/` — Per-model implementations (llama, qwen, deepseek_v2/v4, etc.)
  - `srt/layers/` — Reusable layers: `attention/` (many backends), `moe/`, `quantization/`, `linear.py`, `radix_attention.py`
  - `srt/mem_cache/` — KV cache: radix tree, HiCache, SWA, Mamba, hybrid caches
  - `srt/speculative/` — Speculative decoding (EAGLE, ngram, frozen-KV MTP, draft workers)
  - `srt/disaggregation/` — Prefill-decode disaggregation (Mooncake, NIXL, Mori backends)
  - `srt/distributed/` — Parallelism state (TP/PP/EP/DP)
  - `srt/entrypoints/` — HTTP, gRPC, Ollama, OpenAI-compatible servers
  - `srt/environ.py` — All `SGLANG_*` env vars (single source of truth)
  - `srt/server_args.py` — CLI argument definitions (very large; uses arg groups)
- **`python/sglang/jit_kernel/`** — JIT-compiled Triton kernels (with own tests under `tests/`)
- **`sgl-kernel/`** — Separate CUDA/C++ kernel library. Published as `sglang-kernel` on PyPI, imported as `sgl_kernel`. Has its own `CMakeLists.txt`, `csrc/`, `include/`, `tests/`, `benchmark/`.
- **`test/`** — Test tree. `test/registered/` is CI-discovered; `test/registered/unit/` mirrors `srt/` for unit tests; `test/manual/` is non-CI.
- **`benchmark/`**, **`scripts/`**, **`docs/`**, **`docs_new/`**, **`examples/`**

## Build & Install

```bash
# Python package (editable)
pip install -e "python"

# sgl-kernel (from sgl-kernel/ directory)
cd sgl-kernel && make build
# Limit resources:
make build MAX_JOBS=2 CMAKE_ARGS="-DSGL_KERNEL_COMPILE_THREADS=1"
```

## Linting

```bash
pip install pre-commit && pre-commit install
pre-commit run --all-files     # run all checks; re-run if first pass fails
```

## Testing

**Unit tests (no server, no model weights):**
```bash
pytest test/registered/unit/ -v
pytest test/registered/unit/mem_cache/ -v        # one module
pytest test/registered/unit/ --cov --cov-config=.coveragerc -v   # with coverage
```

**Single E2E test file:**
```bash
python3 test/registered/core/test_srt_endpoint.py
python3 test/registered/core/test_srt_endpoint.py TestSRTEndpoint.test_simple_decode
```

**JIT kernel tests (live outside `test/`):**
```bash
python3 python/sglang/jit_kernel/tests/test_add_constant.py
```

**Suite runner (CI-style):**
```bash
python3 test/run_suite.py --hw cuda --suite base-b-test-1-gpu-small
python3 test/run_suite.py --hw cpu --suite base-a-test-cpu
```

**CI registration** — every test file under `test/registered/` must call a registration function at module level (e.g. `register_cuda_ci(est_time=80, stage="base-b", runner_config="1-gpu-small")`). Parameters must be literals (AST-parsed by `run_suite.py`). See `test/README.md` and `.claude/skills/write-sglang-test/SKILL.md` for suite tables and templates.

**Accuracy sanity check:**
```bash
python3 -m sglang.launch_server --model Qwen/Qwen2-7B-Instruct
python3 -m sglang.test.few_shot_gsm8k --num-questions 200
```

## Architecture Essentials

**Multi-process model.** A launch spawns: `TokenizerManager` (FastAPI + tokenization + metrics) ↔ ZMQ ↔ `Scheduler` (one per TP rank, owns the batch loop) ↔ `ModelRunner` (GPU forward + CUDA graphs). The `Scheduler` and `ModelRunner` communicate via shared memory / IPC tensors. `TokenizerManager` is the only process that talks to clients.

**RadixAttention.** The signature feature: a radix-tree prefix cache in `srt/mem_cache/radix_cache.py` (and C++ variant `cpp_radix_tree/`) that shares KV blocks across requests with common prefixes. Variants: `hiradix_cache` (HiCache, hierarchical with host offload), `swa_radix_cache` (sliding window), `mamba_radix_cache` (SSM state).

**Attention backends.** Pluggable via `srt/layers/attention/`. Active backends: FlashInfer (default on NVIDIA), FlashAttention, Triton, cutlass MLA, TRT-LLM MHA/MLA, FlashMLA, NSA, DSA, plus AMD (AITER, Wave), Intel AMX, XPU. Selection is per-architecture and per-quantization.

**Quantization.** `srt/layers/quantization/` holds ~30 schemes. FP8, MXFP4, NVFP4, W8A8, AWQ, GPTQ, INT4, Quark, etc. Each has its own linear/gemm path and often its own MoE variant.

**MoE.** `srt/layers/moe/` has many fused implementations: cutlass, FlashInfer+TRT-LLM, FlashInfer+cuteDSL, Marlin, Triton, plus expert-parallel dispatch under `srt/eplb/` and `srt/layers/moe/token_dispatcher/`.

**Speculative decoding.** `srt/speculative/` — EAGLE (v1/v2), multi-layer EAGLE, ngram, frozen-KV MTP, standalone workers. Each has its own draft worker, info class, and CUDA graph runner. **Must read `.claude/skills/speculative-naming/SKILL.md` before modifying.**

**Disaggregation.** `srt/disaggregation/` — splits prefill and decode onto different GPU pools. Backends: Mooncake, NIXL, Mori. Mixins attach to `Scheduler` (`SchedulerDisaggregationPrefillMixin`, `SchedulerDisaggregationDecodeMixin`).

**Parallelism.** TP (tensor), PP (pipeline), EP (expert), DP (data), DP attention — all configured through `srt/distributed/parallel_state.py` and consumed via `get_tp_group()`, `get_pp_group()`, etc. DP attention is in `srt/layers/dp_attention.py`.

**CUDA graphs.** `srt/model_executor/cuda_graph_runner.py` and variants. "Breakable" graphs allow partial recapture when shapes change; "piecewise" handles PP stages.

## Component-Specific Rules

**Before modifying these components, read the listed skill first** (from `.claude/rules/modify-component-must-read.md`):

| Component | Skill to read |
|-----------|---------------|
| `srt/speculative/` or related IPC/CLI flags | `speculative-naming` |
| `__init__` of `Scheduler`, `TokenizerManager`, or `ModelRunner` | `large-class-init-style` |
| Any `SGLANG_*` env var, or `srt/environ.py` | `env-var-conventions` |
| Scripted runtime | `scripted-runtime-notes` |

## Adding a New Kernel

**JIT kernel** (Triton, compiled at first use): implement under `python/sglang/jit_kernel/`, add tests under `jit_kernel/tests/test_*.py`, register with `register_cuda_ci(stage="base-b", runner_config="kernel-unit-1-gpu-large")`. See `.claude/skills/add-jit-kernel/SKILL.md`.

**AOT kernel** (CUDA/C++, shipped in `sglang-kernel` wheel): implement in `sgl-kernel/csrc/`, expose via `include/sgl_kernel_ops.h`, register torch extension in `csrc/common_extension.cc` (use `m.def` + `m.impl` with schema), update `CMakeLists.txt`, add Python binding under `sgl-kernel/python/sgl_kernel/`. See `.claude/skills/add-sgl-kernel/SKILL.md` and `sgl-kernel/README.md`.

## Server Entry Points

- `sglang serve --model-path <model> [options]` — recommended CLI (new)
- `python3 -m sglang.launch_server --model-path <model>` — still works, emits deprecation warning
- `python3 -m sglang.bench_serving` — benchmark a running server
- `python3 -m sglang.bench_offline_throughput` — throughput without HTTP overhead

## Key Files to Read First

- `python/sglang/srt/server_args.py` — every CLI flag, grouped by arg group
- `python/sglang/srt/managers/scheduler.py` — the batch loop (very large; mixins split concerns)
- `python/sglang/srt/managers/tokenizer_manager.py` — request lifecycle
- `python/sglang/srt/model_executor/model_runner.py` — weight loading + forward dispatch
- `python/sglang/srt/layers/attention/base_attn_backend.py` — backend interface
- `python/sglang/srt/environ.py` — env var catalog
- `test/README.md` — CI system overview
- `.claude/skills/` — 20+ skills covering CI, profiling, incident triage, kernel authoring, naming conventions
