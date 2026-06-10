---
name: start-server
description: >
  Start SGLang inference server locally with the project's environment
  setup (venv, proxy, ModelScope, HPC-X UCX). Use this skill whenever
  the user asks to start, launch, or bring up an SGLang serving instance
  on this machine — including single-server TP mode and PD disaggregation
  mode with prefill/decode workers + router. Also use it when the user
  says "start the server", "launch tp4", "bring up pd mode", or
  references any pyscripts/q-*.yaml config.
---

# Start SGLang Inference Server (Local)

This skill captures the complete, reproducible procedure for launching
SGLang inference servers on this machine. Two modes are supported:

- **Single-server mode** — one `launch_server` process, TP=4 across all 4 GPUs.
- **PD disaggregation mode** — separate prefill (TP=2) + decode (TP=2) workers,
  connected by a `sglang_router` process.

Both modes share the same environment setup (venv activation, proxy,
ModelScope, HPC-X UCX). The sections below walk through every step.

## ⚠️ 铁律：不许改用户的命令

**严格按照用户给出的命令执行，一个字都不许改。** 如果遇到报错：
1. 原样报告错误信息
2. 问用户怎么办
3. **绝不**自作主张修改命令参数（包括 URL、IP、端口等）

用户的命令就是对的。如果跑不通，是我的环境问题，不是用户的命令问题。

---

## Quick Reference

```bash
# Single-server TP=4
cd /root/sglang && source .venv/bin/activate
# <set env vars — see Step 1>
no_proxy="*" python -m sglang.launch_server --config ./pyscripts/q-tp4.yaml
# Verify:
python3 ./pyscripts/req.py --host 127.0.0.1 --port 8000 --stream

# PD disaggregation (prefill tp=2 + decode tp=2 + router)
cd /root/sglang && source .venv/bin/activate
# <set env vars — see Step 1>
no_proxy="*" python -m sglang.launch_server --config ./pyscripts/q-prefilltp2.yaml
no_proxy="*" python -m sglang.launch_server --config ./pyscripts/q-decodetp2.yaml
python -m sglang_router.launch_router \
  --pd-disaggregation \
  --prefill "http://0.0.0.0:30000" \
  --decode  "http://0.0.0.0:30001" \
  --policy round_robin \
  --host 0.0.0.0 --port 8000
# Verify:
python3 ./pyscripts/req.py --host 127.0.0.1 --port 8000 --stream
```

---

## Step 1: Environment Setup (Required for Both Modes)

Every launch must run these in the shell that will start the server.
**Do not skip any of these** — the proxy is needed for model download,
ModelScope is the weight source, and HPC-X UCX is needed by the NIXL
transfer backend in PD mode.

```bash
# 1a. Enter repo and activate venv
cd /root/sglang
source .venv/bin/activate

# 1b. Proxy (lab squid — required for model download)
export http_proxy="http://lab22-squid.eng.xrvm.cn:3128"
export https_proxy="http://lab22-squid.eng.xrvm.cn:3128"
export no_proxy=localhost,127.0.0.1,.local,.xrvm.cn,10.244.1.235,0.0.0.0
export HTTP_PROXY="http://lab22-squid.eng.xrvm.cn:3128"
export HTTPS_PROXY="http://lab22-squid.eng.xrvm.cn:3128"

# 1c. Weight source (ModelScope mirror)
export SGLANG_USE_MODELSCOPE=true
export HF_ENDPOINT=https://hf-mirror.com

# 1d. HPC-X UCX (required for NIXL transfer backend)
export PATH=/opt/hpcx/ucx/bin:$PATH
export LD_LIBRARY_PATH=/opt/hpcx/ucx/lib:/opt/hpcx/ucx/lib/ucx:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
export UCX_MODULE_DIR=/opt/hpcx/ucx/lib/ucx
```

### Why each variable matters

| Variable | Why |
|---|---|
| `http_proxy` / `https_proxy` | Model downloads go through lab squid; without it, HuggingFace/ModelScope is unreachable. |
| `no_proxy` | Localhost and lab-internal hosts must bypass the proxy, otherwise the server's own health checks and inter-worker traffic get routed through squid. |
| `SGLANG_USE_MODELSCOPE=true` | Tells SGLang to pull weights from ModelScope instead of HuggingFace (HuggingFace direct is blocked). |
| `HF_ENDPOINT` | Mirror endpoint for any residual HF client calls. |
| `PATH` / `LD_LIBRARY_PATH` / `UCX_MODULE_DIR` | NIXL transfer backend loads UCX from HPC-X; without these, PD mode fails to initialize the transfer backend. |

---

## Step 2a: Single-Server Mode (TP=4)

Uses all 4 GPUs (GPU 0, 1, 2, 3) in one process.

### Config file: `pyscripts/q-tp4.yaml`

```yaml
model-path: qwen/qwen3.5-27b-fp8
host: 0.0.0.0
port: 8000
tensor-parallel-size: 4
enable-metrics: true
log-requests: true
```

### Launch command

```bash
no_proxy="*" python -m sglang.launch_server --config ./pyscripts/q-tp4.yaml
```

> **Why `no_proxy="*"` inline?** The server process itself should never
> route its internal traffic (health checks, metrics, TP collectives)
> through the proxy. The env var `no_proxy` from Step 1 already covers
> localhost, but setting `no_proxy="*"` for the launch command is a
> belt-and-suspenders safeguard.

### What to wait for

The server prints a lot of startup logs. Look for this line:

```
The server is fired up and ready to roll.
```

If it hangs or errors, check:
- GPU memory: `nvidia-smi` — all 4 GPUs should show ~7GB used for a 27B FP8 model
- Port 8000 free: `ss -tlnp | grep 8000`

### Verification

In a **separate terminal** (with the same venv activated, env vars don't
matter for the client):

```bash
cd /root/sglang && source .venv/bin/activate
python3 ./pyscripts/req.py --host 127.0.0.1 --port 8000 --stream
```

Expected: a streaming chat completion response ending with `Latency: X.XXXs`.

---

## Step 2b: PD Disaggregation Mode (Prefill TP=2 + Decode TP=2 + Router)

Three processes total:

| Process | Config | GPUs | Port |
|---|---|---|---|
| Prefill worker | `pyscripts/q-prefilltp2.yaml` | GPU 0, 1 (`base-gpu-id: 0`, tp=2) | 30000 |
| Decode worker | `pyscripts/q-decodetp2.yaml` | GPU 2, 3 (`base-gpu-id: 2`, tp=2) | 30001 |
| Router | (CLI flags below) | — | 8000 |

### Config files

`pyscripts/q-prefilltp2.yaml`:
```yaml
model-path: qwen/qwen3.5-27b-fp8
host: 0.0.0.0
port: 30000
tensor-parallel-size: 2
base-gpu-id: 0
disaggregation-mode: prefill
disaggregation-transfer-backend: nixl
enable-metrics: true
log-requests: true
```

`pyscripts/q-decodetp2.yaml`:
```yaml
model-path: qwen/qwen3.5-27b-fp8
host: 0.0.0.0
port: 30001
tensor-parallel-size: 2
base-gpu-id: 2
disaggregation-mode: decode
disaggregation-transfer-backend: nixl
enable-metrics: true
log-requests: true
```

### Launch commands

Run each in its own terminal (or background with `&`):

**Terminal 1 — Prefill worker:**
```bash
cd /root/sglang && source .venv/bin/activate
# <Step 1 env vars>
no_proxy="*" python -m sglang.launch_server --config ./pyscripts/q-prefilltp2.yaml
```

**Terminal 2 — Decode worker:**
```bash
cd /root/sglang && source .venv/bin/activate
# <Step 1 env vars>
no_proxy="*" python -m sglang.launch_server --config ./pyscripts/q-decodetp2.yaml
```

Wait for both to print `The server is fired up and ready to roll.`

**Terminal 3 — Router:**
```bash
cd /root/sglang && source .venv/bin/activate
python -m sglang_router.launch_router \
  --pd-disaggregation \
  --prefill "http://0.0.0.0:30000" \
  --decode  "http://0.0.0.0:30001" \
  --policy round_robin \
  --host 0.0.0.0 --port 8000
```

> **Note:** The router does **not** need `no_proxy="*"` or the UCX env
> vars — it's a pure HTTP proxy between clients and the workers.

### What to wait for

- Both workers print `The server is fired up and ready to roll.`
- Router prints something like `Router started on port 8000`

### Verification

```bash
cd /root/sglang && source .venv/bin/activate
python3 ./pyscripts/req.py --host 127.0.0.1 --port 8000 --stream
```

---

## Step 3: Print a Launch Summary

**After the server is verified working, print all of the following to the
user. Do not hide or abbreviate any of it.** This is critical for
reproducibility and debugging.

### For single-server mode, print:

```
========== SGLang Server Launched ==========
Mode:              single-server
Config file:       pyscripts/q-tp4.yaml
Model:             qwen/qwen3.5-27b-fp8
GPUs:              0, 1, 2, 3 (TP=4)
Listen:            0.0.0.0:8000
Metrics:           enabled
Transfer backend:  n/a

Launch command:
  no_proxy="*" python -m sglang.launch_server --config ./pyscripts/q-tp4.yaml

Verify command:
  python3 ./pyscripts/req.py --host 127.0.0.1 --port 8000 --stream
================================================
```

### For PD disaggregation mode, print:

```
========== SGLang PD Disaggregation Launched ==========
Mode:                 PD disaggregation
Prefill config:       pyscripts/q-prefilltp2.yaml
Decode config:        pyscripts/q-decodetp2.yaml
Model:                qwen/qwen3.5-27b-fp8
Prefill GPUs:         0, 1 (TP=2, base-gpu-id=0)
Decode GPUs:          2, 3 (TP=2, base-gpu-id=2)
Prefill port:         30000
Decode-port:          30001
Router-port:          8000
Transfer backend:     nixl
Routing policy:       round_robin

Launch commands:
  # Terminal 1 — prefill
  no_proxy="*" python -m sglang.launch_server --config ./pyscripts/q-prefilltp2.yaml

  # Terminal 2 — decode
  no_proxy="*" python -m sglang.launch_server --config ./pyscripts/q-decodetp2.yaml

  # Terminal 3 — router
  python -m sglang_router.launch_router \
    --pd-disaggregation \
    --prefill "http://0.0.0.0:30000" \
    --decode  "http://0.0.0.0:30001" \
    --policy round_robin \
    --host 0.0.0.0 --port 8000

Verify command:
  python3 ./pyscripts/req.py --host 127.0.0.1 --port 8000 --stream
==========================================================
```

---

## Troubleshooting

### "The server is fired up" never appears

- Check `nvidia-smi` — are the expected GPUs using memory?
- Check for port conflicts: `ss -tlnp | grep <port>`
- Look at the last few lines of the log — common errors are OOM, weight
  download failure (proxy misconfigured), or NIXL backend init failure
  (UCX env vars missing).

### `req.py` times out or connection refused

- Verify the port matches: tp4 → 8000, PD → 8000 (router), prefill-only → 30000, decode-only → 30001.
- In PD mode, make sure the **router** is running, not just the workers.
- Try the worker directly: `python3 ./pyscripts/req.py --host 127.0.0.1 --port 30000 --stream` (prefill) or `--port 30001` (decode).

### Weights fail to download

- Confirm proxy works: `curl -x http://lab22-squid.eng.xrvm.cn:3128 -I https://modelscope.cn`
- Confirm ModelScope env: `echo $SGLANG_USE_MODELSCOPE` should print `true`.

### PD mode: NIXL backend errors

- Confirm UCX env vars are set in the worker shells (not just the router shell).
- Verify: `python -c "import nixl_cu13; print('OK')"`.
- Check GPU assignment doesn't overlap: prefill `base-gpu-id: 0` + tp=2 → GPU 0,1; decode `base-gpu-id: 2` + tp=2 → GPU 2,3. **If both use the same `base-gpu-id`, they will fight for the same GPUs.**

### Want to stop the server

- Single server: Ctrl-C in its terminal, or `pkill -f launch_server`.
- PD mode: Ctrl-C each of the three terminals, or:
  ```bash
  pkill -f sglang_router.launch_router
  pkill -f 'launch_server.*q-prefilltp2'
  pkill -f 'launch_server.*q-decodetp2'
  ```
- Confirm GPUs are freed: `nvidia-smi` should show 0 MiB used on all GPUs.
