#!/usr/bin/env bash
# ============================================================================
# setup_notes_env.sh — 按 qwen38_attention_survey_notes.md §1 恢复"最终可用组合"
#
# 目标环境: RTX 5090 (SM120) + CUDA 13.0，包装进 /venv/main（portal 服务用的
# 系统 python 不受影响）。全部装完后自动做 import + 版本断言验证。
#
# 用法:
#   ./setup_notes_env.sh                 # 安装缺失/不一致项 + 验证（幂等，可重复跑）
#   ./setup_notes_env.sh --check-only    # 仅验证，不安装
#   PYTHON_BIN=/path/to/python ./setup_notes_env.sh   # 指定目标解释器
#
# 笔记要点（为什么要这些版本）:
#   - sgl-kernel wheel 与 torch 主版本 ABI 硬绑定，错配 import 报 undefined symbol
#   - flashinfer 三件套 (python / jit-cache+cu130 / cubin) 版本必须完全一致
#   - sgl-kernel 0.4.x 不在 PyPI，wheel 来自 github sgl-project/whl releases
# ============================================================================
set -euo pipefail

# ------------------------- 版本区（笔记更新时改这里） -------------------------
TORCH_VERSION="2.13.0+cu130"
TORCHVISION_VERSION="0.28.0+cu130"     # 笔记未记载；torch 2.13.0 配套解析结果（2026-08-31 实测）
SGL_KERNEL_VERSION="0.4.6.post1"
SGL_KERNEL_CU="130"                    # wheel 的 cu 索引（cu130）
FLASHINFER_VERSION="0.6.18"
FLASHINFER_JIT_CACHE="${FLASHINFER_VERSION}+cu130"
TRANSFORMERS_VERSION="5.12.1"
# -----------------------------------------------------------------------------

PYTORCH_INDEX="https://download.pytorch.org/whl/cu130"
FI_JIT_INDEX="https://flashinfer.ai/whl/cu130/"
FI_CUBIN_INDEX="https://flashinfer.ai/whl/"

log()  { echo "[setup-env] $*"; }
die()  { echo "[setup-env][ERROR] $*" >&2; exit 1; }

# ------------------------------- 解释器选择 ----------------------------------
if [[ -n "${PYTHON_BIN:-}" ]]; then
    PY="$PYTHON_BIN"
elif [[ -x /venv/main/bin/python ]]; then
    PY="/venv/main/bin/python"
else
    PY="$(command -v python3)"
    log "[WARN] /venv/main 不存在，回退系统 python3（与 portal 服务共享环境，注意影响）"
fi
log "目标解释器: $PY"

CHECK_ONLY=0
[[ "${1:-}" == "--check-only" ]] && CHECK_ONLY=1

# 已装版本查询（未安装输出空串）
installed_ver() {
    "$PY" - "$1" <<'PYEOF'
import sys
from importlib.metadata import version, PackageNotFoundError
try:
    print(version(sys.argv[1]))
except PackageNotFoundError:
    print("")
PYEOF
}

# 不一致才装，保证幂等
ensure() {  # ensure <dist-name> <expected-version> <pip-args...>
    local dist="$1" expected="$2"; shift 2
    local got; got="$(installed_ver "$dist")"
    if [[ "$got" == "$expected" ]]; then
        log "[skip] $dist==$expected 已满足"
        return 0
    fi
    log "[install] $dist: '$got' -> '$expected'"
    "$PY" -m pip install "$@"
}

# ------------------------------- 1. torch ------------------------------------
# torch 附带的 nvidia-*/triton 传递依赖由 pip 随之解析（cuda 相关，不做 pin）
if [[ "$CHECK_ONLY" -eq 0 ]]; then
    ensure "torch" "$TORCH_VERSION" \
        --upgrade "torch==$TORCH_VERSION" "torchvision==$TORCHVISION_VERSION" \
        --index-url "$PYTORCH_INDEX" --extra-index-url https://pypi.org/simple
fi

# ---------------------------- 2. sgl-kernel ----------------------------------
# wheel 文件名必须保持原始五段 tag 结构（改名会被 pip 拒绝）；
# METADATA 无依赖声明，--no-deps 安装，ABI 由上面的 torch 版本保证
SGL_KERNEL_EXPECTED="${SGL_KERNEL_VERSION}+cu${SGL_KERNEL_CU}"
if [[ "$CHECK_ONLY" -eq 0 ]] && [[ "$(installed_ver sglang-kernel)" != "$SGL_KERNEL_EXPECTED" ]]; then
    WHEEL_NAME="sglang_kernel-${SGL_KERNEL_VERSION}+cu${SGL_KERNEL_CU}-cp310-abi3-manylinux2014_$(uname -m).whl"
    WHEEL_URL="https://github.com/sgl-project/whl/releases/download/v${SGL_KERNEL_VERSION}/${WHEEL_NAME}"
    TMPDIR_DL="$(mktemp -d)"
    trap 'rm -rf "$TMPDIR_DL"' EXIT
    log "[install] 下载 sgl-kernel wheel（~380MB）"
    curl -fL --retry 3 --retry-delay 2 -o "${TMPDIR_DL}/${WHEEL_NAME}" "$WHEEL_URL" \
        || die "sgl-kernel wheel 下载失败: $WHEEL_URL"
    "$PY" -m pip install --force-reinstall --no-deps "${TMPDIR_DL}/${WHEEL_NAME}"
fi

# --------------------------- 3. flashinfer 三件套 ----------------------------
# python 从 PyPI；jit-cache 和 cubin 都在 flashinfer 自建 index（PyPI 版本滞后）。
# 三者版本必须一致，否则 import flashinfer 时版本检查直接报错。
if [[ "$CHECK_ONLY" -eq 0 ]]; then
    ensure "flashinfer-python" "$FLASHINFER_VERSION" \
        "flashinfer-python==$FLASHINFER_VERSION"
    ensure "flashinfer-jit-cache" "$FLASHINFER_JIT_CACHE" \
        --force-reinstall "flashinfer-jit-cache==$FLASHINFER_JIT_CACHE" \
        --extra-index-url "$FI_JIT_INDEX"
    ensure "flashinfer-cubin" "$FLASHINFER_VERSION" \
        "flashinfer-cubin==$FLASHINFER_VERSION" \
        --extra-index-url "$FI_CUBIN_INDEX"
fi

# ---------------------------- 4. transformers --------------------------------
if [[ "$CHECK_ONLY" -eq 0 ]]; then
    ensure "transformers" "$TRANSFORMERS_VERSION" \
        "transformers==$TRANSFORMERS_VERSION"
fi
# torchaudio/torchcodec 按仓库 pyproject pin 保持不动（torchaudio==2.11.0 与 torch 2.13 共存）

# ------------------------------- 5. 验证 -------------------------------------
log "开始验证..."
if ! E_TORCH="$TORCH_VERSION" E_TV="$TORCHVISION_VERSION" \
E_SGLK="$SGL_KERNEL_EXPECTED" E_FI="$FLASHINFER_VERSION" \
E_JIT="$FLASHINFER_JIT_CACHE" E_CUBIN="$FLASHINFER_VERSION" \
E_TR="$TRANSFORMERS_VERSION" \
"$PY" - <<'PYEOF'
import importlib.metadata as im
import os, sys

ok = True

def row(label, dist, expected, import_name=None):
    global ok
    try:
        got = im.version(dist)
        if import_name:
            __import__(import_name)
        good = (got == expected)
    except Exception as e:
        got, good = f"<ERROR: {type(e).__name__}: {e}>", False
    if not good:
        ok = False
    print(f"  [{'PASS' if good else 'FAIL'}] {label}: {got}" +
          ("" if good else f"  (期望 {expected})"))

print("[1/4] torch 栈")
row("torch",       "torch",       os.environ["E_TORCH"], "torch")
row("torchvision", "torchvision", os.environ["E_TV"],    "torchvision")
import torch
print(f"        CUDA: {torch.version.cuda} | device cap: "
      f"{torch.cuda.get_device_capability() if torch.cuda.is_available() else 'n/a'}")

print("[2/4] sgl-kernel")
row("sglang-kernel", "sglang-kernel", os.environ["E_SGLK"], "sgl_kernel")

print("[3/4] flashinfer 三件套")
row("flashinfer-python",   "flashinfer-python",   os.environ["E_FI"],    "flashinfer")
row("flashinfer-jit-cache","flashinfer-jit-cache",os.environ["E_JIT"])
row("flashinfer-cubin",    "flashinfer-cubin",    os.environ["E_CUBIN"])

print("[4/4] transformers / compressed-tensors / sglang")
row("transformers", "transformers", os.environ["E_TR"], "transformers")
try:
    import compressed_tensors
    print(f"  [PASS] compressed-tensors: {im.version('compressed-tensors')}")
except Exception as e:
    ok = False
    print(f"  [FAIL] compressed-tensors: <ERROR: {e}>")
import sglang
print(f"        sglang {sglang.__version__} @ {sglang.__file__}")
if "/sgl-workspace/sglang/python" not in sglang.__file__:
    print("  [WARN] sglang 非本仓库 editable 安装，KV descale 修复可能不在！")

sys.exit(0 if ok else 1)
PYEOF
then
    die "验证未通过，请检查上方 FAIL 项"
fi

log "全部验证通过 ✓"
log "启动命令（笔记 §2 基线）:"
log "  $PY -m sglang.launch_server --model-path /sgl-workspace/models/Qwen3.8-27B-NVFP4 \\"
log "      --kv-cache-dtype bf16 --mem-fraction-static 0.85 --context-length 32768 --port 30000"
