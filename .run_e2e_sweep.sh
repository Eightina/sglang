#!/bin/bash
# End-to-end ITL across the FP8 baseline and three MXFP4 configs, all on one
# binary. Fixed: 98k context, 96 output tokens, 8 repetitions, streaming p50.
set -u
OUT=/tmp/e2e_results.txt
: > "$OUT"

run_one() {
  local NAME="$1" SCRIPT="$2" PS="$3" OFF="$4" LABEL="$5"
  # Kill the whole engine, not just the launcher: surviving scheduler children
  # keep both the port and the VRAM, which makes the next launch OOM *and*
  # leaves the health endpoint answering for the previous config.
  pkill -f "sglang" 2>/dev/null
  sleep 15
  export SGLANG_DISABLE_MXFP4_PAGED_INDEX="$OFF"
  local PSARG=""
  [ -n "$PS" ] && PSARG="--page-size $PS"
  local LOG="/tmp/e2e_$NAME.log"
  MEM=0.85 MAMBA=8 CTX=200000 LOG="$LOG" \
    EXTRA="--disable-prefill-cuda-graph $PSARG" "$SCRIPT" > /dev/null 2>&1 &
  # Readiness must be proven by THIS run's log. Polling the port instead lets a
  # stale server answer, which once produced an "FP8" row that was really an
  # MXFP4 one while the actual FP8 server had already died of OOM.
  local ready=no
  for i in $(seq 1 60); do
    if grep -q "The server is fired up" "$LOG" 2>/dev/null; then
      ready=yes; break
    fi
    sleep 10
  done
  {
    echo "### $LABEL"
    if [ "$ready" != yes ]; then
      echo "  SERVER DID NOT START -- see $LOG"
      grep -iE "Traceback|out of memory" "$LOG" | head -3
      return
    fi
    grep -oE "#tokens: [0-9]+" "$LOG" | head -1
    timeout 1500 python /tmp/itl.py 30000 98000 96 8 2>&1 | tail -1
  } >> "$OUT"
}

run_one fp8           /tmp/launch_qwen38_fp8.sh   ""  0 "1. FP8 baseline (flashinfer)"
run_one mx_ps1        /tmp/launch_qwen38_mxfp4.sh ""  0 "2. MXFP4 page_size=1 (shipping default)"
run_one mx_ps64_tok   /tmp/launch_qwen38_mxfp4.sh 64  1 "3. MXFP4 page_size=64, token index"
run_one mx_ps64_paged /tmp/launch_qwen38_mxfp4.sh 64  0 "4. MXFP4 page_size=64, paged index"

pkill -f "sglang" 2>/dev/null
echo "=== DONE ===" >> "$OUT"
