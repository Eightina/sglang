#!/usr/bin/env bash
# SGLang Chat TUI 启动脚本
#
# 用法:
#   ./chat_tui.sh                                # 连接 http://127.0.0.1:30000
#   ./chat_tui.sh --base-url http://host:port    # 指定服务地址
#   SGLANG_TUI_URL=http://host:port ./chat_tui.sh
#
# 其余参数原样透传给 chat_tui.py (--model/--system 等)。
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if ! python3 -c 'import rich, prompt_toolkit, httpx' 2>/dev/null; then
  echo "[chat-tui] 缺少依赖, 请先安装: pip install rich prompt_toolkit httpx" >&2
  exit 1
fi

# 未显式指定 --base-url 时, 使用环境变量或默认端口 30000
args=("$@")
has_url=0
for a in ${args[@]+"${args[@]}"}; do
  case "$a" in
    --base-url*) has_url=1 ;;
  esac
done
if [ "$has_url" -eq 0 ]; then
  args+=("--base-url" "${SGLANG_TUI_URL:-http://127.0.0.1:30000}")
fi

exec python3 "$DIR/chat_tui.py" "${args[@]}"
