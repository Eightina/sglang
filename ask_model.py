#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""最简单的模型直测脚本：改下面的 TEXT，然后 `python3 ask_model.py`。

- 默认走 /v1/chat/completions（套 chat template），打印模型回复
- RAW=True 改走 sglang 原生 /generate（不套模板），用于区分
  「chat template 问题」和「模型/量化问题」
- 也可以命令行直接传内容：python3 ask_model.py "1+1等于几?"
"""

import json
import sys

import requests

BASE_URL = "http://127.0.0.1:30000"

# ============ 在这里写要说的内容 ============
TEXT = "你好，请用两句话介绍一下你自己。"
# ===========================================

TEMPERATURE = 0.0    # 0 = 贪心解码，排除采样随机性（调胡言乱语时建议保持 0）
MAX_TOKENS = 512
TIMEOUT = 600        # 秒；27B 模型首 token 可能较慢
RAW = True          # True = 走 /generate 原生接口，不套 chat template
SHOW_JSON = False    # True = 额外打印原始响应 JSON（排查时对照用）


def chat(text: str) -> None:
    """OpenAI 兼容 chat 接口（套 chat template）。"""
    r = requests.post(
        f"{BASE_URL}/v1/chat/completions",
        json={
            "messages": [{"role": "user", "content": text}],
            "temperature": TEMPERATURE,
            "max_tokens": MAX_TOKENS,
        },
        timeout=TIMEOUT,
    )
    if r.status_code != 200:
        sys.exit(f"HTTP {r.status_code}: {r.text[:500]}")
    data = r.json()
    choice = data["choices"][0]
    msg = choice.get("message") or {}
    reasoning = msg.get("reasoning_content")
    if reasoning:
        print("---- thinking ----")
        print(reasoning)
        print("---- content ----")
    print(msg.get("content", ""))
    usage = data.get("usage") or {}
    print(f"\n[finish={choice.get('finish_reason')} "
          f"prompt_tokens={usage.get('prompt_tokens')} "
          f"completion_tokens={usage.get('completion_tokens')}]")
    if SHOW_JSON:
        print(json.dumps(data, ensure_ascii=False, indent=2))


def raw_generate(text: str) -> None:
    """sglang 原生 /generate：直接把文本喂给模型，不套 chat template。"""
    r = requests.post(
        f"{BASE_URL}/generate",
        json={
            "text": text,
            "sampling_params": {
                "temperature": TEMPERATURE,
                "max_new_tokens": MAX_TOKENS,
            },
        },
        timeout=TIMEOUT,
    )
    if r.status_code != 200:
        sys.exit(f"HTTP {r.status_code}: {r.text[:500]}")
    data = r.json()
    print(data.get("text", ""))
    meta = data.get("meta_info") or {}
    fr = meta.get("finish_reason")
    fr = fr.get("type") if isinstance(fr, dict) else fr
    print(f"\n[finish={fr} completion_tokens={meta.get('completion_tokens')}]")
    if SHOW_JSON:
        print(json.dumps(data, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    text = sys.argv[1] if len(sys.argv) > 1 else TEXT
    print(f">>> {text}\n")
    raw_generate(text) if RAW else chat(text)
