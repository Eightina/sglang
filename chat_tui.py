#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SGLang Chat TUI —— 连接 SGLang 推理服务的终端聊天界面。

除基本多轮流式对话外, 内置一组面向 token 级精度测试的快捷命令:

    /det       确定性测试: 同一 prompt 跑两次逐 token 对比 (greedy / sampled 两种模式)
    /speed     流式测速: TTFT / TPOT / tok/s, 附逐 token 延迟火花图
    /entropy   逐 token 置信度热图: 基于返回的 top logprobs 计算所选 token 概率与熵
    /sweep     采样参数扫描: greedy / 当前参数 / 高温 / top_k 并行对比与一致度
    /all       依次运行以上四项
    /lp        普通对话也显示逐 token logprob 候选表
    /raw       切换到 /v1/completions 裸补全模式 (绕过 chat template)

启动:  ./chat_tui.sh                              # 默认 http://127.0.0.1:30000
       ./chat_tui.sh --base-url http://host:port  # 指定服务地址
"""

from __future__ import annotations

import argparse
import asyncio
import difflib
import json
import math
import os
import sys
import time
from dataclasses import asdict, dataclass

import httpx
from prompt_toolkit import PromptSession
from prompt_toolkit.application import Application
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.history import InMemoryHistory
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.keys import Keys
from prompt_toolkit.layout import Layout
from prompt_toolkit.layout.containers import Window
from prompt_toolkit.layout.controls import FormattedTextControl
from prompt_toolkit.patch_stdout import patch_stdout
from rich import box
from rich.console import Console
from rich.markup import escape
from rich.panel import Panel
from rich.table import Table
from rich.text import Text


class TeeFile:
    """rich 输出的分流器: 写入当前 stdout 的同时按行记录到回看缓冲。"""

    MAX_LINES = 20000

    def __init__(self):
        self.lines: list[str] = []
        self._pending = ""

    def write(self, s: str) -> None:
        if not s:
            return
        # 剔除备用屏幕切换序列, 避免回看重放时触发切屏
        parts = (self._pending + s).replace("\x1b[?1049h", "").replace("\x1b[?1049l", "").split("\n")
        self._pending = parts.pop()
        if parts:
            self.lines.extend(parts)
            if len(self.lines) > self.MAX_LINES:
                del self.lines[: len(self.lines) - self.MAX_LINES]
        try:
            sys.stdout.write(s)  # 动态取当前 stdout, 兼容 patch_stdout
        except Exception:
            pass

    def flush(self) -> None:
        try:
            sys.stdout.flush()
        except Exception:
            pass

    def isatty(self) -> bool:
        try:
            return sys.stdout.isatty()
        except Exception:
            return False

    @property
    def encoding(self) -> str:
        return "utf-8"


_TEE = TeeFile()
console = Console(file=_TEE, highlight=False)

SCROLL = "__scroll__"  # 主输入行按 PgUp/Ctrl+O 时 prompt 的返回标记


def _nav_keybindings() -> KeyBindings:
    """回看模式按键表: 返回按键名, 由 pager 解释。"""
    kb = KeyBindings()
    for key in ("up", "down", "pageup", "pagedown", "space", "enter",
                "b", "f", "d", "u", "g", "G", "home", "end",
                "q", "escape", "c-c"):
        @kb.add(key, eager=True)
        def _exit_key(event, _key=key):
            event.app.exit(result=_key)

    @kb.add(Keys.Any)
    def _any_key(event):
        event.app.exit(result="__ignore__")

    return kb


KB_NAV = _nav_keybindings()
# 迷你布局: 仅占最后一行, 用于翻页时读取按键而不干扰画面
_NAV_LAYOUT = Layout(Window(FormattedTextControl(text=""), height=1, char=" "))

DEFAULT_DET_PROMPT = "用三句话解释 Transformer 的 self-attention 机制。"
SPARK = "▁▂▃▄▅▆▇█"


class Quit(Exception):
    """用户请求退出 TUI。"""


def prob_color(p: float) -> str:
    """按置信概率给 token 选颜色。"""
    if p >= 0.9:
        return "green"
    if p >= 0.6:
        return "yellow"
    if p >= 0.3:
        return "dark_orange"
    return "red"


def bar(p: float, width: int = 8) -> str:
    filled = max(0, min(width, round(p * width)))
    return "█" * filled + "░" * (width - filled)


def tok_disp(t: str, width: int = 14) -> str:
    """token 文本用于展示: 换行可见化并截断。"""
    s = (t or "").replace("\n", "⏎").replace("\t", "⇥")
    return s if len(s) <= width else s[: width - 1] + "…"


def shorten(s: str, n: int) -> str:
    s = s.replace("\n", "⏎")
    return s if len(s) <= n else s[: n - 1] + "…"


def tokens_of(entries) -> list[str]:
    return [e.get("token", "") for e in (entries or [])]


def seq_ratio(a, b) -> float:
    if not a and not b:
        return 1.0
    return difflib.SequenceMatcher(None, list(a), list(b)).ratio()


@dataclass
class Params:
    """采样参数 (与 /v1/chat/completions 字段对应)。"""

    temperature: float = 0.7
    top_p: float = 1.0
    top_k: int = -1  # -1 表示关闭
    min_p: float = 0.0  # 0 表示关闭
    repetition_penalty: float = 1.0
    max_tokens: int = 4096
    seed: int | None = None
    top_logprobs: int = 10  # /lp 显示时的候选数

    def payload(self, extra: bool = True) -> dict:
        d = {"temperature": self.temperature, "top_p": self.top_p, "max_tokens": self.max_tokens}
        if extra:  # sglang 扩展参数
            if self.top_k and self.top_k > 0:
                d["top_k"] = self.top_k
            if self.min_p and self.min_p > 0:
                d["min_p"] = self.min_p
            if self.repetition_penalty and self.repetition_penalty != 1.0:
                d["repetition_penalty"] = self.repetition_penalty
        if self.seed is not None:
            d["seed"] = self.seed
        return d


class SglClient:
    """对 SGLang OpenAI 兼容接口的轻量异步封装。"""

    def __init__(self, base_url: str, timeout: float = 600.0):
        self.base = base_url.rstrip("/")
        self._http = httpx.AsyncClient(timeout=timeout)

    async def close(self) -> None:
        await self._http.aclose()

    async def health(self) -> bool:
        try:
            r = await self._http.get(f"{self.base}/health", timeout=5.0)
            return r.status_code == 200
        except Exception:
            return False

    async def get_model(self) -> str | None:
        try:
            r = await self._http.get(f"{self.base}/v1/models", timeout=5.0)
            r.raise_for_status()
            data = r.json().get("data") or []
            return data[0].get("id") if data else None
        except Exception:
            return None

    async def get_model_info(self) -> dict | None:
        try:
            r = await self._http.get(f"{self.base}/get_model_info", timeout=5.0)
            r.raise_for_status()
            return r.json()
        except Exception:
            return None

    async def post_json(self, path: str, payload: dict) -> dict:
        r = await self._http.post(f"{self.base}{path}", json=payload)
        r.raise_for_status()
        return r.json()

    async def stream_sse(self, path: str, payload: dict):
        """逐条 yield SSE data 中的 JSON 对象 (不含 [DONE])。"""
        async with self._http.stream("POST", f"{self.base}{path}", json=payload) as r:
            r.raise_for_status()
            async for line in r.aiter_lines():
                if not line.startswith("data:"):
                    continue
                data = line[len("data:"):].strip()
                if not data:
                    continue
                if data == "[DONE]":
                    return
                yield json.loads(data)


class App:
    def __init__(self, base_url: str, model: str | None = None, system: str | None = None):
        self.client = SglClient(base_url)
        self.base_url = base_url
        self.model = model
        self.system = system
        self.params = Params()
        self.messages: list[dict] = []  # {"role", "content"}
        self.show_lp = False  # 普通对话是否显示逐 token logprob
        self.raw_mode = False  # 是否使用 /v1/completions 裸补全

        self.commands = {
            "help": self.cmd_help, "h": self.cmd_help,
            "q": self.cmd_quit, "quit": self.cmd_quit, "exit": self.cmd_quit,
            "clear": self.cmd_clear,
            "system": self.cmd_system,
            "params": self.cmd_params,
            "set": self.cmd_set,
            "model": self.cmd_model,
            "raw": self.cmd_raw,
            "lp": self.cmd_lp,
            "topk": self.cmd_topk,
            "det": self.cmd_det,
            "speed": self.cmd_speed, "spd": self.cmd_speed,
            "entropy": self.cmd_entropy, "ent": self.cmd_entropy,
            "sweep": self.cmd_sweep, "sw": self.cmd_sweep,
            "all": self.cmd_all,
            "save": self.cmd_save,
            "less": self.cmd_less, "pg": self.cmd_less, "page": self.cmd_less,
        }

    # ------------------------------------------------------------------ 输出

    def out(self, text: str, style: str | None = None) -> None:
        """流式纯文本输出 (不做 markup 解析, 靠终端自行换行)。"""
        console.print(text, end="", markup=False, highlight=False, soft_wrap=True, style=style)
        console.file.flush()

    def rule(self, title: str, style: str = "magenta") -> None:
        console.rule(f"[bold {style}]{escape(title)}", style="dim")

    # ------------------------------------------------------------------ 请求

    def _req_messages(self, prompt: str | None = None) -> list[dict]:
        msgs = []
        if self.system:
            msgs.append({"role": "system", "content": self.system})
        if prompt is not None:
            msgs.append({"role": "user", "content": prompt})
        else:
            msgs.extend(self.messages)
        return msgs

    def _ladder(self, logprobs_k: int) -> list[tuple[bool, int]]:
        """请求降级阶梯: 400 时先去 sglang 扩展参数, 再去 logprobs。"""
        if logprobs_k > 0:
            return [(True, logprobs_k), (False, logprobs_k), (False, 0)]
        return [(True, 0), (False, 0)]

    def _chat_payload(self, messages, stream: bool, lp: int, extra: bool, override: dict | None) -> dict:
        payload = {"model": self.model or "default", "messages": messages, "stream": stream}
        payload.update(self.params.payload(extra=extra))
        if stream:
            payload["stream_options"] = {"include_usage": True}
        if lp:
            payload["logprobs"] = True
            payload["top_logprobs"] = lp
        for k, v in (override or {}).items():
            if v is None:
                payload.pop(k, None)
            else:
                payload[k] = v
        return payload

    def _raw_payload(self, prompt: str, stream: bool, lp: int, extra: bool, override: dict | None) -> dict:
        payload = {"model": self.model or "default", "prompt": prompt, "stream": stream}
        payload.update(self.params.payload(extra=extra))
        if stream:
            payload["stream_options"] = {"include_usage": True}
        if lp:
            payload["logprobs"] = True
            payload["top_logprobs"] = lp
        for k, v in (override or {}).items():
            if v is None:
                payload.pop(k, None)
            else:
                payload[k] = v
        return payload

    def _note_degrade(self, e: httpx.HTTPStatusError, extra: bool, lp: int) -> None:
        what = []
        if extra:
            what.append("扩展采样参数")
        if lp:
            what.append("logprobs")
        console.print(
            f"[dim]· 服务器拒绝了含 {'/'.join(what)} 的请求 ({e.response.status_code}), 降级重试[/dim]"
        )

    async def once_chat(self, messages, logprobs_k: int = 0, override: dict | None = None) -> dict:
        """非流式 chat 请求, 带参数降级。"""
        last: Exception | None = None
        for extra, lp in self._ladder(logprobs_k):
            payload = self._chat_payload(messages, False, lp, extra, override)
            try:
                return await self.client.post_json("/v1/chat/completions", payload)
            except httpx.HTTPStatusError as e:
                last = e
                if e.response is not None and e.response.status_code == 400 and (extra or lp):
                    self._note_degrade(e, extra, lp)
                    continue
                raise
        raise last  # type: ignore[misc]

    async def stream_chat(self, messages, logprobs_k: int = 0, override: dict | None = None):
        """流式 chat 请求, 带参数降级 (在收到首块时探测 400)。"""
        for extra, lp in self._ladder(logprobs_k):
            payload = self._chat_payload(messages, True, lp, extra, override)
            agen = self.client.stream_sse("/v1/chat/completions", payload)
            try:
                first = await agen.__anext__()
            except httpx.HTTPStatusError as e:
                await agen.aclose()
                if e.response is not None and e.response.status_code == 400 and (extra or lp):
                    self._note_degrade(e, extra, lp)
                    continue
                raise
            except StopAsyncIteration:
                return
            yield first
            try:
                async for chunk in agen:
                    yield chunk
            finally:
                await agen.aclose()
            return

    async def stream_raw(self, prompt: str, logprobs_k: int = 0, override: dict | None = None):
        """流式 /v1/completions 裸补全。"""
        for extra, lp in self._ladder(logprobs_k):
            payload = self._raw_payload(prompt, True, lp, extra, override)
            agen = self.client.stream_sse("/v1/completions", payload)
            try:
                first = await agen.__anext__()
            except httpx.HTTPStatusError as e:
                await agen.aclose()
                if e.response is not None and e.response.status_code == 400 and (extra or lp):
                    self._note_degrade(e, extra, lp)
                    continue
                raise
            except StopAsyncIteration:
                return
            yield first
            try:
                async for chunk in agen:
                    yield chunk
            finally:
                await agen.aclose()
            return

    # ------------------------------------------------------------- 回复渲染

    async def stream_reply(self, *, messages, label: str = "assistant", style: str = "bold green",
                           logprobs_k: int = 0, override: dict | None = None) -> dict:
        """流式输出一条回复。返回 {text, entries, finish, usage, elapsed, ttft, aborted}。"""
        start = time.perf_counter()
        ttft: float | None = None
        parts: list[str] = []
        entries: list[dict] = []
        finish = None
        usage = None
        aborted = False

        console.print(Text(f"{label} ❯ ", style=style), end="", soft_wrap=True)
        console.file.flush()
        agen = self.stream_chat(messages, logprobs_k=logprobs_k, override=override)
        try:
            while True:
                try:
                    ev = await agen.__anext__()
                except StopAsyncIteration:
                    break
                choices = ev.get("choices") or []
                if choices:
                    c = choices[0]
                    delta = c.get("delta") or {}
                    reasoning = delta.get("reasoning_content")
                    if reasoning:
                        if ttft is None:
                            ttft = time.perf_counter() - start
                        self.out(reasoning, style="dim")
                    piece = delta.get("content") or ""
                    if piece:
                        if ttft is None:
                            ttft = time.perf_counter() - start
                        parts.append(piece)
                        self.out(piece)
                    lp = (c.get("logprobs") or {}).get("content")
                    if lp:
                        entries.extend(lp)
                    if c.get("finish_reason"):
                        finish = c["finish_reason"]
                if ev.get("usage"):
                    usage = ev["usage"]
        except (KeyboardInterrupt, asyncio.CancelledError):
            aborted = True
            self.out("  [已中断]", style="bold red")
        finally:
            await agen.aclose()

        console.print()  # 收尾换行
        elapsed = time.perf_counter() - start
        self.print_meta(len(entries) or None, finish, elapsed, ttft, usage)
        if self.show_lp and logprobs_k > 0 and entries:
            self.render_lp_table(entries)
        return {
            "text": "".join(parts), "entries": entries, "finish": finish,
            "usage": usage, "elapsed": elapsed, "ttft": ttft, "aborted": aborted,
        }

    async def stream_reply_raw(self, prompt: str) -> dict:
        start = time.perf_counter()
        ttft: float | None = None
        parts: list[str] = []
        n_lp_tokens = 0
        finish = None
        usage = None
        aborted = False

        console.print(Text("raw ❯ ", style="bold magenta"), end="", soft_wrap=True)
        console.file.flush()
        agen = self.stream_raw(prompt)
        try:
            while True:
                try:
                    ev = await agen.__anext__()
                except StopAsyncIteration:
                    break
                choices = ev.get("choices") or []
                if choices:
                    c = choices[0]
                    piece = c.get("text") or ""
                    if piece:
                        if ttft is None:
                            ttft = time.perf_counter() - start
                        parts.append(piece)
                        self.out(piece)
                    lps = c.get("logprobs") or {}
                    n_lp_tokens += len(lps.get("tokens") or [])
                    if c.get("finish_reason"):
                        finish = c["finish_reason"]
                if ev.get("usage"):
                    usage = ev["usage"]
        except (KeyboardInterrupt, asyncio.CancelledError):
            aborted = True
            self.out("  [已中断]", style="bold red")
        finally:
            await agen.aclose()

        console.print()
        elapsed = time.perf_counter() - start
        self.print_meta(n_lp_tokens or None, finish, elapsed, ttft, usage)
        return {"text": "".join(parts), "finish": finish, "usage": usage,
                "elapsed": elapsed, "ttft": ttft, "aborted": aborted}

    def print_meta(self, n_tokens: int | None, finish, elapsed: float,
                   ttft: float | None, usage: dict | None) -> None:
        usage = usage or {}
        n_tok = n_tokens or usage.get("completion_tokens") or 0
        bits = [f"{n_tok} tok"] if n_tok else []
        if ttft is not None:
            bits.append(f"ttft {ttft * 1000:.0f}ms")
        bits.append(f"{elapsed:.2f}s")
        if elapsed > 0 and n_tok:
            bits.append(f"{n_tok / elapsed:.1f} tok/s")
        bits.append(f"finish {finish or '?'}")
        ptd = usage.get("prompt_tokens_details") or {}
        if ptd.get("cached_tokens"):
            bits.append(f"cached {ptd['cached_tokens']}")
        console.print(Text("   · " + " · ".join(bits), style="dim"))

    def render_lp_table(self, entries: list[dict], cap: int = 24) -> None:
        """逐 token 的 top-k 候选表 (紧凑视图)。"""
        if not entries:
            console.print(Text("   (服务器未返回 logprobs)", style="dim"))
            return
        table = Table(box=box.SIMPLE, header_style="dim", show_edge=False, pad_edge=False)
        table.add_column("#", justify="right", style="dim")
        table.add_column("token")
        table.add_column("logprob", justify="right")
        table.add_column("prob", justify="right")
        table.add_column("候选 (top)")
        for i, e in enumerate(entries[:cap]):
            lp = e.get("logprob", 0.0)
            p = math.exp(lp) if lp > -20 else 0.0
            cand = Text()
            tops = e.get("top_logprobs") or []
            for j, alt in enumerate(tops[:3]):
                if j:
                    cand.append("  ")
                ap = math.exp(alt.get("logprob", -20))
                cand.append(tok_disp(alt.get("token", "?"), 10), style=prob_color(ap))
                cand.append(f" {ap * 100:4.1f}% {bar(ap, 6)}", style="dim")
            t = Text(tok_disp(e.get("token", "?")), style=prob_color(p))
            table.add_row(str(i), t, f"{lp:7.3f}", f"{p * 100:5.1f}%", cand)
        console.print(table)
        if len(entries) > cap:
            console.print(Text(f"   … 共 {len(entries)} 个 token, 仅显示前 {cap} 个", style="dim"))

    # ------------------------------------------------------------- 对话主流程

    def _last_user(self) -> str | None:
        for m in reversed(self.messages):
            if m.get("role") == "user":
                return m["content"]
        return None

    def _prompt_from_args(self, joined: str, default: str = DEFAULT_DET_PROMPT) -> str:
        return joined.strip() or self._last_user() or default

    async def do_chat(self, text: str) -> None:
        console.print(Text("you ❯ ", style="bold cyan"), end="", soft_wrap=True)
        console.print(Text(text), soft_wrap=True)
        self.messages.append({"role": "user", "content": text})
        try:
            if self.raw_mode:
                r = await self.stream_reply_raw(text)
            else:
                lp = self.params.top_logprobs if self.show_lp else 0
                r = await self.stream_reply(messages=self._req_messages(), logprobs_k=lp)
        except httpx.ConnectError:
            console.print(f"[red]无法连接 {self.base_url}[/red] — 请确认 sglang 服务已启动且端口正确")
            self.messages.pop()
            return
        if r.get("text"):
            self.messages.append({"role": "assistant", "content": r["text"]})
        elif r.get("aborted") and self.messages and self.messages[-1]["role"] == "user":
            self.messages.pop()  # 没有产出任何内容则回退用户消息

    # ------------------------------------------------------------- 派发

    async def dispatch(self, line: str) -> None:
        stripped = line.strip()
        if not stripped:
            return
        if stripped.startswith("//"):  # 转义: 想真正发送以 / 开头的内容
            await self.do_chat(stripped[1:])
            return
        if stripped.startswith("/"):
            parts = stripped.split(maxsplit=1)
            name = parts[0][1:].lower()
            rest = parts[1] if len(parts) > 1 else ""
            handler = self.commands.get(name)
            if handler is None:
                console.print(f"[dim]未知命令 /{escape(name)} — 输入 /help 查看全部[/dim]")
                return
            await handler(rest.split(), rest)
            return
        await self.do_chat(stripped)

    # ------------------------------------------------------------- 基础命令

    async def cmd_quit(self, args, raw) -> None:
        raise Quit()

    async def cmd_help(self, args, raw) -> None:
        rows = [
            ("直接输入 / //x", "流式多轮对话; //x 发送以 / 开头的原文"),
            ("/clear", "清空会话历史 (保留 system prompt)"),
            ("/system [文本]", "查看或设置 system prompt; `/system clear` 清除"),
            ("/params", "查看当前采样参数与模式"),
            ("/set <键> <值>", "改参数: temperature top_p top_k min_p max_tokens seed repetition_penalty top_logprobs"),
            ("/model", "查看服务端模型信息"),
            ("/save [文件]", "保存会话记录为 JSON"),
            ("/less  (PgUp / Ctrl+O)", "翻页回看全部输出: 空格/b/f 翻页, d/u 半页, g/G 首/尾, q 退出"),
            ("/q", "退出 (Ctrl+D 同效)"),
        ]
        prec = [
            ("/det [greedy|sampled] [prompt]", "确定性: 同一 prompt 跑两次逐 token 对比; sampled 验证 seed 复现"),
            ("/speed [n] [prompt]", "流式测速: TTFT / TPOT / tok/s + 逐 token 延迟火花图 (n=生成 token 上限)"),
            ("/entropy [prompt]", "逐 token 置信度热图与高熵位置 (top-20 logprobs, greedy)"),
            ("/sweep [prompt]", "采样参数扫描: greedy/当前/高温/top_k 并行对比 + 一致度"),
            ("/all [prompt]", "依次运行 det → speed → entropy → sweep"),
            ("/lp", "切换: 普通对话也显示逐 token logprob 候选表"),
            ("/topk <n>", "设置 logprob 候选数并开启显示"),
            ("/raw", "切换 /v1/completions 裸补全 (绕过 chat template, 仅发最后一条消息)"),
        ]
        t1 = Table(box=box.SIMPLE, title="基础对话", title_style="bold cyan")
        t1.add_column("命令", style="cyan"); t1.add_column("说明")
        for a, b in rows:
            t1.add_row(a, b)
        t2 = Table(box=box.SIMPLE, title="token 级精度测试", title_style="bold magenta")
        t2.add_column("命令", style="magenta"); t2.add_column("说明")
        for a, b in prec:
            t2.add_row(a, b)
        console.print(t1)
        console.print(t2)

    async def cmd_clear(self, args, raw) -> None:
        self.messages.clear()
        console.print("[dim]已清空会话历史[/dim]")

    async def cmd_system(self, args, raw) -> None:
        if not args:
            cur = self.system if self.system else "(未设置)"
            console.print(Text(f"system: {shorten(cur, 200)}", style="dim"))
            return
        text = raw.strip()
        if text.lower() in ("clear", "off", "none"):
            self.system = None
            console.print("[dim]system prompt 已清除[/dim]")
            return
        self.system = text
        console.print(Text(f"system 已设置: {shorten(text, 120)}", style="dim"))

    async def cmd_params(self, args, raw) -> None:
        p = self.params
        t = Table(box=box.SIMPLE, show_header=False)
        t.add_column(style="cyan"); t.add_column()
        t.add_row("服务", f"{self.base_url}   模型: {self.model or '?'}")
        t.add_row("temperature / top_p", f"{p.temperature} / {p.top_p}")
        t.add_row("top_k / min_p", f"{p.top_k} / {p.min_p}")
        t.add_row("repetition_penalty", str(p.repetition_penalty))
        t.add_row("max_tokens", str(p.max_tokens))
        t.add_row("seed", str(p.seed))
        t.add_row("top_logprobs (候选数)", str(p.top_logprobs))
        t.add_row("模式", f"lp显示={'开' if self.show_lp else '关'}   raw={'开' if self.raw_mode else '关'}")
        console.print(t)

    async def cmd_set(self, args, raw) -> None:
        if not args:
            await self.cmd_params([], "")
            return
        key = args[0].lower()
        val = " ".join(args[1:])
        p = self.params
        try:
            if key == "seed":
                p.seed = None if val.lower() in ("", "none", "off") else int(val)
            elif key in ("temperature", "top_p", "min_p", "repetition_penalty"):
                setattr(p, key, float(val))
            elif key in ("top_k", "max_tokens", "top_logprobs"):
                setattr(p, key, int(val))
            else:
                console.print(f"[red]未知参数 {escape(key)}[/red] — 可选: temperature top_p top_k min_p "
                              "max_tokens seed repetition_penalty top_logprobs")
                return
        except ValueError:
            console.print(f"[red]无效值: {escape(val)}[/red]")
            return
        console.print(Text(f"{key} = {getattr(p, key)}", style="dim"))

    async def cmd_model(self, args, raw) -> None:
        m = await self.client.get_model()
        if m:
            self.model = m
        t = Table(box=box.SIMPLE, show_header=False)
        t.add_column(style="cyan"); t.add_column()
        t.add_row("openai model", m or "(查询失败)")
        info = await self.client.get_model_info()
        for k in sorted(info or {}):
            t.add_row(f"get_model_info.{k}", Text(shorten(str(info[k]), 100)))
        console.print(t)

    async def cmd_raw(self, args, raw) -> None:
        self.raw_mode = not self.raw_mode
        state = "[magenta]开[/magenta]" if self.raw_mode else "关"
        console.print(f"raw 模式: {state} — 使用 /v1/completions, 不套 chat template, 仅发送最后一条用户消息")

    async def cmd_lp(self, args, raw) -> None:
        self.show_lp = not self.show_lp
        console.print(f"逐 token logprob 显示: {'[magenta]开[/magenta]' if self.show_lp else '关'} "
                      f"(候选数 {self.params.top_logprobs}, /topk n 修改)")

    async def cmd_topk(self, args, raw) -> None:
        if args and args[0].isdigit():
            self.params.top_logprobs = int(args[0])
            self.show_lp = True
            console.print(f"[dim]top_logprobs = {self.params.top_logprobs}, 显示已开启[/dim]")
        else:
            console.print("用法: /topk <n>   (n=每个 token 展示的候选数)")

    async def cmd_save(self, args, raw) -> None:
        path = args[0] if args else f"chat_tui_transcript_{time.strftime('%Y%m%d_%H%M%S')}.json"
        data = {
            "base_url": self.base_url, "model": self.model, "params": asdict(self.params),
            "system": self.system, "messages": self.messages,
        }
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            console.print(f"[dim]会话已保存: {escape(path)}[/dim]")
        except OSError as e:
            console.print(f"[red]保存失败:[/red] {escape(str(e))}")

    # ------------------------------------------------- token 级精度测试命令

    def _pick_runs(self, results) -> tuple[list | None, list | None]:
        seqs = []
        for r in results:
            if isinstance(r, Exception):
                seqs.append(None)
            else:
                try:
                    seqs.append(tokens_of(r["choices"][0]["logprobs"]["content"]) or None)
                except (KeyError, IndexError, TypeError):
                    seqs.append(None)
        return seqs[0], seqs[1]

    def _diff_view(self, a: list[str], b: list[str], i: int, ctx: int = 6, tail: int = 10) -> Text:
        lines = []
        for label, seq, style in (("run1", a, "green"), ("run2", b, "red")):
            t = Text(f"{label} ▏", style="dim")
            for tok in seq[max(0, i - ctx): i]:
                t.append(tok)
            t.append(" ⟂ ", style="bold magenta")
            for tok in seq[i: i + tail]:
                t.append(tok, style=style)
            lines.append(t)
        out = Text("\n")
        out.append_text(lines[0])
        out.append("\n")
        out.append_text(lines[1])
        return out

    async def cmd_det(self, args, raw) -> None:
        """同一 prompt 两次采样, 逐 token 对比确定性。"""
        mode = "greedy"
        rest = list(args)
        if rest and rest[0].lower() in ("greedy", "g", "sampled", "s"):
            mode = "sampled" if rest[0].lower().startswith("s") else "greedy"
            rest = rest[1:]
        prompt = " ".join(rest) or self._last_user() or DEFAULT_DET_PROMPT
        seed = self.params.seed if self.params.seed is not None else 1234
        cap = min(self.params.max_tokens, 512)
        if mode == "greedy":
            override = {"temperature": 0.0, "seed": seed, "max_tokens": cap}
            desc = "greedy (temperature=0)"
        else:
            override = {"temperature": self.params.temperature or 0.7,
                        "seed": seed, "max_tokens": cap}
            desc = f"sampled (t={override['temperature']}, seed={seed})"
        self.rule(f"确定性测试 · {desc} · 2 runs · “{shorten(prompt, 40)}”", "magenta")
        msgs = self._req_messages(prompt)
        try:
            r1, r2 = await asyncio.gather(self.once_chat(msgs, 1, override),
                                          self.once_chat(msgs, 1, override))
        except httpx.HTTPError as e:
            console.print(f"[red]请求失败:[/red] {escape(str(e)[:200])}")
            return
        s1, s2 = self._pick_runs([r1, r2])
        t1 = (r1["choices"][0].get("message") or {}).get("content", "")
        t2 = (r2["choices"][0].get("message") or {}).get("content", "")
        console.print(Text(f"run1 {shorten(t1, 66)}", style="dim"))
        console.print(Text(f"run2 {shorten(t2, 66)}", style="dim"))
        if s1 and s2:
            if s1 == s2:
                console.print(f"[green]完全一致 ✓ 两次运行 token 序列逐位相同 ({len(s1)} tok) — 确定性 OK[/green]")
            else:
                i = next((i for i, (x, y) in enumerate(zip(s1, s2)) if x != y), min(len(s1), len(s2)))
                ratio = seq_ratio(s1, s2)
                console.print(f"[yellow]出现分歧: 首个不同 token @ {i} · token 相似度 {ratio * 100:.1f}%[/yellow]")
                console.print(self._diff_view(s1, s2, i), soft_wrap=True)
        else:
            console.print("[dim](服务器未返回 logprobs, 退化为文本对比)[/dim]")
            ratio = difflib.SequenceMatcher(None, t1, t2).ratio()
            if t1 == t2:
                console.print(f"[green]完全一致 ✓ 文本相同 ({len(t1)} chars)[/green]")
            else:
                console.print(f"[yellow]文本不同: 相似度 {ratio * 100:.1f}%[/yellow]")

    async def cmd_speed(self, args, raw) -> None:
        """流式测速: TTFT / TPOT / tok/s + 逐 token 延迟火花图。"""
        mtoks, rest = 128, list(args)
        if rest and rest[0].isdigit():
            mtoks = max(1, min(int(rest[0]), 2048))
            rest = rest[1:]
        prompt = " ".join(rest) or self._last_user() or DEFAULT_DET_PROMPT
        self.rule(f"流式测速 · max_tokens={mtoks} · “{shorten(prompt, 40)}”", "magenta")
        start = time.perf_counter()
        ttft = None
        parts: list[str] = []
        chunk_times: list[float] = []
        chunk_tok_counts: list[int] = []
        entries: list[dict] = []
        finish = None
        usage = None
        agen = self.stream_chat(self._req_messages(prompt), logprobs_k=1,
                                override={"max_tokens": mtoks})
        try:
            while True:
                try:
                    ev = await agen.__anext__()
                except StopAsyncIteration:
                    break
                choices = ev.get("choices") or []
                if choices:
                    c = choices[0]
                    piece = (c.get("delta") or {}).get("content") or ""
                    lp = (c.get("logprobs") or {}).get("content") or []
                    if piece:
                        now = time.perf_counter()
                        if ttft is None:
                            ttft = now - start
                        chunk_times.append(now)
                        chunk_tok_counts.append(max(1, len(lp)))
                        parts.append(piece)
                        self.out(piece)
                    if lp:
                        entries.extend(lp)
                    if c.get("finish_reason"):
                        finish = c["finish_reason"]
                if ev.get("usage"):
                    usage = ev["usage"]
        except (KeyboardInterrupt, asyncio.CancelledError):
            self.out("  [已中断]", style="bold red")
        finally:
            await agen.aclose()
        console.print()
        elapsed = time.perf_counter() - start

        # 把每个 chunk 的时间戳展开到 token 粒度: 优先用该 chunk 携带的 logprob 条目数,
        # 无 logprobs 时按 1 chunk ≈ 1 token 估算
        token_times: list[float] = []
        for t, m in zip(chunk_times, chunk_tok_counts):
            token_times.extend([t] * max(1, m))
        if not token_times:
            token_times = chunk_times or [start]

        n = len(token_times)
        lat = [token_times[i] - token_times[i - 1] for i in range(1, n)]
        bits = []
        if ttft is not None:
            bits.append(f"TTFT {ttft * 1000:.0f}ms")
        bits.append(f"总耗时 {elapsed:.2f}s")
        bits.append(f"{n} tok")
        if elapsed > 0:
            bits.append(f"{n / elapsed:.1f} tok/s")
        if lat:
            lat_ms = sorted(x * 1000 for x in lat)
            tpot = (token_times[-1] - token_times[0]) / max(1, n - 1) * 1000
            p50 = lat_ms[len(lat_ms) // 2]
            p90 = lat_ms[int(len(lat_ms) * 0.9)]
            bits.append(f"TPOT {tpot:.0f}ms (p50 {p50:.0f} / p90 {p90:.0f} / max {lat_ms[-1]:.0f}ms)")
        if usage:
            ptd = usage.get("prompt_tokens_details") or {}
            if ptd.get("cached_tokens"):
                bits.append(f"cached {ptd['cached_tokens']}")
        console.print(Text("   " + " · ".join(bits), style="dim"))
        if lat:
            k = min(len(lat), 64)
            group = [sum(lat[i * len(lat) // k: (i + 1) * len(lat) // k]) / max(1, len(lat) // k)
                     for i in range(k)]
            lo, hi = min(group), max(group)
            line = "".join(SPARK[0 if hi == lo else int((v - lo) / (hi - lo) * (len(SPARK) - 1))]
                           for v in group)
            console.print(Text(f"   逐 token 延迟  {line}  ({lo * 1000:.0f}~{hi * 1000:.0f}ms)",
                               style="dim"))
        if finish:
            console.print(Text(f"   finish={finish}", style="dim"))

    def _entropy_of(self, e: dict) -> float | None:
        tops = e.get("top_logprobs") or []
        if not tops:
            return None
        ps = [math.exp(t.get("logprob", -20)) for t in tops]
        z = sum(ps)
        if z <= 0:
            return None
        return -sum((p / z) * math.log(p / z) for p in ps if p > 0)

    async def cmd_entropy(self, args, raw) -> None:
        """greedy 回复的逐 token 置信度热图。"""
        prompt = " ".join(args) or self._last_user() or DEFAULT_DET_PROMPT
        self.rule(f"逐 token 置信度 · greedy · top-20 · “{shorten(prompt, 40)}”", "magenta")
        try:
            resp = await self.once_chat(self._req_messages(prompt), 20,
                                        {"temperature": 0.0,
                                         "max_tokens": min(self.params.max_tokens, 256)})
        except httpx.HTTPError as e:
            console.print(f"[red]请求失败:[/red] {escape(str(e)[:200])}")
            return
        entries = ((resp["choices"][0].get("logprobs") or {}).get("content")) or []
        if not entries:
            console.print("[dim]服务器未返回 logprobs, 无法分析[/dim]")
            return
        hm = Text()
        for e in entries:
            p = math.exp(e.get("logprob", -20))
            hm.append(tok_disp(e.get("token", "?"), 20), style=prob_color(p))
        console.print(hm, soft_wrap=True)
        probs = [math.exp(e.get("logprob", -20)) for e in entries]
        ents = [self._entropy_of(e) for e in entries]
        ent_vals = [x for x in ents if x is not None]
        mean_p = sum(probs) / len(probs)
        stats = Text("   ")
        stats.append(f"{len(entries)} tok · 平均置信 {mean_p * 100:.1f}% · 最低 {min(probs) * 100:.1f}%")
        if ent_vals:
            stats.append(f" · 平均熵(截断) {sum(ent_vals) / len(ent_vals):.2f} nats")
        stats.append(f" · p<0.5 的 token: {sum(1 for x in probs if x < 0.5)}")
        console.print(stats, style="dim")
        console.print(Text("   图例: ", style="dim"), end="")
        for thr, name in ((0.9, "≥90%"), (0.6, "60~90%"), (0.3, "30~60%"), (0.0, "<30%")):
            console.print(Text(f"■{name} ", style=prob_color(thr)), end="")
        console.print()
        order = sorted(range(len(entries)), key=lambda i: probs[i])[:5]
        t = Table(box=box.SIMPLE, title="置信度最低的 5 个位置", title_style="bold")
        t.add_column("#", justify="right", style="dim")
        t.add_column("token")
        t.add_column("p", justify="right")
        t.add_column("主要候选")
        for i in order:
            e = entries[i]
            cand = Text()
            for j, alt in enumerate((e.get("top_logprobs") or [])[:4]):
                if j:
                    cand.append("  ")
                ap = math.exp(alt.get("logprob", -20))
                cand.append(tok_disp(alt.get("token", "?"), 10), style=prob_color(ap))
                cand.append(f" {bar(ap, 6)}", style="dim")
            tt = Text(tok_disp(e.get("token", "?")), style=prob_color(probs[i]))
            t.add_row(str(i), tt, f"{probs[i] * 100:5.1f}%", cand)
        console.print(t)

    async def cmd_sweep(self, args, raw) -> None:
        """并行采样参数扫描, 与 greedy 逐 token 对比一致度。"""
        prompt = " ".join(args) or self._last_user() or DEFAULT_DET_PROMPT
        seed = self.params.seed if self.params.seed is not None else 42
        base = {"seed": seed, "max_tokens": min(self.params.max_tokens, 256)}
        configs = [
            ("greedy t=0", {"temperature": 0.0, **base}),
            (f"当前 t={self.params.temperature} top_p={self.params.top_p}", base),
            ("t=1.0 top_p=1.0", {"temperature": 1.0, "top_p": 1.0, **base}),
            ("t=0.7 top_k=5", {"temperature": 0.7, "top_k": 5, **base}),
        ]
        self.rule(f"采样参数扫描 · seed={seed} · “{shorten(prompt, 40)}”", "magenta")
        msgs = self._req_messages(prompt)
        try:
            results = await asyncio.gather(
                *(self.once_chat(msgs, 1, ov) for _, ov in configs), return_exceptions=True)
        except httpx.HTTPError as e:
            console.print(f"[red]请求失败:[/red] {escape(str(e)[:200])}")
            return
        t = Table(box=box.SIMPLE, title="并行对比", title_style="bold")
        t.add_column("配置", style="cyan")
        t.add_column("tok", justify="right")
        t.add_column("finish")
        t.add_column("输出预览")
        seqs = []
        for (name, _), r in zip(configs, results):
            if isinstance(r, Exception):
                t.add_row(name, "-", "-", Text(f"错误: {shorten(str(r), 60)}", style="red"))
                seqs.append(None)
                continue
            try:
                entry = r["choices"][0]
                seqs.append(tokens_of(entry.get("logprobs", {}).get("content")) or None)
                text = (entry.get("message") or {}).get("content", "")
                n_tok = len(seqs[-1] or [])
                t.add_row(name, str(n_tok), str(entry.get("finish_reason")),
                          Text(shorten(text, 72)))
            except (KeyError, IndexError, TypeError) as e:
                t.add_row(name, "-", "-", Text(f"解析失败: {e}", style="red"))
                seqs.append(None)
        console.print(t)
        gseq = seqs[0]
        if gseq:
            gtext = (results[0]["choices"][0].get("message") or {}).get("content", "")
            for (name, _), r, s in list(zip(configs, results, seqs))[1:]:
                if isinstance(r, Exception):
                    continue
                if s:
                    i = next((i for i, (x, y) in enumerate(zip(gseq, s)) if x != y),
                             min(len(gseq), len(s)))
                    pos = "无分歧" if gseq == s else f"首个分歧 @ {i}"
                    console.print(Text(
                        f"   vs [{name}] 一致度 {seq_ratio(gseq, s) * 100:5.1f}% · {pos}",
                        style="dim"))
                else:
                    text = (r["choices"][0].get("message") or {}).get("content", "")
                    console.print(Text(
                        f"   vs [{name}] 文本一致度 {difflib.SequenceMatcher(None, gtext, text).ratio() * 100:5.1f}%",
                        style="dim"))
        else:
            console.print("[dim](greedy 未返回 logprobs, 无法做 token 级对比)[/dim]")

    async def cmd_all(self, args, raw) -> None:
        joined = " ".join(args)
        prompt = joined or self._last_user() or DEFAULT_DET_PROMPT
        await self.cmd_det(["greedy", prompt], "")
        console.print()
        await self.cmd_speed(["64", prompt], "")
        console.print()
        await self.cmd_entropy([prompt], "")
        console.print()
        await self.cmd_sweep([prompt], "")

    # ------------------------------------------------------------- 回看翻页

    async def cmd_less(self, args, raw) -> None:
        await self.pager()

    async def _read_nav_key(self) -> str:
        """回看模式下读取单个按键 (迷你 Application, 只占最后一行)。"""
        app = Application(layout=_NAV_LAYOUT, key_bindings=KB_NAV)
        try:
            return await app.run_async()
        except KeyboardInterrupt:
            return "q"
        except Exception:
            return "q"

    def _draw_page(self, lines, pos: int, view: int) -> None:
        """在备用屏幕重绘一页: 内容 + 底部状态行, 光标留在最后一行。"""
        h = max(6, console.size.height)
        w = max(40, console.size.width)
        status = (f" 回看 {pos + 1}~{min(pos + view, len(lines))} / {len(lines)} 行  ·  "
                  "↑↓/enter 行 · PgUp/PgDn/b/f/space 页 · d/u 半页 · g/G 首/尾 · q 退出")
        out = ("\x1b[H\x1b[2J" + "\n".join(lines[pos:pos + view])
               + f"\x1b[{h - 1};1H\x1b[2K" + status[:w - 1]
               + f"\x1b[{h};1H")
        try:
            sys.stdout.write(out)  # 直写 stdout, 不经过 rich/回看缓冲
            sys.stdout.flush()
        except Exception:
            pass

    async def pager(self) -> None:
        """在备用屏幕里分页浏览本次会话的全部输出 (退出后原界面不变)。"""
        if not _TEE.lines:
            console.print("[dim](暂无内容可回看)[/dim]")
            return
        console.set_alt_screen()
        try:
            pos = None
            while True:
                view = max(3, max(6, console.size.height) - 2)
                total = len(_TEE.lines)
                if pos is None:
                    pos = max(0, total - view)
                pos = max(0, min(pos, max(0, total - view)))
                self._draw_page(_TEE.lines, pos, view)
                key = await self._read_nav_key()
                if key in ("q", "escape", "c-c"):
                    break
                step = {"up": -1, "down": 1, "enter": 1,
                        "pageup": -view, "pagedown": view,
                        "b": -view, "f": view,
                        "d": view // 2, "u": -(view // 2),
                        "space": view,
                        "home": "top", "g": "top",
                        "end": "bottom", "G": "bottom"}.get(key)
                if step == "top":
                    pos = 0
                elif step == "bottom":
                    pos = max(0, total - view)
                elif step is not None:
                    pos += step
        except KeyboardInterrupt:
            pass
        finally:
            console.set_alt_screen(False)

    # ------------------------------------------------------------- 启动

    async def banner(self) -> None:
        connected = await self.client.health()
        if not self.model:
            self.model = await self.client.get_model()
        head = Text("SGLang Chat TUI\n", style="bold cyan")
        status = Text("server   ")
        status.append(self.base_url + "  ", style="bold")
        status.append("● 已连接" if connected else "● 未连接 — 请确认服务已启动 (端口/地址)",
                      style="green" if connected else "red")
        head.append_text(status)
        head.append("\nmodel    ", style="dim")
        head.append(self.model or "(未知, 首次请求时使用 default)")
        p = self.params
        head.append("\nparams   ", style="dim")
        head.append(f"temp={p.temperature} top_p={p.top_p} top_k={p.top_k} "
                    f"max_tokens={p.max_tokens} seed={p.seed}")
        head.append("\nhint     ", style="dim")
        head.append("直接输入对话 · PgUp/Ctrl+O 回看历史 · /help 命令 · /all 一键精度测试 · /q 退出")
        console.print(Panel(head, box=box.ROUNDED, border_style="cyan", padding=(0, 1)))

    async def run(self) -> None:
        await self.banner()
        session = PromptSession(history=InMemoryHistory())
        kb = KeyBindings()

        @kb.add("pageup", eager=True)
        @kb.add("c-o", eager=True)
        def _open_pager(event) -> None:
            event.app.exit(result=SCROLL)

        with patch_stdout(raw=True):
            while True:
                try:
                    line = await session.prompt_async(
                        HTML("<ansibold><ansicyan>you</ansicyan></ansibold> <ansicyan>❯</ansicyan> "),
                        key_bindings=kb)
                except KeyboardInterrupt:
                    console.print("[dim]^C (输入 /q 退出)[/dim]")
                    continue
                except EOFError:
                    break
                if line == SCROLL:
                    await self.pager()
                    continue
                try:
                    await self.dispatch(line)
                except Quit:
                    break
                except KeyboardInterrupt:
                    console.print("[dim]已中断当前操作[/dim]")
                except httpx.HTTPError as e:
                    console.print(f"[red]网络错误:[/red] {escape(str(e)[:300])}")
                except Exception as e:  # noqa: BLE001 — 保持 TUI 存活
                    console.print(f"[red]错误:[/red] {escape(type(e).__name__)}: {escape(str(e)[:300])}")
        console.print("[dim]bye[/dim]")


def main() -> None:
    ap = argparse.ArgumentParser(description="SGLang Chat TUI (默认连接 http://127.0.0.1:30000)")
    ap.add_argument("--base-url", default=os.environ.get("SGLANG_TUI_URL", "http://127.0.0.1:30000"),
                    help="sglang 服务地址 (env: SGLANG_TUI_URL)")
    ap.add_argument("--model", default=None, help="覆盖模型名 (默认自动从 /v1/models 获取)")
    ap.add_argument("--system", default=None, help="初始 system prompt")
    args = ap.parse_args()

    app = App(args.base_url.rstrip("/"), model=args.model, system=args.system)
    try:
        asyncio.run(app.run())
    except KeyboardInterrupt:
        console.print("^C 退出")


if __name__ == "__main__":
    main()
