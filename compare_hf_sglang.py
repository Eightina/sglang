#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Qwen3.8-27B-NVFP4 数值问题对比脚本：HF transformers golden vs sglang。

核心问题（稳定可复现）
----------------------
sglang 服务 unsloth/Qwen3.8-27B-NVFP4 时，第一个 full-attention 层
（layers.3）的 RadixAttention 输出比 HF 参考小约 39 倍
（norm 2.23 vs 87.31），从该层起误差向下游全部层级累积，
导致生成文本渐进崩坏（首 token 正确 -> 逐 token 漂移 -> 乱码/循环）。

已验证正确的部分（问题不在这些环节）：
  - checkpoint 数据自洽（NVFP4 反量化 vs MTP 层 bf16 参考同量级）
  - NVFP4 W4A4 MLP（单层手算 rel_err=9.6%，双重量化理论范围）
  - FP8 W8A8 投影（rel_err=2.6%）、lm_head（logprobs 幅度正常）
  - qkv_proj 投影输出（与 HF 逐位一致，cos=0.99998）
  - RoPE 配置（theta=1e7, 32 对频率, partial 0.25，两引擎一致；
    config 的 mrope_interleaved 在纯文本 T=H=W 下重排为恒等）
  - GDN 线性注意力层（无 RoPE，层 0-2 与 HF 一致）
  - 换 attention 后端（flashinfer/triton）、禁用 cuda graph /
    radix cache / overlap 均无法解决 -> 问题在后端共同路径。

用法
----
# 1) 采集 HF golden 参考（CPU 上跑 27B，约 8-15 分钟；结果缓存后不再重跑）
#    需要: pip install --no-deps accelerate
python3 compare_hf_sglang.py --collect-hf

# 2) 快速对比（服务器正常模式即可；展示 logprobs 分歧 + 贪心续写分歧）
python3 compare_hf_sglang.py --logprobs

# 3) 逐层定位（需以 dump 模式重启 sglang 后再跑）：
#    python -m sglang.launch_server <常规参数> \
#        --debug-tensor-dump-output-folder /tmp/sgl_dump \
#        --debug-tensor-dump-layers 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15
#    然后向 127.0.0.1:30000 发一条请求（任意内容）后运行：
python3 compare_hf_sglang.py --compare

判定标准：attn_ratio = norm(sglang 层3 attn 输出) / norm(HF 层3 attn 核心)
正常应 ≈ 1.0；当前 ≈ 0.026（39 倍缩小）即命中本问题。
"""

import argparse
import glob
import json
import os
import sys

import torch

BASE = "/sgl-workspace/sglang"
MODEL_DIR = "/sgl-workspace/models/Qwen3.8-27B-NVFP4"
SGLANG_URL = "http://127.0.0.1:30000"
HF_CACHE = os.path.join(BASE, "hf_golden_cache.pt")
DUMP_ROOT = "/tmp/sgl_dump"
PROMPT = "The capital of France is"  # 固定 5 token，缓存/dump 均按此对齐
N_HOOK_LAYERS = 8  # HF 侧 hook 层数（覆盖层 3 = 第一个 FA 层）

import requests


# ---------------------------------------------------------------- HF golden
def collect_hf(cache_path: str) -> None:
    """加载 HF 模型（CPU），hook 抓层 0-7 内部输出 + hidden states + logprobs。"""
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        sys.exit("需要 transformers")
    print("loading HF model (CPU, 首次约 3-4 分钟) ...", flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR, dtype=torch.bfloat16, device_map="cpu"
    )
    model.eval()

    captured = {}

    def mk_hook(name):
        def hook(module, inp, out):
            t = out[0] if isinstance(out, tuple) else out
            if isinstance(t, torch.Tensor):
                captured[name] = t.detach().float().cpu()
        return hook

    def mk_input_hook(name):
        def hook(module, inp, out):
            if inp and isinstance(inp[0], torch.Tensor):
                captured[name] = inp[0].detach().float().cpu()
        return hook

    text = getattr(model, "language_model", None) or model.model
    if hasattr(text, "model"):
        text = text.model
    layers = text.layers
    print("text layers:", len(layers), flush=True)
    handles = []
    for i in range(min(N_HOOK_LAYERS, len(layers))):
        layer = layers[i]
        if hasattr(layer, "linear_attn"):  # GDN 层
            gdn = layer.linear_attn
            for sub in ("in_proj_qkv", "in_proj_z"):
                handles.append(
                    getattr(gdn, sub).register_forward_hook(
                        mk_hook(f"model.layers.{i}.{sub}")
                    )
                )
            handles.append(
                gdn.out_proj.register_forward_hook(
                    mk_hook(f"model.layers.{i}.linear_attn.out_proj")
                )
            )
        else:  # full-attention 层
            attn = layer.self_attn
            handles.append(attn.q_proj.register_forward_hook(mk_hook(f"model.layers.{i}.qkv_q")))
            handles.append(attn.k_proj.register_forward_hook(mk_hook(f"model.layers.{i}.qkv_k")))
            handles.append(attn.v_proj.register_forward_hook(mk_hook(f"model.layers.{i}.qkv_v")))
            # o_proj 的“输入”= gated attention 核心输出（HF 语义下的 attn*sigmoid(gate)）
            handles.append(attn.o_proj.register_forward_hook(mk_input_hook(f"model.layers.{i}.attn_gated")))
            handles.append(attn.o_proj.register_forward_hook(mk_hook(f"model.layers.{i}.o_proj")))
        handles.append(layer.mlp.down_proj.register_forward_hook(mk_hook(f"model.layers.{i}.mlp.down_proj")))

    inputs = tok(PROMPT, return_tensors="pt")
    print("forward ...", flush=True)
    with torch.no_grad():
        out = model(**inputs, output_hidden_states=True)

    # 立即快照 prefill 的 hook 数据（后续 generate 的 decode 会覆盖同名键）
    captured_prefill = {k: v.clone() for k, v in captured.items()}
    hidden = [h.float().cpu() for h in out.hidden_states]

    # top-10 logprobs（golden）
    logits = out.logits[0, -1].float()
    top = torch.topk(torch.softmax(logits, -1), 10)
    golden_logprobs = [(tok.decode([i]), torch.log(p).item()) for p, i in zip(top.values, top.indices)]
    greedy = tok.decode(model.generate(**inputs, max_new_tokens=6, do_sample=False)[0][inputs["input_ids"].shape[1]:])

    for h in handles:
        h.remove()

    cache = {
        "prompt": PROMPT,
        "hidden_states": hidden,
        "modules": captured_prefill,
        "logprobs": golden_logprobs,
        "greedy": greedy,
    }
    torch.save(cache, cache_path)
    print(f"golden 已缓存 -> {cache_path}", flush=True)
    print("top-10 logprobs:")
    for t, lp in golden_logprobs:
        print(f"  {t!r}: {lp:.3f}")
    print("greedy:", repr(greedy))


def load_hf_cache(cache_path: str) -> dict:
    if not os.path.exists(cache_path):
        sys.exit(f"缓存不存在: {cache_path}\n先运行: python3 {sys.argv[0]} --collect-hf")
    cache = torch.load(cache_path, map_location="cpu", weights_only=False)
    if cache.get("prompt") != PROMPT:
        sys.exit(f"缓存的 prompt 不符: {cache['prompt']!r} != {PROMPT!r}，请删除缓存重采")
    return cache


# ---------------------------------------------------------------- sglang 侧
def query_logprobs() -> tuple[list, str]:
    r = requests.post(
        f"{SGLANG_URL}/v1/completions",
        json={"model": "test", "prompt": PROMPT, "temperature": 0,
              "max_tokens": 6, "logprobs": 10},
        timeout=600,
    )
    if r.status_code != 200:
        sys.exit(f"HTTP {r.status_code}: {r.text[:300]}")
    d = r.json()
    ch = d["choices"][0]
    first = ch["logprobs"]["top_logprobs"][0]
    logprobs = sorted(first.items(), key=lambda kv: kv[1], reverse=True)[:10]
    return logprobs, ch["text"]


def trigger_and_find_dump(hf_cache: dict) -> str:
    """发一条请求触发 dump，返回 5-token prefill pass（过滤掉 warmup/decode pass）。"""
    seq_len = hf_cache["hidden_states"][0].shape[1]  # HF hidden: [1, seq, hidden]
    r = requests.post(
        f"{SGLANG_URL}/generate",
        json={"text": PROMPT, "sampling_params": {"temperature": 0, "max_new_tokens": 1}},
        timeout=600,
    )
    if r.status_code != 200:
        sys.exit(f"触发请求失败 HTTP {r.status_code}: {r.text[:300]}")
    passes = sorted(glob.glob(f"{DUMP_ROOT}/*/Pass*.pt"), key=os.path.getmtime, reverse=True)
    for p in passes:
        d = torch.load(p, map_location="cpu", weights_only=False)
        emb = d.get("model.embed_tokens")
        if isinstance(emb, torch.Tensor) and emb.shape[0] == seq_len:
            return p
    sys.exit(f"未找到 seq_len={seq_len} 的 prefill pass；请确认 dump 参数与请求内容一致")


def load_sglang_dump(path: str) -> dict:
    d = torch.load(path, map_location="cpu", weights_only=False)
    out = {}
    for k, v in d.items():
        if isinstance(v, list):
            v = v[0] if len(v) == 1 else v
        if isinstance(v, torch.Tensor):
            out[k] = v.float()
    return out


# ---------------------------------------------------------------- 对比
def cmp(a: torch.Tensor, b: torch.Tensor):
    a, b = a.float().reshape(-1), b.float().reshape(-1)
    n = min(a.numel(), b.numel())
    a, b = a[:n], b[:n]
    cos = torch.nn.functional.cosine_similarity(a, b, dim=0).item()
    rel = ((a - b).norm() / (b.norm() + 1e-9)).item()
    return cos, rel


def compare_logprobs(hf_cache: dict) -> bool:
    print("=== logprobs 对比（prefill 最后一个 token 的 next-token 分布）===")
    sgl_lp, sgl_text = query_logprobs()
    hf_lp = hf_cache["logprobs"]
    print(f"{'HF golden':>28} | {'sglang':>28}")
    for i in range(10):
        ht, hp = hf_lp[i] if i < len(hf_lp) else ("", 0.0)
        st, sp = sgl_lp[i] if i < len(sgl_lp) else ("", 0.0)
        print(f"{ht!r:>24} {hp:>7.3f} | {st!r:>24} {sp:>7.3f}")
    hf_top1 = hf_lp[0][0]
    sgl_top1 = sgl_lp[0][0]
    gap = abs(hf_lp[0][1] - sgl_lp[0][1])
    print(f"\ntop-1: HF={hf_top1!r} sglang={sgl_top1!r} | logprob 差 = {gap:.3f}")
    print(f"greedy 续写: HF={hf_cache['greedy']!r}  sglang={sgl_text!r}")
    ok = hf_top1 == sgl_top1 and gap < 0.3
    print(f"[{'PASS' if ok else 'FAIL'}] top-1 一致且 logprob 差 < 0.3"
          f"{'' if ok else '  <-- 与 HF 系统性偏离，命中数值问题'}")
    return ok


def compare_dump(hf_cache: dict, dump_path: str) -> bool:
    d = load_sglang_dump(dump_path)
    hfm = hf_cache["modules"]
    hf_hidden = hf_cache["hidden_states"]

    def sgl(k):
        return d[k].reshape(5, -1)

    def hfm_(k):
        t = hfm[k].float()
        if t.dim() == 3:
            t = t[0]  # 去 batch 维 (1, seq, d) -> (seq, d)
        return t.reshape(5, -1)

    print(f"dump: {dump_path}\n")
    print("=== 1) 逐层残差对照：sglang (attn分支+mlp分支) vs HF hidden delta ===")
    print(f"{'layer':>5} {'type':>5} {'cos':>9} {'rel_err':>9}")
    worst = None
    for n in range(16):
        ak = f"model.layers.{n}.linear_attn.out_proj"
        typ = "GDN"
        if ak not in d:
            ak = f"model.layers.{n}.o_proj"
            typ = "FA"
        delta_s = (sgl(ak) + d[f"model.layers.{n}.mlp.down_proj"]).reshape(-1)
        delta_h = (hf_hidden[n + 1] - hf_hidden[n]).reshape(-1)
        cos, rel = cmp(delta_s, delta_h)
        mark = ""
        if typ == "FA" and worst is None:
            worst = n
            mark = "  <-- 第一个 FA 层，首个发散点"
        print(f"{n:>5} {typ:>5} {cos:>9.5f} {rel:>9.4f}{mark}")

    print("\n=== 2) 层 3（第一个 FA 层）内部定位 ===")
    sgl_qkv = sgl("model.layers.3.qkv_proj")          # [q_gate 12288 | k 1024 | v 1024]
    seg_map = [("q+gate", slice(0, 12288), "model.layers.3.qkv_q"),
               ("k", slice(12288, 13312), "model.layers.3.qkv_k"),
               ("v", slice(13312, None), "model.layers.3.qkv_v")]
    for name, sl, hk in seg_map:
        cos, rel = cmp(sgl_qkv[:, sl], hfm_(hk))
        print(f"  qkv_proj {name:>7} 段: cos={cos:.5f} rel={rel:.4f}  (投影正确)")

    hf_attn_gated = hfm_("model.layers.3.attn_gated")  # HF: attn*sigmoid(gate)，即 o_proj 输入
    hf_gate = hfm_("model.layers.3.qkv_q").reshape(5, 24, 512)[:, :, 256:].reshape(5, -1)
    hf_attn_core = hf_attn_gated / torch.sigmoid(hf_gate)   # 还原 gate 乘法前
    sgl_attn_core = sgl("model.layers.3.attn")             # sglang RadixAttention 输出（gate 乘法前）

    n_hf = hf_attn_core.norm().item()
    n_sgl = sgl_attn_core.norm().item()
    ratio = n_sgl / n_hf
    print(f"\n  attention 核心输出（gate 乘法前）:")
    print(f"    HF norm     = {n_hf:.3f}")
    print(f"    sglang norm = {n_sgl:.3f}")
    print(f"    ratio (sgl/hf) = {ratio:.4f}")
    ok = abs(ratio - 1.0) < 0.5
    print(f"  [{'PASS' if ok else 'FAIL'}] 比值应 ≈ 1.0"
          f"{f'；当前缩小 {1/ratio:.0f} 倍 -> 命中本问题' if not ok else ''}")
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--collect-hf", action="store_true", help="采集/刷新 HF golden 缓存（CPU 慢）")
    ap.add_argument("--logprobs", action="store_true", help="快速对比：logprobs + 贪心续写")
    ap.add_argument("--compare", action="store_true", help="逐层定位：解析 sglang dump 与 HF 对照")
    ap.add_argument("--cache", default=HF_CACHE)
    args = ap.parse_args()

    if args.collect_hf:
        collect_hf(args.cache)
    if args.logprobs:
        compare_logprobs(load_hf_cache(args.cache))
    if args.compare:
        hf_cache = load_hf_cache(args.cache)
        compare_dump(hf_cache, trigger_and_find_dump(hf_cache))
    if not any([args.collect_hf, args.logprobs, args.compare]):
        ap.print_help()


if __name__ == "__main__":
    main()
