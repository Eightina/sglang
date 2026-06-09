好的，两份文档我帮你梳理了。以下是针对你当前场景（qwen3-0.6b，4× RTX PRO 4000 Blackwell）的实操方案。

  # 一、四个工具，选哪个

  │           工具           │              场景               │      你需要用吗      │

  │ bench_serving            │ 在线服务性能（TTFT、ITL、吞吐） │ ✅ 主力工具          │

  │ bench_offline_throughput │ 极限吞吐（跳过 HTTP 开销）      │ ✅ 测上限用          │

  │ bench_one_batch_server   │ 单 batch 延迟                   │ 可选                 │

  │ bench_one_batch          │ kernel 级 profiling             │ 做 kernel 优化时再用 │

  # 二、建立 Baseline

  # 2.1 在线服务性能（最重要）


    find ~/.cache -name "qwen3.5-27b-fp8" -type d 2>/dev/null | head -5

  ### 场景 1：短输入短输出（高并发吞吐）
  python3 -m sglang.bench_serving \
    --backend sglang \
    --host 127.0.0.1 --port 30000 \
    --model /root/.cache/modelscope/hub/models/qwen/qwen3___5-27b-fp8 \
    --dataset-name random-ids \
    --random-input-len 256 --random-output-len 64 \
    --num-prompts 200 \
    --request-rate inf \
    --max-concurrency 64 \
    --output-file ./baseline-pp2tp2/baseline_short.jsonl

  python3 -m sglang.bench_serving \
    --backend sglang \
    --host 127.0.0.1 --port 8000 \
    --model /root/.cache/modelscope/hub/models/qwen/qwen3___5-27b-fp8 \
    --dataset-name random-ids \
    --random-input-len 256 --random-output-len 64 \
    --num-prompts 200 \
    --request-rate inf \
    --max-concurrency 64 \
    --output-file ./mybench/og-pdtp2/og_short.jsonl

  ### 场景 2：长输入长输出（接近真实 LLM 使用）
  python3 -m sglang.bench_serving \
    --backend sglang \
    --host 127.0.0.1 --port 30000 \
    --model /root/.cache/modelscope/hub/models/qwen/qwen3___5-27b-fp8 \
    --dataset-name random-ids \
    --random-input-len 1024 --random-output-len 512 \
    --num-prompts 100 \
    --request-rate inf \
    --max-concurrency 32 \
    --output-file ./baseline-pp2tp2/baseline_long.jsonl

  python3 -m sglang.bench_serving \
    --backend sglang \
    --host 127.0.0.1 --port 8000 \
    --model /root/.cache/modelscope/hub/models/qwen/qwen3___5-27b-fp8 \
    --dataset-name random-ids \
    --random-input-len 1024 --random-output-len 512 \
    --num-prompts 100 \
    --request-rate inf \
    --max-concurrency 32 \
    --output-file ./mybench/og-pdtp2/og_long.jsonl

  <!-- ### 场景 3：真实对话分布（ShareGPT 数据集）
  python3 -m sglang.bench_serving \
    --backend sglang \
    --host 127.0.0.1 --port 30000 \
    --model qwen/qwen3.5-27b-fp8 \
    --dataset-name sharegpt \
    --num-prompts 200 \
    --output-file ./baseline/baseline_sharegpt.jsonl -->

  ## 2.2 极限吞吐（无 HTTP 开销）

  python3 -m sglang.bench_offline_throughput \
    --model-path qwen/qwen3.5-27b-fp8 \
    --dataset-name random \
    --num-prompts 100 \
    --random-input-len 512 --random-output-len 256 \
    --output-file ./baseline/baseline_maxthoughput.jsonl


  ## 2.3 关键指标说明

  │              指标               │               含义               │  关注场景   │

  │ Output token throughput (tok/s) │ 输出吞吐                         │ 最核心指标  │

  │ TTFT (ms)                       │ 首 token 延迟                    │ 交互体验    │

  │ ITL (ms)                        │ token 间延迟                     │ 流式体验    │

  │ TPOT (ms)                       │ 每 token 处理时间（首 token 后） │ decode 效率 │

  │ End-to-End Latency (ms)         │ 端到端延迟                       │ 整体        │

  # 三、可调的服务参数

  重启服务时加这些参数来对比：

  # 常用调优参数

```
python -m sglang.launch_server --model-path qwen/qwen3.5-27b-fp8 \
--max-running-requests 64 \        # 最大并发请求数（默认自动）
--chunked-prefill-size 8192 \      # chunked prefill 大小
--mem-fraction-static 0.88 \       # KV cache 占显存比例（默认 0.88）
--disable-radix-cache \            # 禁用 prefix cache（无共享前缀时）
--disable-cuda-graph \             # 禁用 CUDA graph（调试用，会降低性能）
--speculative-algorithm EAGLE3 \   # 投机解码（需要 draft model）
--tp-size 2                        # tensor parallel（多卡）
```

  # 四、对比工作流

  ## 1. Baseline
  python3 -m sglang.bench_serving ... --output-file baseline.jsonl

  ## 2. 改参数，重启服务，再测
  python3 -m sglang.bench_serving ... --output-file tuned.jsonl

  ## 3. 对比
   JSONL 里每行是一个 JSON，直接比较 throughput/latency 字段
  python3 -c "
  import json
  for f in ['baseline.jsonl', 'tuned.jsonl']:
      r = json.loads(open(f).readlines()[-1])
      print(f'{f}: {r[\"output_throughput\"]:.1f} tok/s, TTFT={r[\"mean_ttft_ms\"]:.1f}ms')
  "
