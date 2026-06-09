import argparse
import json
import time
import requests


def test_basic(host="127.0.0.1", port=30000, stream=False):
    """Basic chat completion test"""
    url = f"http://{host}:{port}/v1/chat/completions"

    data = {
        # "model": "qwen/qwen3-0.6b",
        "model": "qwen/qwen3.6-27b-fp8",
        "messages": [{"role": "user", "content": "What is the capital of France?"}],
        "stream": stream,
    }

    print(f"Testing {url}...")
    print(f"Request: {data['messages'][0]['content']}")
    print("-" * 60)

    start = time.time()
    response = requests.post(url, json=data, stream=stream)

    if stream:
        print("Response (streaming): ", end="", flush=True)
        for line in response.iter_lines():
            if line:
                line_str = line.decode("utf-8")
                if line_str.startswith("data: ") and line_str != "data: [DONE]":
                    chunk = json.loads(line_str[6:])
                    if chunk["choices"][0].get("delta", {}).get("content"):
                        print(chunk["choices"][0]["delta"]["content"], end="", flush=True)
        print()
    else:
        result = response.json()
        content = result["choices"][0]["message"]["content"]
        print(f"Response:\n{content}")
        print("-" * 60)
        print(f"Usage: {result['usage']}")

    elapsed = time.time() - start
    print(f"Latency: {elapsed:.3f}s")
    print("-" * 60)


def test_health(host="127.0.0.1", port=30000):
    """Test health endpoint"""
    url = f"http://{host}:{port}/health"
    try:
        response = requests.get(url, timeout=5)
        print(f"Health check: {response.status_code} - {response.text}")
    except Exception as e:
        print(f"Health check failed: {e}")


def test_model_info(host="127.0.0.1", port=30000):
    """Test model info endpoint"""
    url = f"http://{host}:{port}/model_info"
    try:
        response = requests.get(url, timeout=5)
        info = response.json()
        print(f"Model: {info.get('model_path', 'N/A')}")
        print(f"Max context length: {info.get('max_context_len', 'N/A')}")
    except Exception as e:
        print(f"Model info failed: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SGLang test client")
    parser.add_argument("--host", default="127.0.0.1", help="Server host")
    parser.add_argument("--port", type=int, default=30000, help="Server port")
    parser.add_argument("--stream", action="store_true", help="Enable streaming")
    parser.add_argument("--health", action="store_true", help="Run health check")
    parser.add_argument("--model-info", action="store_true", help="Show model info")
    args = parser.parse_args()

    if args.health:
        test_health(args.host, args.port)
    elif args.model_info:
        test_model_info(args.host, args.port)
    else:
        test_basic(args.host, args.port, args.stream)