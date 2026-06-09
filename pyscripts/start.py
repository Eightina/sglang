from sglang.test.doc_patch import launch_server_cmd
from sglang.utils import wait_for_server, print_highlight, terminate_process

# This is equivalent to running the following command in your terminal
# python3 -m sglang.launch_server --model-path qwen/qwen2.5-0.5b-instruct --host 0.0.0.0

server_process, port = launch_server_cmd(
    """
python -m sglang.launch_server --config ./pyscripts/q.yaml
"""
)

wait_for_server(f"https://:{port}")


# python3 -m sglang.launch_server --model-path qwen/qwen3-0.6b --port 30000
# python3 -m sglang.launch_server --model-path qwen/qwen3.5-27b-fp8 --port 30000
