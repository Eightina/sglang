"""HumanEval driver: force the multiprocessing start method to spawn.

human_eval.execution.check_correctness sandboxes each completion in a
multiprocessing.Process; with the default fork start method, filelock>=3.32's
fork audit hook aborts the run ("os.fork is unsafe while filelock is changing
descriptor ownership"). Spawn avoids os.fork entirely.
"""

import multiprocessing
import runpy
import sys

if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    sys.argv = ["sglang.test.run_eval"] + sys.argv[1:]
    runpy.run_module("sglang.test.run_eval", run_name="__main__")
