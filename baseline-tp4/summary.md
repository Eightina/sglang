
baseline_short tp4
```
============ Serving Benchmark Result ============
Backend:                                 sglang
Traffic request rate:                    inf
Max request concurrency:                 64
Successful requests:                     200
Benchmark duration (s):                  14.64
Total input tokens:                      25437
Total input text tokens:                 25437
Total generated tokens:                  6453
Total generated tokens (retokenized):    6482
Request throughput (req/s):              13.66
Input token throughput (tok/s):          1737.23
Output token throughput (tok/s):         440.71
Peak output token throughput (tok/s):    808.00
Peak concurrent requests:                83
Total token throughput (tok/s):          2177.94
Concurrency:                             58.63
----------------End-to-End Latency----------------
Mean E2E Latency (ms):                   4292.37
Median E2E Latency (ms):                 4177.61
P90 E2E Latency (ms):                    6553.20
P99 E2E Latency (ms):                    7846.01
---------------Time to First Token----------------
Mean TTFT (ms):                          1583.23
Median TTFT (ms):                        1602.64
P99 TTFT (ms):                           3296.73
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          87.97
Median TPOT (ms):                        94.55
P99 TPOT (ms):                           134.91
---------------Inter-Token Latency----------------
Mean ITL (ms):                           86.73
Median ITL (ms):                         119.42
P95 ITL (ms):                            148.68
P99 ITL (ms):                            240.74
Max ITL (ms):                            1294.70
==================================================
```
baseline_long tp4
```
============ Serving Benchmark Result ============
Backend:                                 sglang
Traffic request rate:                    inf
Max request concurrency:                 32
Successful requests:                     100
Benchmark duration (s):                  43.01
Total input tokens:                      50561
Total input text tokens:                 50561
Total generated tokens:                  25820
Total generated tokens (retokenized):    26316
Request throughput (req/s):              2.33
Input token throughput (tok/s):          1175.60
Output token throughput (tok/s):         600.35
Peak output token throughput (tok/s):    1184.00
Peak concurrent requests:                37
Total token throughput (tok/s):          1775.95
Concurrency:                             28.00
----------------End-to-End Latency----------------
Mean E2E Latency (ms):                   12041.46
Median E2E Latency (ms):                 11977.26
P90 E2E Latency (ms):                    21499.77
P99 E2E Latency (ms):                    24911.75
---------------Time to First Token----------------
Mean TTFT (ms):                          763.37
Median TTFT (ms):                        259.24
P99 TTFT (ms):                           3247.27
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          45.83
Median TPOT (ms):                        46.33
P99 TPOT (ms):                           73.54
---------------Inter-Token Latency----------------
Mean ITL (ms):                           43.86
Median ITL (ms):                         26.90
P95 ITL (ms):                            205.76
P99 ITL (ms):                            305.65
Max ITL (ms):                            3152.56
==================================================
```