
baseline_short pp2tp2
```
============ Serving Benchmark Result ============
Backend:                                 sglang
Traffic request rate:                    inf
Max request concurrency:                 64
Successful requests:                     200
Benchmark duration (s):                  33.06
Total input tokens:                      25437
Total input text tokens:                 25437
Total generated tokens:                  6453
Total generated tokens (retokenized):    6434
Request throughput (req/s):              6.05
Input token throughput (tok/s):          769.39
Output token throughput (tok/s):         195.18
Peak output token throughput (tok/s):    373.00
Peak concurrent requests:                73
Total token throughput (tok/s):          964.57
Concurrency:                             56.51
----------------End-to-End Latency----------------
Mean E2E Latency (ms):                   9341.16
Median E2E Latency (ms):                 9641.10
P90 E2E Latency (ms):                    12654.31
P99 E2E Latency (ms):                    14336.93
---------------Time to First Token----------------
Mean TTFT (ms):                          6194.29
Median TTFT (ms):                        6872.58
P99 TTFT (ms):                           9083.55
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          98.54
Median TPOT (ms):                        102.03
P99 TPOT (ms):                           138.99
---------------Inter-Token Latency----------------
Mean ITL (ms):                           100.76
Median ITL (ms):                         62.25
P95 ITL (ms):                            268.15
P99 ITL (ms):                            420.41
Max ITL (ms):                            665.84
==================================================
```
baseline_long pp2tp2
```
============ Serving Benchmark Result ============
Backend:                                 sglang
Traffic request rate:                    inf
Max request concurrency:                 32
Successful requests:                     100
Benchmark duration (s):                  68.20
Total input tokens:                      50561
Total input text tokens:                 50561
Total generated tokens:                  25820
Total generated tokens (retokenized):    26090
Request throughput (req/s):              1.47
Input token throughput (tok/s):          741.34
Output token throughput (tok/s):         378.58
Peak output token throughput (tok/s):    594.00
Peak concurrent requests:                36
Total token throughput (tok/s):          1119.92
Concurrency:                             27.09
----------------End-to-End Latency----------------
Mean E2E Latency (ms):                   18478.19
Median E2E Latency (ms):                 17500.08
P90 E2E Latency (ms):                    29912.99
P99 E2E Latency (ms):                    36534.34
---------------Time to First Token----------------
Mean TTFT (ms):                          5559.47
Median TTFT (ms):                        5948.13
P99 TTFT (ms):                           11935.21
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          51.08
Median TPOT (ms):                        51.25
P99 TPOT (ms):                           72.69
---------------Inter-Token Latency----------------
Mean ITL (ms):                           50.24
Median ITL (ms):                         37.10
P95 ITL (ms):                            144.00
P99 ITL (ms):                            302.69
Max ITL (ms):                            1693.59
==================================================

```