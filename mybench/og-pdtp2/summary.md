og_short pdtp2
```
============ Serving Benchmark Result ============
Backend:                                 sglang
Traffic request rate:                    inf
Max request concurrency:                 64
Successful requests:                     200
Benchmark duration (s):                  38.17
Total input tokens:                      25437
Total input text tokens:                 25437
Total generated tokens:                  6453
Total generated tokens (retokenized):    6456
Request throughput (req/s):              5.24
Input token throughput (tok/s):          666.40
Output token throughput (tok/s):         169.06
Peak output token throughput (tok/s):    238.00
Peak concurrent requests:                75
Total token throughput (tok/s):          835.45
Concurrency:                             53.62
----------------End-to-End Latency----------------
Mean E2E Latency (ms):                   10233.79
Median E2E Latency (ms):                 11055.53
P90 E2E Latency (ms):                    12625.49
P99 E2E Latency (ms):                    14904.87
---------------Time to First Token----------------
Mean TTFT (ms):                          9300.38
Median TTFT (ms):                        10147.72
P99 TTFT (ms):                           13928.62
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          29.47
Median TPOT (ms):                        30.37
P99 TPOT (ms):                           31.95
---------------Inter-Token Latency----------------
Mean ITL (ms):                           29.88
Median ITL (ms):                         30.46
P95 ITL (ms):                            32.24
P99 ITL (ms):                            33.17
Max ITL (ms):                            35.05
==================================================

```
og_long pdtp2

```
============ Serving Benchmark Result ============
Backend:                                 sglang
Traffic request rate:                    inf
Max request concurrency:                 32
Successful requests:                     100
Benchmark duration (s):                  52.03
Total input tokens:                      50561
Total input text tokens:                 50561
Total generated tokens:                  25820
Total generated tokens (retokenized):    26293
Request throughput (req/s):              1.92
Input token throughput (tok/s):          971.76
Output token throughput (tok/s):         496.25
Peak output token throughput (tok/s):    629.00
Peak concurrent requests:                42
Total token throughput (tok/s):          1468.00
Concurrency:                             26.07
----------------End-to-End Latency----------------
Mean E2E Latency (ms):                   13561.98
Median E2E Latency (ms):                 13642.61
P90 E2E Latency (ms):                    20376.72
P99 E2E Latency (ms):                    26160.73
---------------Time to First Token----------------
Mean TTFT (ms):                          4920.96
Median TTFT (ms):                        4614.20
P99 TTFT (ms):                           12518.28
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          33.72
Median TPOT (ms):                        34.06
P99 TPOT (ms):                           34.55
---------------Inter-Token Latency----------------
Mean ITL (ms):                           33.60
Median ITL (ms):                         33.89
P95 ITL (ms):                            35.37
P99 ITL (ms):                            36.84
Max ITL (ms):                            50.59
==================================================

```

og_agent pdtp2
```
============ Serving Benchmark Result ============
Backend:                                 sglang
Traffic request rate:                    inf
Max request concurrency:                 32
Successful requests:                     100
Benchmark duration (s):                  31.96
Total input tokens:                      108929
Total input text tokens:                 108929
Total generated tokens:                  2353
Total generated tokens (retokenized):    2426
Request throughput (req/s):              3.13
Input token throughput (tok/s):          3408.50
Output token throughput (tok/s):         73.63
Peak output token throughput (tok/s):    153.00
Peak concurrent requests:                38
Total token throughput (tok/s):          3482.13
Concurrency:                             26.95
----------------End-to-End Latency----------------
Mean E2E Latency (ms):                   8614.16
Median E2E Latency (ms):                 9467.19
P90 E2E Latency (ms):                    10338.12
P99 E2E Latency (ms):                    11477.45
---------------Time to First Token----------------
Mean TTFT (ms):                          8076.12
Median TTFT (ms):                        8869.10
P99 TTFT (ms):                           10433.66
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          20.86
Median TPOT (ms):                        24.39
P99 TPOT (ms):                           30.60
---------------Inter-Token Latency----------------
Mean ITL (ms):                           23.93
Median ITL (ms):                         29.52
P95 ITL (ms):                            31.13
P99 ITL (ms):                            31.91
Max ITL (ms):                            35.40
==================================================
```

og_rag pdtp2

```
============ Serving Benchmark Result ============
Backend:                                 sglang
Traffic request rate:                    inf
Max request concurrency:                 32
Successful requests:                     100
Benchmark duration (s):                  128.71
Total input tokens:                      209281
Total input text tokens:                 209281
Total generated tokens:                  52444
Total generated tokens (retokenized):    52501
Request throughput (req/s):              0.78
Input token throughput (tok/s):          1625.96
Output token throughput (tok/s):         407.45
Peak output token throughput (tok/s):    572.00
Peak concurrent requests:                36
Total token throughput (tok/s):          2033.41
Concurrency:                             26.00
----------------End-to-End Latency----------------
Mean E2E Latency (ms):                   33469.84
Median E2E Latency (ms):                 34432.01
P90 E2E Latency (ms):                    45917.02
P99 E2E Latency (ms):                    62005.59  
---------------Time to First Token----------------
Mean TTFT (ms):                          15638.81
Median TTFT (ms):                        16182.65
P99 TTFT (ms):                           29123.43
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          33.88
Median TPOT (ms):                        34.21
P99 TPOT (ms):                           35.30
---------------Inter-Token Latency----------------
Mean ITL (ms):                           34.07
Median ITL (ms):                         34.53
P95 ITL (ms):                            36.01
P99 ITL (ms):                            36.98
Max ITL (ms):                            51.94
==================================================

```