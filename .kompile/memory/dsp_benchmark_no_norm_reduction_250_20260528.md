---
name: dsp_benchmark_no_norm_reduction_250_20260528
description: 250-token no-normalization/no-reduction compile-all config was much slower than OPTIMAL
type: project
---

Command: platform-tests/run-benchmark.sh --tokens 250 --config DIAG_Triton_gc_noATTN --skip-audit\nLog: /tmp/dsp-benchmark-config-no-norm-reduction-250-20260528.log\nResult: PASS as JUnit benchmark, but performance regression.\nMetrics: steady=30.99 tok/s, lateSteady=32.95 tok/s, overall=11.30 tok/s.\nBaseline: steady=57.48 tok/s, lateSteady=64.96 tok/s.\nReplay health: segments=1, captured=1/1, replays=245, capture valid=true, Triton launches=1166, no perm/OOM failures.\nDecision: reject this config; do not change OPTIMAL to the no-ATTENTION/no-normalization type set.
