---
name: dsp_benchmark_where_static_output_first_250_20260528
description: 
type: project
---

First 250-token benchmark after 3-input Where trait refinement (clear OP_TRAIT_DYNAMIC_OUTPUT_SIZE alongside DATA_DEPENDENT in NativePlanCompiler and NativeDynamicShapePlan deserialize). Command: platform-tests/run-benchmark.sh --tokens 250 --skip-audit. Metrics: steady=57.57 tok/s, lateSteady=65.64 tok/s, overall=22.76 tok/s. Baseline: steady=57.48, lateSteady=64.96. Replay health OK: segments=1, captured=1/1, replays=245, capture valid=true. Maven/JUnit failed only because the benchmark target assertion is still 100 tok/s; this is a target miss, not a crash/correctness failure. Improvement is small; confirm with a second 250-token run before commit.
