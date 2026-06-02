---
name: dsp_benchmark_where_static_output_rejected_250_20260528
description: 
type: project
---

3-input Where OP_TRAIT_DYNAMIC_OUTPUT_SIZE refinement was rejected on 2026-05-28. Change under test: clear OP_TRAIT_DYNAMIC_OUTPUT_SIZE for 3-input Where in NativePlanCompiler and NativeDynamicShapePlan deserialize path. Build passed. First 250-token run: steady=57.57, lateSteady=65.64; baseline was steady=57.48, lateSteady=64.96, so first run looked like a tiny/noisy improvement. Confirmation 250-token run: steady=54.64, lateSteady=62.20, below baseline. Replay health was OK in both runs, output coherent, Maven failed only due target assertion below 100. Decision: reject, revert exact source lines, rebuild native artifacts to restore installed baseline before next optimization.
