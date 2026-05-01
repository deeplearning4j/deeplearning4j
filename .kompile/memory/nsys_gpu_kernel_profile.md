---
name: nsys-gpu-kernel-profile
description: nsys GPU kernel profile — 22-island architecture verified, kernels inside graph replays reported as regular by CUPTI, split is only real gap op
type: project
---

# nsys GPU Kernel Profile — SmolDocling-256M Decode (2026-04-29)

**Profile**: `/tmp/nsys_profile_20260429_051332.sqlite` (250 tokens, RTX 4090, 53 tok/s baseline)
**Previous**: `/tmp/nsys_profile_20260427_182616.sqlite` (pre-checkIndices fix)

## CRITICAL: CUPTI Reporting Behavior

CUPTI reports kernels launched INSIDE CUDA graph replays as regular kernel events
(launchType=1 REGULAR). This means you CANNOT distinguish "gap kernels" from
"in-graph kernels" by just counting kernel events. You MUST cross-reference kernel
timestamps with CUPTI_ACTIVITY_KIND_GRAPH_TRACE timestamps to determine which
kernels are inside vs outside graph replays.

## Per-Step Architecture (nsys-verified)

Each decode step = 22 island-gap cycles on the GPU stream:

```
H2D(8B) → splitCuda(25us) → GRAPH_REPLAY(3us)    ← tiny island
H2D(8B) → splitCuda(25us) → GRAPH_REPLAY(3us)    ← tiny island
... (repeat 9 tiny + 1 medium + 12 full) ...
H2D(8B) → splitCuda(25us) → GRAPH_REPLAY(2400us) ← full island
[decode loop gap: sync + argmax + D2H + D2D KV scatter + updates]
```

## Key Metrics

- **22 graph replays/step** (from 416 distinct graphExecIds total)
- **22 split kernels/step** — the ONLY native gap kernel
- **Graph replay durations**: 9×3us + 1×1ms + 12×2.4ms = ~30ms nsys / ~12ms real
- **Decode loop gap**: ~37ms nsys / ~8ms real (sync + token handling)
- **Total step**: ~83ms nsys / ~19ms real (53 tok/s)

## Kernel Counts (total across all 232 steady-state steps)

These include kernels INSIDE graph replays:
- 14360 Assign<float,float16> (62/step) — FP32→FP16 cast in-graph
- 9738 Assign<float,float> (42/step) — copy in-graph
- 9555 gatherCudaLinearKernel (41/step) — gather in-graph
- 6064 cuBLAS GEMV fp16 (26/step) — matmul in-graph
- 4836 concatCuda<long> (21/step) — concat in-graph
- 4680 gatherCuda<float> (20/step) — gather in-graph
- 2757 shapeOfCudaKernel (12/step) — shape_of in-graph
- 2521 concatCuda<float> (11/step) — concat in-graph
- 2479 whereElementWise (11/step) — mask construction in-graph
- 2479 EqualTo (11/step) — mask construction in-graph
- 2478 tileKernelDouble (11/step) — tile in-graph
- 2340 skipRmsNormKernel (10/step) — normalization in-graph
- 2340 stackScalarsCuda (10/step) — stack in-graph
- 1680 triton_fused_fused_rope (7/step) — Triton RoPE
- 1116 broadcastAdd (5/step)
- 1080 fusedCausalMaskSoftmax (5/step)
- 840 triton_fused_swish_mul (4/step)
- 812 triton_fused_add (4/step)

## What This Means for Optimization

1. Gap ops are NOT the bottleneck — only 22 split kernels execute natively
2. All other ops (assign, gather, concat, etc.) are INSIDE the merged CUDA graph
3. The bottleneck is the total GPU compute time of the 22 graph replays
4. Reducing island count (merging islands) would reduce graph launch overhead
5. The split op forcing 22 island boundaries is the key structural limitation
