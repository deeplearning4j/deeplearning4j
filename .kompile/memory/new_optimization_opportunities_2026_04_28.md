---
name: new-optimization-opportunities-2026-04-28
description: "Optimization opportunities — nsys-verified targets: 22 island transitions, kernel fusion. GPU argmax already done, KV scatter already efficient."
type: project
---

## Optimization Opportunities (updated 2026-04-29)

### RESOLVED: GraphOptimizer now runs in GenerationPipeline
GenerationPipeline.java:230 calls `GraphOptimizer.optimize()` for models >= 100 ops.
Benchmark log confirms: "Optimization completed in 3613ms, 60 optimizations applied".
All sigmoid*x→swish fusions already applied.

### RESOLVED: GPU-side argmax
GPU argmax IS implemented in autoregressive_decode.cu. The argmaxLauncher kernel runs on
GPU, only 8B D2H for the token ID. The earlier claim of "3MB D2H logits readback" was WRONG —
that only happens in the Java fallback path (javaArgmax).

### RESOLVED: KV scatter is already efficient
The C++ decode loop uses kvScatterBatched — a single kernel copying ~135KB (1 position per
head per layer). The earlier claim of "3MB D2D per step" was WRONG — that was the Java-side
scatterKvToStatic which only runs during 2 warmup steps, not the native decode loop.

### nsys-verified targets (from /tmp/nsys_profile_20260429_051332.sqlite):

**1. HIGH: Reduce 22 islands per step**
Each step does 22 island-gap cycles. Each gap = H2D(8B) + splitCuda(25us) + graph launch.
The split ops are ONNX model artifacts at attention/layer boundaries. If split ops could
be absorbed into adjacent islands or the model restructured to avoid splits, island count
drops. TILE inclusion was tried → -6.4% regression. Need different approach.

**2. HIGH: Kernel fusion to reduce GPU compute**
matmul (252ms total) and onnx_multi_head_attention (183ms total) dominate. Per-step GPU
compute is the bottleneck at ~19ms/step. Need op fusion (skip_rms_norm already done,
rms_norm_linear already done, fused softmax already done).

**3. LOW: reshape_no_copy 105ms total — copies are LOAD-BEARING**
546 calls. Copies produce C-contiguous outputs for cuBLAS. Bypassing them regressed
matmul by +65ms (net -29%). NEVER optimize away these copies.

### FAILED attempts (do NOT retry):
- TILE in tritonIncludeTypes: -6.4% (2026-04-29)
- mergeViewGaps: -5.4% (2026-04-26)
- FuseGatedMLPPattern: -5.5% (2026-04-28)
- Pre-sync KV scatter move: no improvement (2026-04-29) — GPU compute is bottleneck, not CPU scheduling
- reshape_no_copy view bypass: -29% (2026-04-27)
