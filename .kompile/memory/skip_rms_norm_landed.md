---
name: skip-rms-norm-landed
description: Fused skip_rms_norm op landed — 60 add kernels eliminated, +2.5% throughput (commit 31e5078e02)
type: project
---

Fused skip_rms_norm op merged residual add + RMS normalization into a single CUDA kernel.

**Impact:** 52.00 → 53.28 tok/s (+2.5%), correctness PASS. Commit 31e5078e02.

**What it does:** SkipSimplifiedLayerNormalization (2 per layer × 30 layers = 60 instances) previously emitted add + rms_norm (2 kernels each). Now emits skip_rms_norm (1 kernel). Total add ops in graph dropped from 216 to 63, rms_norm from 205 to 1.

**Files:** rms_norm.h, cuda/rms_norm.cu, cpu/rms_norm.cpp, llm_ops.cpp, OpTraitTable.cpp, llm.h, SkipRmsNorm.java, SDNN.java, DifferentialFunctionClassHolder.java, SkipSimplifiedLayerNormalization.kt

**How to apply:** Small kernel fusion gives diminishing returns at ~50 tok/s. The bottleneck is the single cudaStreamSynchronize per step (~13-14ms) which IS the graph replay execution time. Kernel count reduction saves launch overhead but doesn't reduce the actual compute.
