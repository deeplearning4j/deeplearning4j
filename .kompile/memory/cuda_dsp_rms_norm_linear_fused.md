---
name: cuda_dsp_rms_norm_linear_fused
description: Fused rms_norm_linear kernel landed — 51.88 tok/s, eliminates 9-kernel decomposition but bottleneck is elsewhere
type: project
---

Fused rms_norm_linear kernel committed (11005b4ae6) on 2026-04-27.

**What it does:** Replaces 9-kernel decomposed rms_norm_linear (square, mean, add_eps, sqrt, reciprocal, multiply×2, mmul, assign) with single fused helper call.

**CUDA implementation:**
- M=1 decode: single-kernel shared-memory path (3-phase: warp-reduce invRMS, normalize to shmem, grid-stride dot products)
- M>1 prefill: rmsNorm kernel + cuBLAS GEMM (2 launches)
- Launch dims registered in LaunchDims.h/.cu as `rms_norm_linear`
- Shared memory default 33792 bytes (covers K up to 8192)

**CPU implementation:**
- PRAGMA_OMP_SIMD_ARGS(reduction(+:sumSq)) on sum-of-squares
- PRAGMA_OMP_SIMD on normalize loops
- Delegates matmul to MmulHelper::mmul (MKL/OneDNN/BLAS)
- parallel_tad for row parallelism

**Benchmark result:** 51.88 tok/s (up from ~50 tok/s baseline). Modest gain — the rms_norm_linear fusion is correct and saves kernel launches but the decode bottleneck is dominated by other factors in composite replay.

**Key files:**
- `libnd4j/include/ops/declarable/helpers/rms_norm.h` — declaration
- `libnd4j/include/ops/declarable/helpers/cuda/rms_norm.cu` — CUDA impl
- `libnd4j/include/ops/declarable/helpers/cpu/rms_norm.cpp` — CPU impl
- `libnd4j/include/ops/declarable/generic/nn/llm_ops.cpp` — op wiring
- `platform-tests/.../RmsNormLinearTest.java` — tests (9 pass)

**Why:** nsys profiling showed 150 Mean reduction kernels at 454us avg = 1.42ms/step from decomposed rms_norm path. Fusing eliminates the temp allocations and kernel launch overhead.

**How to apply:** Next optimization should target the actual decode bottleneck — likely composite replay overhead, cudaStreamSynchronize waits, or other kernel fusion opportunities beyond rms_norm_linear.
