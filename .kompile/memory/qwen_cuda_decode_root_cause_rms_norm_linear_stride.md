---
name: qwen-cuda-decode-root-cause-rms-norm-linear-stride
description: "[project] ROOT CAUSE FOUND AND PROPERLY FIXED: rmsNormLinearFusedKernel now uses stride-based indexing — no more per-step dup('c')"
type: project
---

## Root Cause: rmsNormLinearFusedKernel Non-Contiguous Weight Bug

**Status**: Root cause found, PROPERLY FIXED with stride-based kernel (May 4 2026)

### Problem

CUDA Qwen3.5 produces correct first token (271 = "France" from prefill) but garbage second token (76828) in STEP 3 single-token decode.

The bug is in `rmsNormLinear()` in `libnd4j/include/ops/declarable/helpers/cuda/rms_norm.cu`.

### Root Cause

The `NormalizationFusionOptimizations.FuseRMSNormLinearPattern` optimizer fuses the final `rms_norm + matmul(lm_head.weight.permute(1,0))` into a single `rms_norm_linear` op.

The weight passed to the fused op is the **permuted** LM head weight:
- Original `lm_head.weight` shape: `[vocab_size V, hidden_size K]`, strides `[K, 1]` (C-order)
- After `.permute(1, 0)`: shape `[K, V]`, strides `[1, K]` (F-order / column-major)

The old kernel indexed weight as `W[k * N + j]` which assumes C-contiguous `[K, N]` with strides `[N, 1]`.

### Fix (PROPER — stride-based kernel, no dup)

The kernel now accepts `wStride0` and `wStride1` parameters and uses stride-based indexing:

```cpp
// Kernel signature now includes strides:
SD_KERNEL void rmsNormLinearFusedKernel(
    ..., const LongType wStride0, const LongType wStride1, ...) {
  // Phase 3: stride-based weight indexing
  acc += normX[k] * static_cast<float>(W[k * wStride0 + j * wStride1]);
}
```

The caller passes actual strides from the NDArray:
```cpp
LongType wStride0 = effWeight->strideAt(0);  // stride along K dimension
LongType wStride1 = effWeight->strideAt(1);  // stride along N dimension
```

This eliminates the old `dup('c')` workaround which copied ~545MB of weight data (vocab_size × hidden_size × 2 bytes for FP16) on every single decode step. For VLM with 152k vocab, that's 152064 × 3584 × 2 = ~1GB per step.

**File**: `libnd4j/include/ops/declarable/helpers/cuda/rms_norm.cu` lines 440-640

### Why This Matters for Performance

The old dup('c') was the single largest per-step overhead for VLM decode:
- 545MB D2D copy per token at ~500 GB/s bandwidth = ~1ms per token
- At 63 tok/s (15.6ms/token), this was ~6% of total time
- Eliminating it helps toward the 100+ tok/s target

### CPU not affected

CPU `rmsNormLinear_` uses `MmulHelper::mmul` which handles non-contiguous strides via cuBLAS/MKL.
