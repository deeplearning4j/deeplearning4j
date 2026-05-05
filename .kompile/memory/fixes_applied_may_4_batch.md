---
name: Fixes applied May 4 batch
description: "6 fixes applied: optimizer orphan var, attention perf (nullify+syncToDevice), diagnostic, test workarounds"
type: project
---

## Fixes Applied May 4 2026 (pending build verification)

### Fix 1: Qwen optimizer orphaned variable (CORRECTNESS)
**File**: `nd4j/.../optimize/optimizations/NormalizationFusionOptimizations.java`
**What**: Moved `rmsNormOutVar` cleanup to AFTER matmul removal. Previously, the consumer check ran while matmul was still in the consumer list (first removeOp was refused), leaving the variable orphaned.
**Impact**: Should fix Qwen second token garbage (76828) on CUDA.

### Fix 2: Attention direct-write path (CRITICAL PERF)
**File**: `libnd4j/include/ops/declarable/generic/nn/onnx_multi_head_attention.cpp`
**What**: When no attnBias, reshape output directly to 4D and have FlashAttention write into it. Eliminates workspace allocation + nullify + assign on every attention call.
**Impact**: Removes 22,500 memset+memcpy operations per 250-token 30-layer decode. Expected ~2-4ms/step savings.

### Fix 3: Remove redundant syncToDevice (PERF)
**File**: `libnd4j/include/ops/declarable/generic/nn/onnx_multi_head_attention.cpp`
**What**: Removed 8 syncToDevice() calls after assign() — assign already marks device as actual.
**Impact**: Eliminates unnecessary sync overhead per attention layer.

### Fix 4: Restore COMPILE_VIOLATION diagnostic
**File**: `libnd4j/include/graph/impl/NativeDynamicShapePlan.cpp`
**What**: Made sd_printf unconditional again (was gated behind isVerbose/isDebug).
**Impact**: Compile violations now always visible in benchmarks.

### Fix 5: Fixed stale comment
**File**: `libnd4j/include/graph/impl/NativeDynamicShapePlan_slotexec.cpp:2115`
**What**: Comment said `> 0` but code has `>= 2`. Updated comment to match.

### Fix 6: Removed test workarounds
**File**: `platform-tests/.../TestQwen35Pipeline.java`
**What**: Removed forced SLOT_BY_SLOT execution mode (lines 585, 634) and System.gc() (line 477).
**Impact**: Tests now exercise production CUDA_GRAPHS path. Memory managed via resetSession().

### Build Status
- Java (nd4j-api): BUILT ✓
- CUDA (libnd4j + nd4j-cuda-12.9): BUILDING (background)

### Expected Results After Build
- Qwen CUDA: should output France (token 271) on both first AND second token
- VLM benchmark: should see perf improvement from attention direct-write fix (fewer nodes in graph, less overhead)

**Why:** These fix real bugs and real performance regressions.
**How to apply:** Once build completes, run Qwen test then VLM benchmark. Commit if both pass.
