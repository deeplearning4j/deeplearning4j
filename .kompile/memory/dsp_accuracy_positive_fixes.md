---
name: dsp-accuracy-positive-fixes
description: Positive fixes in HEAD that resolved real bugs — do NOT revert these
type: project
---

## Positive Fixes Already in HEAD — Do NOT Revert (May 2 2026)

These commits fixed real bugs. Reverting any of them re-introduces accuracy issues.

### Correctness Fixes
1. **silu/swish_mul in-place aliasing** (commit 529e26f702) — `llm_ops.cpp:312,791,798,807`
   - Fixed sigmoid(x)^2 bug when output aliases input
   - Guard: `if (output->buffer() == input->buffer())`

2. **MKL SDPA heap overrun** (commit 9bb2680e2b) — `sdpa.cpp`
   - Scratch buffer sized for seqQ*seqKV, not just seqKV
   - Also fixed invalid cblas_sgemm_batch_strided strides

3. **NormalizationFusionOptimizations stripTrivialOps** — `NormalizationFusionOptimizations.java`
   - Restricted to cast/identity only (was stripping through reshape)

4. **FP16 rmsNorm_ float32 accumulator** — `helpers/cpu/rms_norm.cpp`
   - Prevents FP16 overflow during norm computation

5. **FP16 rmsNormLinear_ float32 GEMM** — `helpers/cpu/rms_norm.cpp`
   - Prevents zeros on CPUs without AMX-FP16

6. **Mixed-type gamma support** — `helpers/cpu/rms_norm.cpp`, `helpers/cuda/rms_norm.cu`
   - Cast gamma to input dtype; CUDA uses dual-type template <T,G>

7. **rope FP16 fix** — `llm_ops.cpp`
   - Delegates to helpers::fusedRoPE() instead of hardcoded float

8. **ConstantShapeBuffer alignment check** — `impl/ConstantShapeBuffer.cpp`
   - Detects _shapeInfoBuffer corruption early

### Performance Fixes (also correctness-adjacent)
9. **gather/concat DATADEP trait** — `OpTraitTable.cpp`
   - Forces needsZeroedOutput=true for data-dependent ops (correct behavior)

10. **DspSegmentLifecycle invalidateForRebuild** — `DspSegmentLifecycle.h`
    - Correctly resets executeCount_ and frozenConstantDetection on invalidation

**Why:** These are confirmed good fixes. Any "revert everything since 9bb2680e2b" approach must preserve these.
**How to apply:** When bisecting or reverting, keep these changes intact.
