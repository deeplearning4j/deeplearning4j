---
name: session-may2-fix-summary
description: Complete summary of all fixes applied in the May 2 2026 DSP accuracy regression session
type: project
---

## May 2 2026 Session — Complete Fix Summary

### Session Overview

This session investigated DSP accuracy regression across commits April 29 – May 2 2026 (since `9bb2680e2b`). 18 distinct fixes were applied across shared, CPU-specific, and CUDA-specific code. 2 suspected regressions were confirmed as not-a-bug.

**Critical discovery**: The assumed "last-good" baseline (`9bb2680e2b`) was NEVER actually correct. When tested properly (without lenient thresholds), it also produced garbage output. The session fixed genuine bugs in the code, but accuracy verification requires a CPU rebuild first.

---

### All 18 Fixes Applied

#### Shared (CPU + CUDA) — 7 fixes

| # | Fix | File(s) | What Was Wrong |
|---|---|---|---|
| 1 | prezero skip reverted | `NativeDynamicShapePlan_segments.cpp:928-933` | Guard skipped prezero for DATADEP ops (gather, concat) — stale output data |
| 2 | BFS kMaxBfs 256→4096 | `NativeDynamicShapePlan_slotexec.cpp:233` | BFS truncated at 256 in 400+ slot VLM models — slots incorrectly frozen |
| 3 | rms_norm_linear reshape | `llm_ops.cpp` | rank>2 rms_norm_linear silently dropped results |
| 4 | SameDiff.dup() DSP flags | `SameDiff.java` | dup() didn't propagate graphExecutionMode and all DSP config flags |
| 5 | backfillCachedOutputShapes guard | `NativeDynamicShapePlan_slotexec.cpp` | Overly aggressive early-return blocked shape correction on prefill→decode transition |
| 6 | GraphOptimizer DCE BFS seed | `GraphOptimizer.java` | BFS only seeded from direct outputs — pruned KV cache update ops (side-effecting) |
| 7 | GenerationPipeline logits auto-discovery | `GenerationPipeline.java` | "logits"→"lm_logits" name change; fallback now scans for any "logit"-containing output |

#### CPU-Specific — 4 fixes

| # | Fix | File(s) | What Was Wrong |
|---|---|---|---|
| 8 | MKL SDPA prefill bias | `sdpa.cpp:756-763` | Causal mask (biasPtr) was NEVER applied in prefill path (seqQ>1) — unmasked attention |
| 9 | fusedRoPECached strides | `helpers/cpu/fused_llm_ops.cpp` | Hardcoded offsets instead of strideAt() — wrong RoPE for non-contiguous cos/sin |
| 10 | invFreq heap allocation | `helpers/cpu/fused_llm_ops.cpp` | `float invFreq[512]` stack buffer — overflow risk for headDim>1024 |
| 11 | nativeRangeSegments stale replay | `NativeDynamicShapePlan_segments.cpp` | clearNativeRangeSegmentsForSlotRange() added to invalidateForRebuild() |

Note on fix #9: Qwen3.5 uses non-cached FusedRoPE, not fusedRoPECached — this fix does not affect Qwen3.5 directly.

#### CUDA-Specific — 6 fixes + 1 printf fix

| # | Fix | File(s) | What Was Wrong |
|---|---|---|---|
| 12 | MmulHelper dims.x/dims.y swap | `MmulHelper.cu:977` | x/y transposed — CUDA error 9 on large matrices when blocksPerGrid>1024 |
| 13 | markExternalInputVariable + markWarmupDone | `NativeDynamicShapePlan.cpp`, `_gpubackend.cpp`, `autoregressive_decode.cu` | First decode call destroyed all CUDA graphs via invalidateForRebuild |
| 14 | static_cast→dynamic_cast FunctionalReplayHandle | `NativeDynamicShapePlan_segments.cpp` (lines 966, 1501) | UB on CUDA (CudaGraphReplayHandle miscast as FunctionalReplayHandle) |
| 15 | kvInPlaceWrite dtype REQUIRE_TRUE | `onnx_multi_head_attention.cpp` + CUDA helper | Added assertion in both CUDA and CPU to catch dtype mismatches |
| 16 | AttentionFusion permute branch removal | `AttentionFusionOptimizations.java` (4 locations) | Permute ops incorrectly absorbed as K transposes — changed attention input layout |
| 17 | autoregressive_decode debug printfs | `autoregressive_decode.cu` | Unconditional printfs on every token — gated behind env_isVerbose() |

---

### 2 Confirmed Not-A-Bug Items

| # | Item | Finding |
|---|---|---|
| N1 | onnx_mha syncToDevice removal | Arrays already device-current at those call sites — removal was correct |
| N2 | OpenVINO before OneDNN ordering | Ordering was identical in `9bb2680e2b` — not a regression |

---

### 2 Positive Fixes (Do NOT Revert)

| # | Fix | Why It's Positive |
|---|---|---|
| P1 | rmsNormLinear_ float32 GEMM accumulator | Prevents zeros on CPUs without AMX-FP16 |
| P2 | GGMLModelImport forInference() weights | Correct FP16 handling for GGML model weights |

---

### Key Technical Findings

1. **Baseline was never correct.** `9bb2680e2b` also produced garbage. Lenient test thresholds masked it.

2. **MmulHelper dims swap was a latent CUDA blocker.** `getMMulDims()` returns `dim3(blocks, threads, shared)` — the call was passing `(dims.y, dims.x, ...)`. Small matrices worked because blocksPerGrid was always ≤ 1024. FP16 forInference() weights pushed matrix sizes over the threshold, triggering CUDA_ERROR_INVALID_VALUE.

3. **DCE was killing KV cache.** The GraphOptimizer DCE pass seeded BFS only from direct graph outputs. KV cache scatter/update ops are side-effecting and don't appear in the output list — they were being pruned on every compile, causing models to lose context between decode tokens.

4. **MKL SDPA prefill was completely unmasked on CPU.** The biasPtr (causal mask) was correctly applied in the decode path (1-token case) but was extracted and then immediately deleted in the prefill path before the computation loop began. Every CPU model prefill has been attending to future tokens.

5. **CPU binary doesn't have the fixes.** C++ CPU fixes (#8-#11) require a full CPU rebuild. The CPU binary in use during this session predated these changes. Accuracy verification on CPU requires rebuilding first.

6. **AttentionFusion permute absorption changed Q/K/V layout.** The fusion pass was absorbing Permute ops (not just Transpose) into the K-transpose step — silently transposing K with the wrong semantics.

---

### Files Modified This Session

**C++ (require rebuild):**
- `libnd4j/include/graph/NativeDynamicShapePlan.h`
- `libnd4j/include/graph/impl/NativeDynamicShapePlan.cpp`
- `libnd4j/include/graph/impl/NativeDynamicShapePlan_gpubackend.cpp`
- `libnd4j/include/graph/impl/NativeDynamicShapePlan_segments.cpp`
- `libnd4j/include/graph/impl/NativeDynamicShapePlan_slotexec.cpp`
- `libnd4j/include/ops/declarable/generic/nn/llm_ops.cpp`
- `libnd4j/include/ops/declarable/helpers/cuda/autoregressive_decode.cu`
- `libnd4j/include/helpers/cuda/MmulHelper.cu` (dims swap)
- `libnd4j/include/helpers/cpu/fused_llm_ops.cpp` (strides + heap)
- `libnd4j/include/ops/declarable/platform/mkldnn/sdpa.cpp` (prefill bias)

**Java (no rebuild needed beyond mvn install):**
- `nd4j/.../SameDiff.java`
- `nd4j/nd4j-ggml/.../GGMLModelImport.java`
- `nd4j/samediff-llm/.../GenerationPipeline.java`
- `nd4j/.../GraphOptimizer.java` (DCE BFS seed)
- `nd4j/.../AttentionFusionOptimizations.java` (permute removal)
