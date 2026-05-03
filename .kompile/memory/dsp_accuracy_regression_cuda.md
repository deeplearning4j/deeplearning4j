---
name: dsp-accuracy-regression-cuda
description: CUDA-specific DSP accuracy regression issues identified May 2 2026
type: project
---

## CUDA-Specific DSP Accuracy Regression Issues (May 2 2026)

Investigation of commits April 29 - May 2 (since last-good commit `9bb2680e2b`).

### TIER 1 CRITICAL

**1. [DONE] markExternalInputVariable invalidation destroys CUDA graphs (UNCOMMITTED NEW FEATURE)**
- File: `NativeDynamicShapePlan.cpp:2632-2694`, `autoregressive_decode.cu`
- `markExternalInputVariable()` calls `invalidateForRebuild()` which deletes `effectiveExternals_`, `placeholderStagingBuffers_`, resets `executeCount_` to 0
- First call triggers `needsFullInvalidation=true` destroying ALL captured CUDA graphs
- `planPhase_` stays at REPLAYING but `executeCount_=0` → `isFirstFrozenWarmup=true` → re-enters warmup
- Takes 3+ executions to re-capture/re-stabilize — first decode tokens produce wrong results
- **FIXED this session**: gpubackend.cpp `SegmentLifecycle::markWarmupDone(seg.exec)` fix kept in working tree

**2. [NOT A BUG] onnx_mha syncToDevice removed (COMMITTED)**
- File: `onnx_multi_head_attention.cpp`
- **INVESTIGATED this session**: Confirmed the removed syncToDevice calls were genuine no-ops (arrays already device-current at those call sites)
- Do NOT revert — the removal is correct

**3. [DONE] dynamic_cast → static_cast for FunctionalReplayHandle (COMMITTED)**
- File: `NativeDynamicShapePlan_segments.cpp` lines 966 and 1501
- If the object is NOT a FunctionalReplayHandle, static_cast produces UB instead of nullptr
- CPU graph replay uses FunctionalReplayHandle; CUDA uses CudaGraphReplayHandle
- **FIXED this session**: Restored dynamic_cast at both sites to prevent UB and silent corruption

### TIER 2 HIGH

**4. CUDA graph capture with Triton gaps — monolithic capture BANNED**
- Monolithic CUDA graph capture ALWAYS causes accuracy regression with Triton gaps
- Per-segment capture is the correct approach
- If any code path re-introduces monolithic capture, it will break

**5. [DONE] autoregressive_decode REQUIRE_TRUE gated to step < 3 (COMMITTED)**
- File: `autoregressive_decode.cu`
- Validation for plan status, logits checks, KV buffer checks, nextTokenId range only runs first 3 steps
- After step 3, invalid states silently produce wrong results
- **FIXED this session**: Debug printfs gated behind `env_isVerbose()`. Validation gating is a separate ongoing concern.

**6. [FIXED] kvInPlaceWrite dtype mismatch risk**
- File: `onnx_multi_head_attention.cpp` + CUDA helper
- Mixed-type auto-cast at entry casts key/value to query dtype
- If kvInPlaceWrite writes to original (pre-cast) buffer, dtype mismatch corrupts data
- **FIXED this session**: `REQUIRE_TRUE` assertion added in BOTH the CUDA and CPU implementations to catch dtype mismatches at runtime

### TIER 3 MEDIUM

**7. [FIXED] AttentionFusionOptimizations permute absorption**
- File: `AttentionFusionOptimizations.java`
- Permute absorption for rank-4 Q/K/V — incorrect permute detection changed attention input layout
- K transpose absorption had been extended to absorb Permute ops, changing K layout assumptions
- **FIXED this session**: Removed the Permute branch from `extractQKFromMatmul()` at 4 call sites — only true Transpose ops are absorbed now

### ADDITIONAL FIX (New — Not In Original List)

**8. [FIXED] MmulHelper.cu dims.x/dims.y swap at line 977**
- File: `libnd4j/include/helpers/cuda/MmulHelper.cu:977`
- `getMMulDims()` returns `dim3(blocksPerGrid, threadsPerBlock, sharedMem)` — x=blocks, y=threads
- Call was passing `(dims.y, dims.x, dims.z, ...)` — args transposed
- On large matrices where blocksPerGrid > 1024, the swapped value exceeded CUDA's thread-per-block limit → error code 9 (CUDA_ERROR_INVALID_VALUE)
- Became fatal with FP16 `forInference()` weights (larger model weights trigger more large-matrix paths)
- **FIXED this session**: Corrected to `(dims.x, dims.y, dims.z, stream, ...)`

---

**Why:** These are CUDA-specific because they involve CUDA graph capture/replay, device memory sync, and GPU-specific code paths.

**Status (May 2 2026 session end):** Items 1, 3, 5-partial, 6, 7, 8 are FIXED. Item 2 confirmed not a bug. Item 4 (monolithic capture ban) is an ongoing architectural rule — enforce via code review.
