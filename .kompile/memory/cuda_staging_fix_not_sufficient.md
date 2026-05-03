---
name: cuda-staging-fix-not-sufficient
description: CUDA frozen fast-path staging fix applied but NOT sufficient — still all-zero tokens [216, 49229, 0, 0, ...]
type: project
---

## CUDA Staging Fix — Applied But Not Sufficient (May 2 2026)

**Fix applied:** Added `ensureAndSyncStagingBuffers()` call in `platformTryFrozenFastPath` (NativeDynamicShapePlan_cuda.cu) after H2D sync.

**Result:** Still produces `[216, 49229, 0, 0, 0, ...]` — all-zero tokens after first two. 47.36 tok/s throughput but garbage output.

**Hypothesis:** The staging fix is necessary but not sufficient. Other CUDA-specific bugs from the memory file (causal mask off-by-one in autoregressive_decode.cu, executeCount_ reset in invalidateForRebuild, executeSteadyState gate mismatch) may also be contributing.

**Next steps for CUDA:**
1. Check if the other APPLIED UNVERIFIED fixes from dsp_accuracy_cuda_fix_status.md actually made it into the build
2. The causal mask off-by-one (kvJustWritten = currentPosition - 1) is the highest priority
3. May need debug+verbose trace on CUDA similar to what found the softplus bug on CPU

**Why:** Multiple bugs can stack. The staging fix addresses stale CUDA graph replay data, but if the causal mask is wrong, attention will still produce wrong results.

**How to apply:** Don't treat CUDA as fixed after one fix — verify all applied changes are in the build.
