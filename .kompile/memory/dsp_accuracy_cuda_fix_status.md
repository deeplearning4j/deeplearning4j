---
name: dsp-accuracy-cuda-fix-status
description: CUDA VLM accuracy fix status — staging fix not sufficient, still all-zero tokens, needs further investigation May 2 2026
type: project
---

## CUDA VLM (SmolDocling) Accuracy Fix Status (May 2 2026 — Updated)

### END GOAL: run-benchmark.sh --tokens 250 must output text about "mythic heroes"
- Model: SmolDocling VLM
- Input: pathfinder-mythic.pdf
- Test: `cd platform-tests && ./run-benchmark.sh --tokens 250`

### CURRENT STATUS: Still all-zero tokens [216, 49229, 0, 0, 0, ...]
- Output: `<doctag><|endoftext|><|endoftext|>...` — 47.36 tok/s but garbage
- Staging fix applied and verified in build, but NOT sufficient

### FIXES APPLIED AND BUILT (verified in binary)
1. Frozen fast-path staging buffer sync (NativeDynamicShapePlan_cuda.cu)
   - Added ensureAndSyncStagingBuffers() in platformTryFrozenFastPath
   - NOT SUFFICIENT — still all-zero tokens
2. Prezero skip guard removed
3. BFS kMaxBfs=4096
4. Frozen fast-path gate executeCount_ >= 4
5. causal_conv1d kernel flip
6. SameDiff.dup() DSP flag propagation
7. GGMLModelImport forInference()
8. rmsNormLinear reshape fix
9. markExternalInputVariable + gpubackend markWarmupDone

### APPLIED BUT UNVERIFIED (in code, may or may not be in last build)
These were listed in prior memory but need verification they're actually compiled in:

**Bug 1: Causal mask off-by-one in autoregressive_decode.cu:651-668**
- FIX: Use `kvJustWritten = currentPosition - 1` instead of post-incremented currentPosition
- Off-by-one caused attention to see wrong KV positions every decode step
- CRITICAL — likely explains all-zero tokens after prefill

**Bug 3: Frozen fast-path gate mismatch in NativeDynamicShapePlan_slotexec.cpp**
- FIX: Changed `< 3` to `< 4` for executeSteadyState entry
- Must match executeCount_ >= 4 gate

**Bug 4: executeCount_ reset in invalidateForRebuild**
- FIX: invalidateSegmentCaptures method without resetExecuteCount()
- Prevents isFirstFrozenWarmup=true on step 0

**Bug 5: Ungated sd_printf in markExternalInputVariable**
- FIX: Gate behind isVerbose()

### NOT FIXED
**Bug 2: kvScatterBatchedKernel batch dimension**
- Only works for batch=1, not blocking for VLM benchmark

### INVESTIGATION PRIORITIES
1. Verify Bug 1 (causal mask off-by-one) is actually compiled into the binary
2. Run debug+verbose trace on CUDA like we did on CPU to see actual op values
3. Check if CUDA has same softplus alpha=0 bug (CUDA softplus uses different code path)
4. Investigate whether all-zero is from prefill or from first decode step
