---
name: cuda-frozen-fastpath-staging-root-cause
description: "ROOT CAUSE: CUDA frozen fast-path skips ensureAndSyncStagingBuffers — graph replay sees stale data"
type: project
---

## CUDA VLM Root Cause: Frozen Fast-Path Missing Staging Sync (May 2 2026)

### The Bug
`NativeDynamicShapePlan_cuda.cu` platformTryFrozenFastPath (lines 279-305) does syncToSpecial() (H2D to Java-side NDArray buffer) but NEVER calls ensureAndSyncStagingBuffers(). CUDA graphs are captured using staging buffer device addresses, not raw NDArray addresses. Without staging sync, graph replay reads stale capture-time data.

Composite path in gpubackend.cu (line 928-930) correctly calls ensureAndSyncStagingBuffers().

### Symptom
First 2 tokens from Java prefill correct (216, 49229), all native-loop tokens are 0. DSP validation shows: ref=12015 test=0 at step 2.

### Fix Applied
Added ensureAndSyncStagingBuffers() call after the H2D sync loop, gated by execCtx->stagingBuffersSynced (same pattern as composite path). Updated externalInputs pointer to staged buffers for downstream arg table refresh.

**Why:** The monolithic frozen fast path was a performance optimization that skipped the staging buffer sync step the composite path has.
**How to apply:** File: NativeDynamicShapePlan_cuda.cu, after line 305. Status: APPLIED, needs CUDA rebuild to verify.
