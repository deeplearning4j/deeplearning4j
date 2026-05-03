---
name: dsp-regression-autoregressive-decode-changes
description: autoregressive_decode.cu changes since 9bb2680e2b — markExternalInputVariable calls, validation gating, debug printfs
type: project
---

## autoregressive_decode.cu Changes Since 9bb2680e2b (May 2 2026)

File: `libnd4j/include/ops/declarable/helpers/cuda/autoregressive_decode.cu`

### markExternalInputVariable calls (UNCOMMITTED)
- Lines ~443-450: calls markExternalInputVariable() for 8 decode-step-variable inputs
- Purpose: marks external inputs as variable so staging buffers are allocated for CUDA graph replay
- Inputs marked: embeddingsExtIdx, cachePosExtIdx, attentionMaskExtIdx, etc.
- Each call checks needsFullInvalidation — only first call triggers full invalidation
- First call destroys ALL captured CUDA graphs and resets executeCount_ to 0
- Remaining 5-7 calls just set flags (no additional invalidation)

### REQUIRE_TRUE validation gated to step < 3 (COMMITTED)
- Plan status checks: only first 3 steps
- Logits non-zero checks: only first 3 steps
- KV buffer validity checks: only first 3 steps
- nextTokenId range checks: only first 3 steps
- After step 3: ALL validation disabled — invalid states silently produce wrong results
- This was done for performance but means bugs after step 3 are undetectable

### Debug printf tracing (UNCOMMITTED)
- Logits fingerprint: prints sum/max/min of logits array
- Ext input states: prints which inputs are marked as variable
- nextTokenId: prints selected token ID
- These are unconditional printf statements — MUST be removed or gated behind isVerbose/isDebug
- Per feedback rule: gate diagnostics behind isVerbose/isDebug, no unconditional syncToHost

### markExternalInputVariable invalidation chain (UNCOMMITTED)
Timeline on first decode call:
1. markExternalInputVariable() called for first ext input
2. needsFullInvalidation = true (plan has effectiveExternals_ from prefill)
3. Deletes effectiveExternals_ and placeholderStagingBuffers_
4. invalidateForRebuild: resets executeCount_=0, frozenConstantDetection, segment states
5. planPhase_ stays at REPLAYING but executeCount_=0
6. execute() sees isFirstFrozenWarmup=true → enters warmup
7. Takes 3+ executions to re-capture graphs and re-stabilize
8. First few decode tokens may produce wrong results

### argTableStable_ reset (UNCOMMITTED)
- invalidateForRebuild sets argTableStable_ = false
- Fast replay path (skip refresh + ext input sync) disabled until re-detected
- Re-detection requires ~2-3 warmup executions with matching address keys
- During this window: full refresh runs every step (correct but slower)

**Why:** autoregressive_decode is the main decode loop for all LLM generation. Every token passes through this code path.
**How to apply:** Keep markExternalInputVariable calls (needed for CUDA graph replay). Clean up debug printfs. The validation gating at step<3 is a performance optimization that's generally safe but masks bugs.
