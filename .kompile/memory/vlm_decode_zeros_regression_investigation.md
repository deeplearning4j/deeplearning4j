---
name: vlm-decode-zeros-regression-investigation
description: "VLM decode all-zero regression: root cause investigation across commits 9bb2680e2b..a26ab20ba9"
type: project
---

# VLM Decode All-Zero Regression Investigation

## Symptom
- First `generateNative()` call produces CORRECT tokens (12015, 26221, 8639, 13375, 43343)
- Second `generateNative()` call (after session cleanup) produces all-zero logits → token 0 (`<|endoftext|>`) on every step
- Affects ALL configs: SLOT_BY_SLOT, TRITON, OPTIMAL
- Output: `[216, 49229, 0, 0, 0, ...]` — first two tokens from prefill correct, decode loop zeros

## Regression Range
- **Last good commit:** `9bb2680e2b` (fix: MKL SDPA prefill heap overrun)
- **Broken commit:** `a26ab20ba9` (BROKEN: restored working tree snapshot for investigation)
- 79 C++ files changed, ~1021 lines in slotexec alone
- Commits `517d04fe62` → `518e704aee` → `a802bbd5ed` → `a26ab20ba9`

## Confirmed Fixes Applied
1. **prezeroSegmentOutputs unconditional** — FIXED (restored guard `if (!(shapesFrozen_ && executeCount_ >= 2))`)
   - Was zeroing all output slots every step including frozen steady-state
   - Caused all-zero outputs on first plan too
   - File: `NativeDynamicShapePlan_segments.cpp` line ~930

## Ruled Out (did NOT fix the second-plan problem)
1. **DataBuffer.cu syncToSpecial capture path** — `isSpecialActual()` vs `!isPrimaryActual()` — reverted, no effect
2. **DspSegmentLifecycle.h invalidateForRebuild** — resetExecuteCount removal — reverted, no effect
3. **Fused chain aliasing in slotexec** — intermediate slots aliased to final output — reverted, no effect
4. **phaseShapeInferenceOnly auto pre-pass** — disabled, no effect on second plan (first plan works with it enabled)
5. **backfillCachedOutputShapes SHAPE_CACHED fast-exit** — reverted, no effect

## Remaining Suspects (from subagent analysis)
### HIGH priority
- **executeSlot frozen context path changes** — `executeCount_ <= 1` vs `== 0` warmup window, frozen constant handling
- **writeOutputSlot sameBuffer/sameView logic** — new complex check at lines 384-468 may skip data writes
- **setOutputSlotMaxSizes eager expansion** — now immediately expands existing output slot DataBuffers via `db->expand(maxBytes)`
- **Session cleanup corrupting shared state** — `destroySession` + `trimMemoryPool` may affect model weight buffers

### MEDIUM priority
- **nativeRangeSegments_.clear() in resegmentForFreeze** — invalidates persistent CPU backend segments
- **phaseWarmup slot state reset changes** — comment changed but code may differ
- **dynamic_cast → static_cast for FunctionalReplayHandle** — UB if type doesn't match

## Key Observations
- The problem is NOT in the native plan itself (first plan works perfectly)
- The problem is in what happens BETWEEN plans: `clearDynamicShapePlanCache()` → `resetSession()` → `destroySession()` → `trimMemoryPool()`
- Something in the cleanup corrupts state that persists on the SameDiff instance (model weights, variable defs, session buffers)
- The second plan is brand new (fresh C++ NativeDynamicShapePlan), so the corruption must be in the INPUTS to the plan, not the plan code itself

**Why:** Massive investigation/debugging commit introduced ~79 C++ file changes as a "working tree snapshot" on top of the last good commit.
**How to apply:** Focus on what `destroySession` + `trimMemoryPool` do to shared DataBuffers. The weight protection logic (`protectedWeightBuffers_`, `protectedConstantBuffers`, `addFrozenRef`/`removeFrozenRef`) is the most likely area where cleanup of plan 1 corrupts inputs for plan 2.
