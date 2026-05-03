---
name: dsp-regression-infrastructure-changes
description: DSP infrastructure changes since 9bb2680e2b — phaseShapeInferenceOnly, shapeFunctionOverride, validation gating
type: project
---

## DSP Infrastructure Changes Since 9bb2680e2b (May 2 2026)

### phaseShapeInferenceOnly — NEW FUNCTION (COMMITTED)
- File: `NativeDynamicShapePlan.cpp:3461-3764` (~300 lines)
- Shape pre-pass that allocates zero-initialized arrays for shape inference before real execution
- Triggered in execute() at lines 1873-1897: `if (!shapePrePassDone_ && !shapesFrozen_)`
- Sets `slot.state_ = SHAPE_CACHED` and populates cachedOutputShapes from prefill shapes
- Investigation confirmed: zero-initialized arrays do NOT leak into real execution
- warmup reset clears all SHAPE_CACHED state before kernels execute
- HOWEVER: interacts badly with backfillCachedOutputShapes early-return guard (see below)

### backfillCachedOutputShapes early-return guard (COMMITTED)
- File: `NativeDynamicShapePlan_slotexec.cpp:62`
- New: `if (slot.state_ >= NativeSlot::SlotState::SHAPE_CACHED) return;`
- Pre-existing: `if (!slot.shapeCache.cachedOutputShapes.empty()) return;` — was sufficient
- The new guard is MORE aggressive: SHAPE_CACHED persists even after cachedOutputShapes repopulated
- If actual execution produces different shapes than pre-pass, correction blocked
- Risk: shape mismatch → truncated/padded outputs → wrong data

### shapeFunctionOverride (COMMITTED)
- Files: `NativeDynamicShapePlan_slotexec.cpp` (3 sites: lines 1752, 2670, 4016), `DeclarableOp.cpp:993`
- Set when `executeCount_ >= 3` in frozen decode
- Skips in DeclarableOp::execute(): validateNonEmptyInput, validateArguments, validateDataTypes, prepareOutputs
- Helper dispatch is NOT bypassed (guard was reverted)
- Generally safe: shapes don't change in frozen decode
- Risk: masks bugs during development — invalid inputs silently produce wrong results after step 3

### Input/output validation gating (COMMITTED)
- File: `NativeDynamicShapePlan_slotexec.cpp`
- ALL input pointer validation gated at `executeCount_ < 3 || !shapesFrozen_`
- Post-slot output validation gated at `executeCount_ < 3`
- After step 3: no validation runs at all — null pointers, wrong shapes, stale buffers all undetected

### hasTransitiveDynamicUpstream BFS (COMMITTED BUG, UNCOMMITTED FIX)
- File: `NativeDynamicShapePlan_slotexec.cpp:215-232`
- Iterative BFS with stack-allocated queue to find dynamic upstream dependencies
- COMMITTED: kMaxBfs=256 — silently truncates for VLM models with 400+ slots
- UNCOMMITTED FIX: kMaxBfs=4096
- When truncated: returns false instead of true → slot incorrectly frozen → stale output

### refreshStaleViewWrappersInSegment refactor (COMMITTED)
- File: `NativeDynamicShapePlan_slotexec.cpp`
- Major refactor: now iterates by stepIdx not slotIsViewProducer_
- Changes which views get refreshed and in what order
- If any view refresh is missed, downstream ops read stale data

### safeHasValidShapeInfo() (COMMITTED)
- Exception-guarded wrapper for shape validation
- Catches exceptions from corrupt shape info instead of crashing
- Positive change but masks corruption that should be investigated

### writeOutputSlot alignment check (COMMITTED)
- File: `NativeDynamicShapePlan.cpp`
- Checks _shapeInfoBuffer alignment on output write
- Positive: detects heap corruption early

**Why:** These infrastructure changes control when validation runs, how shapes are cached, and how dynamic upstream detection works. They're the skeleton that all ops execute within.
**How to apply:** The BFS fix is critical and must be kept. The backfillCachedOutputShapes guard should be investigated further — may need to be reverted.
