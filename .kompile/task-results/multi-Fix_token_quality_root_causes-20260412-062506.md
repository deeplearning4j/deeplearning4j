# Multi-Task Results: Fix token quality root causes

**Subtasks:** 5

---

## fix-dead-slotArrayCache-guards (qwen)

Subagent 'qwen' completed in 360.1s

## Summary
All done. Here's a summary of changes:

**`NativeDynamicShapePlan_slotexec.cpp`** — 5 edits:
1. **Line ~1285** (frozen view-install): Removed `slotArrayCache_ != outputSlots_` guard + stale comments. Deletion now runs when `old != nullptr && old != newView && !isSlotArrayShared(old, si)`.
2. **Line ~1438** (frozen op-exec): Same fix — removed dead guard and nested `if`, flattened into single condition.
3. **Line ~2043** (view-op-install): Same fix — removed dead guard, stale Phase 2 comment.
4. **Line ~2510** (view-producer-detect): Same fix.
5. **Line ~2523** (view-producer-update): Same fix.
6. **Global replace**: All 29 remaining `slotArrayCache_` → `outputSlots_`.

**`NativeDynamicShapePlan.cpp`** — 2 edits:
1. **Line ~1319**: Removed the verification `if (slotArrayCache_ != outputSlots_)` block entirely (it was a sanity check that's now meaningless).
2. **Global replace**: All 11 remaining `slotArrayCache_` → `outputSlots_`, plus updated 2 comments.

**`NativeDynamicShapePlan.h`** — 2 edits:
1. **Line ~1702**: Removed the `#define slotArrayCache_ outputSlots_` macro and its comment block.
2. **Line ~1291**: Cleaned up the member declaration comment (removed the "backward compat" note).
3. Updated 4 comment references to `slotArrayCache_` → `outputSlots_`.

**Full output (1280 chars) written to:** `/home/agibsonccc/Documents/GitHub/deeplearning4j/.kompile/task-results/qwen-20260412-062506.md`
Use the `read` tool to access the full result if needed.

---

## fix-savedKvCurrentPos-sync (qwen)

Subagent 'qwen' completed in 240.4s

## Summary
Summary of changes across 3 files:

1. **`DynamicShapePlanExecutor.java`** — Added `syncSavedKvPosition(long pos)` method that only updates `savedKvCurrentPos` without calling into C++ (avoids double-increment).

2. **`StaticKvCacheDecodeLoop.java`** (standard decode loop, ~line 538) — After `kvCacheManager.setCachePosition(... + 1)`, calls `decoderDspExec.syncSavedKvPosition(...)` to keep the saved position in sync.

3. **`StaticKvCacheDecodeLoop.java`** (speculative decode path, ~line 1180) — After `cachePos += numToScatter`, calls `frozenStep.getDspExec().syncSavedKvPosition(cachePos)`.

4. **`FrozenDecodeStep.java`** — Added `getDspExec()` getter so the speculative decode loop can access the executor for position syncing.

**Full output (735 chars) written to:** `/home/agibsonccc/Documents/GitHub/deeplearning4j/.kompile/task-results/qwen-20260412-062306.md`
Use the `read` tool to access the full result if needed.

---

## fix-cached2-zeroing (qwen)

Subagent 'qwen' completed in 27.6s

## Summary
The fix adds zeroing to the `cached2` reuse path, matching the normal allocation path's pattern exactly:
- Same condition checks (`needsZeroedOutput`, `!isBatchZeroActive()`, `!shouldPreserveWarmupOutputsDuringCapture()`, `!isViewCapableOp`)
- Same batch-zero registration logic
- Same `nullify()` call

**Full output (302 chars) written to:** `/home/agibsonccc/Documents/GitHub/deeplearning4j/.kompile/task-results/qwen-20260412-061933.md`
Use the `read` tool to access the full result if needed.

---

## fix-gather-trait (qwen)

Subagent 'qwen' completed in 326.7s

## Summary
The investigation is complete. Here's the summary:

## Findings

**The `FULLY_WRITING` trait for `gather` is already correct** — no code change was needed beyond adding a clarifying comment.

### Trait composition analysis

```
DATA_MOVE        = DATA_MOVEMENT | FULLY_WRITING
DATA_MOVE_VALDEP = DATA_MOVE | VALUE_DEPENDENT_SHAPE  // includes FULLY_WRITING
GATHER           = DATA_MOVE_VALDEP | GATHER           // includes FULLY_WRITING ✓
```

### Kernel behavior

Both CPU and CUDA gather kernels iterate over **exactly** the output buffer's declared length (`zLen = shape::length(zShapeInfo)` for CUDA, `numIndices * TAD_size` for CPU). Every element of the allocated output buffer is written.

### Frozen replay safety

- In frozen replay, `calculateOutputShape` produces the **same shape** each step (shapes are frozen).
- The output buffer allocated during step 1 is reused with the exact same shape.
- The kernel writes all `zLen` elements → no stale tail data.
- `needsZeroedOutput = !FULLY_WRITING || isDataDep = !true || false = false` → zeroing is correctly skipped.

### Max-allocation edge case

Max-allocation (`outputSlotMaxSizes_`) is only configured for 4D KV cache outputs (names containing "present"+"key"/"value"). Gather outputs are not subject to max-allocation in practice. Even if they were, the kernel would see the oversized `shapeInfo` and write all max-allocated elements (which would cause OOB reads from input — a separate bug, but not the one under investigation here).

### Change made

Added a clarifying comment to `OpTraitTable.cpp` explaining WHY `FULLY_WRITING` is correct for gather, to prevent future confusion.

**Full output (1650 chars) written to:** `/home/agibsonccc/Documents/GitHub/deeplearning4j/.kompile/task-results/qwen-20260412-062433.md`
Use the `read` tool to access the full result if needed.

---

## fix-refreshArgTables-failure (qwen)

Subagent 'qwen' completed in 170.2s

## Summary
The fix is complete. Here's a summary of what was changed:

## Summary

**Problem**: In `NativeDynamicShapePlan_gpubackend.cpp`, when `refreshArgTablesForReplay()` failed, both call sites logged a warning but proceeded with graph replay using stale arg tables — producing garbage output silently.

**Fix applied to both call sites:**

### Call site 1 (~line 2942) — Normal replay path
- Changed from logging a warning and continuing → now falls back to `executeSegmentSlotBySlot()` execution
- Resets the same state flags as the existing fallback pattern at line 2167-2177: `argTableStable`, `batchD2DCount_`, `cachedShapeKey`, `compilationFailed`
- Calls `platformCleanupSegmentForRebuild(seg)` before fallback

### Call site 2 (~line 3233) — COMPOSITE_REPLAY post-gap path
- Same fallback pattern, but also resets capture-specific keys (`capturedInputAddrKey`, `capturedCreateValueKey`) and `executionCount` to match the existing fallback at line 3153-3160
- This is the more critical path since gap ops have already executed — falling back cleanly avoids propagating stale state

Both changes follow the exact same pattern as the 6 existing `executeSegmentSlotBySlot` fallback sites already in the codebase.

**Full output (1210 chars) written to:** `/home/agibsonccc/Documents/GitHub/deeplearning4j/.kompile/task-results/qwen-20260412-062156.md`
Use the `read` tool to access the full result if needed.

---

**Summary:** 5/5 subtasks completed successfully.