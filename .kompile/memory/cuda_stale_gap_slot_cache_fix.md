---
name: cuda-stale-gap-slot-cache-fix
description: "CUDA VLM zeros root cause: markExternalInputVariable invalidates captures but leaves stale activeGapSlotsCached_, compositeReplay uses wrong gap slots"
type: project
---

## Stale Gap Slot Cache Fix (2026-05-02)

**Root cause:** `markExternalInputVariable` (NativeDynamicShapePlan.cpp:2691-2693) calls `invalidateSegmentCaptures` for all segments, which resets per-segment state (executionCount, lifecycleState, CUDA graph handles). BUT it does NOT clear the plan-level `activeGapSlotsCached_` / `cachedActiveGapSlots_`. 

When `compositeReplay` runs (gpubackend.cu:1217), it checks `activeGapSlotsCached_ && shapesFrozen_ && executeCount_ >= 3` — all TRUE because executeCount_ is high from Java prefill/warmup. So it iterates the STALE `cachedActiveGapSlots_` which were built during pre-invalidation executions. These contain gap slot indices from the old composite schedule that don't match the re-captured segment structure, producing wrong outputs → all-zero logits.

**Fix:** Added `activeGapSlotsCached_ = false; cachedActiveGapSlots_.clear();` in `markExternalInputVariable` after the segment invalidation loop (NativeDynamicShapePlan.cpp:2694-2695).

**Confirmed by SLOT_BY_SLOT --no-freeze test:** Even without graph capture, SLOT_BY_SLOT produces `[216, 49229, 49204, 0, 0, ...]` — same zeros pattern. The compositeReplay path still fires via `phaseReplay` → `segDispatchWarmup` even without freeze/capture. The stale gap slot cache is the common path.

**Why SLOT_BY_SLOT also zeros:** Even with `--no-freeze`, the plan goes through phaseReplay which builds composite handles and calls compositeReplay with the stale gap cache.

**How to apply:** Any code that invalidates segment captures MUST also clear the plan-level gap slot cache. Consider adding this to `invalidateSegmentCaptures` itself rather than relying on callers.
