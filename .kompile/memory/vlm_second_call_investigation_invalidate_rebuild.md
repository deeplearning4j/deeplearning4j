---
name: vlm-second-call-investigation-invalidate-rebuild
description: invalidateForRebuild + markExternalInputVariable on fresh plan RULED OUT — no-op on new plans
type: project
---

## invalidateForRebuild + markExternalInputVariable — RULED OUT (2026-05-02)

### invalidateForRebuild (DspSegmentLifecycle.h:147-178)

Full reset sequence: cleanupSegmentForRebuild → zero all exec fields → lifecycleState=NEEDS_WARMUP → resetExecuteCount() → resetFrozenConstantDetection(). Both reset functions are single-field assignments with no side effects.

### markExternalInputVariable (NativeDynamicShapePlan.cpp:2632-2694)

On a FRESH plan (never executed), `needsFullInvalidation` is `false` because:
- `effectiveExternals_ == nullptr` (never set)
- `placeholderStagingBuffers_ == nullptr` (never set)
- `cachedVariableExtIndices_.empty()` (empty)
- `variableIndicesCached_ == false` (default)

So `invalidateForRebuild` is NEVER CALLED on a fresh plan. The function only sets `externalInputIsVariable_[extIdx] = true` and clears already-empty caches.

### State Machine (lines 37-62)
```
NEEDS_WARMUP → NEEDS_COMPILE → CAPTURE_PENDING → CAPTURED → REPLAYING
     ↑                                    |                       |
     |                              OOM_DEFERRED               FAILED
     └──────── invalidateForRebuild() from any state ────────────┘
```

**Why:** Investigating whether invalidation on the second plan's fresh state could corrupt lifecycle.
**How to apply:** Fresh plans are safe. invalidateForRebuild only fires when staging buffers/caches exist (i.e., after at least one execution). Not the root cause.
