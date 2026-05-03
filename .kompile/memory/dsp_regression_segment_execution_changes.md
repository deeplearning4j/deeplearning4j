---
name: dsp-regression-segment-execution-changes
description: Segment execution and lifecycle changes since 9bb2680e2b — prezero, backend chain, cascade failure, CPU graph replay
type: project
---

## Segment Execution Changes Since 9bb2680e2b (May 2 2026)

### prezeroSegmentOutputs (UNCOMMITTED REGRESSION)
- File: `NativeDynamicShapePlan_segments.cpp:928-933`
- HEAD (committed): unconditional call — correct behavior
- Working tree: guard `if (!(shapesFrozen_ && executeCount_ >= 2))` — REGRESSION
- prezeroSegmentOutputs internal filtering already handles optimization:
  - Skips frozenConstantSlot()
  - Skips !needsZeroedOutput (ops that fully write)
  - Skips isViewCapableOp, isIdentityOp, inPlaceFused, isFusedChainTail
  - Skips FROZEN + isFullyWriting
- The outer guard adds negligible perf benefit but breaks ops needing zeroed output
- GPU backend path (line ~3703) calls prezero unconditionally — inconsistency
- Affected ops: gather, concat (DATADEP trait), argmax/argmin, reshape

### OpenVINO before OneDNN in CPU backend chain (COMMITTED)
- File: `NativeDynamicShapePlan_segments.cpp`
- Changed compilation order: OpenVINO tried BEFORE OneDNN
- If OpenVINO produces different numerical results, accuracy changes
- If OpenVINO compilation fails, falls through silently (see cascade failure below)

### Cascade failure demotion (COMMITTED)
- File: `NativeDynamicShapePlan_segments.cpp`
- Backend compilation failure demoted from throw to silent fallback
- Sets `compilationFailed = true` and continues to next backend
- Risk: segment silently uses fallback backend with different numerical behavior
- No logging of which backend was actually used

### computeSegmentShapeKey caching change (COMMITTED)
- File: `NativeDynamicShapePlan_segments.cpp`
- No longer caches key before successful compile
- Prevents stale shape keys from persisting after failed compiles
- Positive change

### Persistent nativeRangeSegments_ (COMMITTED)
- File: `NativeDynamicShapePlan_segments.cpp`
- CPU graph replay: FunctionalReplayHandle stores lambda captures
- nativeRangeSegments_ map persists across invalidations
- If segment invalidated but map entry persists, replay uses stale lambda captures
- Stale captures may reference old slot arrays, old shapes, old buffer pointers

### DspSegmentLifecycle.h invalidateForRebuild (COMMITTED — POSITIVE)
- Correctly resets executeCount_ and frozenConstantDetection
- State machine: NEEDS_WARMUP → NEEDS_COMPILE → CAPTURE_PENDING → CAPTURED → REPLAYING
- invalidateForRebuild → back to NEEDS_WARMUP

### gpubackend markWarmupDone (UNCOMMITTED FIX)
- File: `NativeDynamicShapePlan_gpubackend.cpp:441`
- Adds `SegmentLifecycle::markWarmupDone(seg.exec)` after shape-change warmup
- Fixes state transition: NEEDS_WARMUP → NEEDS_COMPILE (was stuck in NEEDS_WARMUP)
- Without this: GPU segments never progress past warmup after shape change

**Why:** Segment execution is the scheduling layer that decides what runs, when, and how. Bugs here affect every op in every segment.
**How to apply:** Revert the prezero skip first — it's the simplest fix with the highest impact. Keep the gpubackend markWarmupDone fix.
