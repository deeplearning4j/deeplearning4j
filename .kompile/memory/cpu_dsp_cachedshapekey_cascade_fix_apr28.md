---
name: cpu-dsp-cachedshapekey-cascade-fix-apr28
description: "cachedShapeKey cascade bug: premature write in computeSegmentShapeKey caused fallback backends to skip compile"
type: project
---

# CPU DSP cachedShapeKey Cascade Bug (fixed 2026-04-28)

## Symptom
`TestQwen35Pipeline` with CUDA_GRAPHS (and all TRITON configs) failed: `seg[116-207] permanently failed — all backends exhausted`. Only SLOT_BY_SLOT passed. The cascade tried OneDNN (failed compile), then OpenVINO — but OpenVINO's `compileSegment` was never called despite the cascade reaching it.

## Root Cause
`computeSegmentShapeKey()` in `NativeDynamicShapePlan_segments.cpp` wrote `seg.exec.cachedShapeKey` eagerly inside itself (two locations: symbolic range path line 186, FNV path line 271) — BEFORE any compile attempt.

When the cascade called `executeSegmentWithSpecificBackend` for OneDNN:
1. `computeSegmentShapeKey()` set `cachedShapeKey = 7995885052539077096`
2. OneDNN compile failed → returned KERNEL_FAILURE (cachedShapeKey NOT cleared)

When cascade retried with OpenVINO:
3. `shapesFrozen_ && cachedShapeKey != 0` → `needsCompile = false`
4. OpenVINO compile skipped entirely
5. Execute looked up cache → nothing there → KERNEL_FAILURE

Secondary bug: validation block (`if (seg.exec.executionCount == 1)`) ran even when compile was skipped, reading stale audit from a previously compiled segment → false "VALIDATION OK: all 8 ops covered".

## Fix (NativeDynamicShapePlan_segments.cpp)
1. Removed `cachedShapeKey` writes from inside `computeSegmentShapeKey()` (both symbolic and FNV paths)
2. Removed premature `cachedShapeKey` write before execute call
3. `cachedShapeKey` now set in exactly ONE place: after `status == OK` in `executeSegmentWithSpecificBackend`
4. Validation block guarded behind `needsCompile` to prevent stale audit false positives

## Result
All 6 configs pass: SLOT_BY_SLOT (6.15 tok/s), CUDA_GRAPHS (11.90 tok/s), TRITON_compileAll_safe (29.50), TRITON_compileAll_best (25.38), TRITON_FULL_fused (41.67), TRITON_compileAll_best_gc (42.37)

## Key Rule
`cachedShapeKey` must ONLY be written after successful compile+execute. Never inside `computeSegmentShapeKey()` or before the compile attempt. The frozen fast-path (`cachedShapeKey != 0 → needsCompile=false`) assumes compile already succeeded for the resolved backend.
