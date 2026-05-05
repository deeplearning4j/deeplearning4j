---
type: project
title: TRITON merged capture gap fix — tl_mergedCaptureActive
created: 2026-05-05
status: building
---

# TRITON Merged Capture Gap Fix (May 5 2026)

## Root Cause

During merged CUDA graph capture, `tritonOrderedRangeGuard` unconditionally skips gap ops when `streamIsCapturing=true`. This is correct for per-island capture (gaps are replayed fresh each step) but WRONG for merged capture. In merged capture:

1. Gap ops between islands are tagged as non-leader units in the merged group
2. Non-leader units are SKIPPED during replay (`continue` at line 1157)
3. The gap ops are ALSO skipped during capture (by the guard)
4. Net result: gap ops (e.g., gather reading input_ids at slot 104) NEVER execute after warmup

**Evidence:** Token 269 repeating from step 4 onward (first `executeSteadyState` call via `compositeReplay`).

## Fix Applied

Added `tl_mergedCaptureActive` thread-local flag + `tl_mergedCaptureExternals` pointer.

- **Set:** When merged capture begins (`beginCapture` succeeds)
- **Cleared:** At all merged capture exit points (success, failure, tail)
- **Checked:** In `tritonOrderedRangeGuard` lambda — when `streamIsCapturing && tl_mergedCaptureActive`, gap ops EXECUTE on the capture stream (their CUDA kernels get recorded into the merged graph)

Key differences in merged capture gap execution path:
- No `cudaStreamSynchronize` (illegal during capture)
- `tl_graphExecutionActive` stays true (workspace routing)
- `forceSync_ = true` for device coherency
- Uses `tl_mergedCaptureExternals` (staging buffers) instead of raw external arrays

## Files Modified

- `libnd4j/include/graph/impl/NativeDynamicShapePlan_gpubackend.cu`:
  - Lines 444-450: New thread-local declarations
  - Lines 1932-1990: Modified tritonOrderedRangeGuard to check `tl_mergedCaptureActive`
  - Lines 2713-2945: Set/clear flags at all merged capture entry/exit points

## Status

Build in progress. Will test all 6 Qwen configs after build completes.
