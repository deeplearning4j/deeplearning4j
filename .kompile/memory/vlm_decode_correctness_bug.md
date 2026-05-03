---
name: vlm-decode-correctness-bug
description: VLM autoregressive decode produces all zeros after step 2 in OPTIMAL mode — investigation state, fixes tried, next steps
type: project
---

# VLM Decode Correctness Bug — CUDA Graph Capture/Replay

## Symptom
Autoregressive decode produces `[216, 49229, 0, 0, 0, ...]` — 2 correct Java-produced tokens then all zeros from C++ native decode loop.

## Confirmed Scope
- **SLOT_BY_SLOT**: 5/5 correct
- **TRITON_NO_GC** (no graph capture): 5/5 correct  
- **OPTIMAL** (with CUDA graph capture/replay): BROKEN — diverges at step 2
- The capture/replay `.cu` code was NOT changed in the broken commit — the bug is in plan state setup before capture

## Fixes Applied (uncommitted on ag_new_release_updates_2)

### Crash fixes (WORKING):
1. **BFS queue overflow** — `kMaxBfs` 256→4096 in `hasTransitiveDynamicUpstream()` (`NativeDynamicShapePlan_slotexec.cpp` ~line 233)
2. **Lifecycle assertion crash** — Added `SegmentLifecycle::markWarmupDone(seg.exec)` after shape-change warmup (`NativeDynamicShapePlan_gpubackend.cpp` ~line 441)

### Correctness fixes (DID NOT FIX zeros):
3. **Fused-chain aliasing revert** — Reverted intermediate chain slot allocation back to aliasing final output (`NativeDynamicShapePlan_slotexec.cpp` ~lines 2255-2277)
4. **DataBuffer capture-safe H2D guard** — Changed from `isSpecialActual()` to `!isPrimaryActual()` (`DataBuffer.cu` line 914). Confirmed zero CAPTURE_H2D events — fix is working but not the root cause
5. **backfillCachedOutputShapes early exit removal** — Removed `if (state_ >= SHAPE_CACHED) return;` guard (`NativeDynamicShapePlan_slotexec.cpp` line 62)
6. **resetExecuteCount removal from invalidateForRebuild** — Removed `plan->resetExecuteCount()` and `plan->resetFrozenConstantDetection()` from `DspSegmentLifecycle.h`. Rationale: per-segment invalidation was triggering plan-wide phaseWarmup() that destroyed all segments. Fix is architecturally correct but didn't fix the zeros — invalidation path may not be firing during benchmark.

## Files with Uncommitted Changes
1. `libnd4j/include/graph/DspSegmentLifecycle.h` — resetExecuteCount removal
2. `libnd4j/include/array/cuda/DataBuffer.cu` — capture-safe H2D guard
3. `libnd4j/include/graph/impl/NativeDynamicShapePlan_gpubackend.cpp` — markWarmupDone crash fix
4. `libnd4j/include/graph/impl/NativeDynamicShapePlan_slotexec.cpp` — BFS kMaxBfs + fused-chain aliasing + backfill early exit
5. 3 deleted stale test files (TestAutoregressiveDecodeIArgs, TestLLMOpCorrectness, TestLLMBenchmarkSuite)

## Next Investigation Steps
1. **Run with DSP diagnostics** via `run-benchmark.sh --tokens 10` with `-Dnd4j.dsp.diagnostics=EXECUTE,SEGMENT -Dnd4j.dsp.diagnostics.level=full` to see what `executeSteadyState()` and `phaseReplay` actually do step-by-step
2. **`platformTryFrozenFastPath`** in `executeSteadyState()` (`NativeDynamicShapePlan_cuda.cu` line 155) — may not be refreshing ext input staging buffers correctly during graph replay
3. **`phaseShapeInferenceOnly` pre-pass** (`NativeDynamicShapePlan.cpp` line 3396) — allocates zero-initialized arrays and sets SHAPE_CACHED state; these zeros may persist into captured graph
4. **Investigate what changed in the BROKEN commit** (`a26ab20ba9`) that affects plan state before capture — the capture code itself wasn't changed

## Key Architecture Notes
- `executeCount_` is plan-level, `seg.exec.executionCount` is segment-level
- `executeSteadyState()` gates on `executeCount_ < 3` to fall back to full `execute()`
- `isFirstFrozenWarmup = (shapesFrozen && executeCount == 0)` triggers `phaseWarmup()` which is plan-wide destructive
- `platformTryFrozenFastPath` requires `executeCount_ >= 1` AND valid replayHandle
- `platformShouldUseGraph` requires `shapesFrozen_` and checks `isCapturable || (shapesFrozen_ && executeCount_ > 0)`
- `detectFrozenConstants()` gates on `executeCount_ == 1` — only runs once after initial freeze

## Benchmark Command
```bash
cd platform-tests && bash run-benchmark.sh --tokens 250
```
For diagnostics, add to script or env: `-Dnd4j.dsp.diagnostics=EXECUTE,SEGMENT -Dnd4j.dsp.diagnostics.level=full`
