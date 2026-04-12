# Multi-Task Results: Fix decode loop degeneration

**Subtasks:** 3

---

## run-decode-regression-tests (qwen)

Subagent 'qwen' completed in 205.6s

## Summary
All 4 decode regression test suites passed. Here's the summary:

| Test Class | Tests Ran | Passed | Failed | Skipped | Status |
|---|---|---|---|---|---|
| **StaticKvDecodeRegressionTest** | 9 | 9 | 0 | 0 | ✅ PASS |
| **DecodeFeatureInteractionRegressionTest** | 11 | 11 | 0 | 0 | ✅ PASS |
| **DecodeLoopStateTransitionTest** | 18 | 18 | 0 | 0 | ✅ PASS |
| **TestDSPExecutionCorrectness** | 40 | 40 | 0 | 0 | ✅ PASS |
| **Total** | **78** | **78** | **0** | **0** | **All green** |

No errors or failures across any test suite. All builds succeeded cleanly.

**Full output (558 chars) written to:** `/home/agibsonccc/Documents/GitHub/deeplearning4j/.kompile/task-results/qwen-20260412-200528.md`
Use the `read` tool to access the full result if needed.

---

## investigate-phase-transition (qwen)

Subagent 'qwen' completed in 300.5s

## Summary
Now I have all the information I need. Let me compile my findings.

## Findings: DSP Plan Stuck at SHAPES_FROZEN

### 1. `advancePlanPhase()` conditions (NativeDynamicShapePlan.cpp:1573-1622)

The transition **SHAPES_FROZEN → POINTERS_STABLE** requires:
- `shapesFrozen_` is true (line 1583)
- `planPhase_ >= SHAPES_FROZEN` (line 1583)
- **`frozenExecutionCount_ >= 2`** (line 1586)
- **Every segment** must return `true` from `segmentHasStablePointersForPlanPhase()` (line 1589)

The transition **POINTERS_STABLE → REPLAYING** requires:
- Every replay-eligible segment must return `true` from `segmentIsFullyReplayingForPlanPhase()` (line 1612)

### 2. What makes `segmentHasStablePointersForPlanPhase` return false? (lines 137-168)

For each backend type:

- **EMULATED_REPLAY** (line 143): Returns `seg.exec.argTableStable` — this is the most likely culprit for decode loops using emulated replay

... (7473 chars total) ...

| 1 | `NativeDynamicShapePlan.cpp:1586` | `advancePlanPhase()` requires `frozenExecutionCount_ >= 2` — needs 2 full frozen executions before advancing |
| 2 | `NativeDynamicShapePlan.cpp:143` | EMULATED_REPLAY segments block advancement if `argTableStable == false` |
| 3 | `NativeDynamicShapePlan.cpp:150-161` | GPU_COMPILER segments block advancement if `needsReplayInvariantTracking && !argTableStable` |
| 4 | `NativeDynamicShapePlan.cpp:1763-1770` | Baseline key fix IS present for EMULATED_REPLAY — correctly sets `cachedShapeKey` and `capturedInputAddrKey` during warmup |
| 5 | `NativeDynamicShapePlan.cpp:1358` + `:1404` | `frozenExecutionCount_` is incremented at end of `execute()` — correct timing |
| 6 | `NativeDynamicShapePlan_gpubackend.cpp:2605-2608` | `argTableStable` is set true only when extAddrs, createValues, AND shapeKeys all match |
| 7 | `NativeDynamicShapePlan_gpubackend.cpp:2444,2622,2656` | `argTableStable` is reset to false on many error/invalidation paths |
| 8 | `NativeDynamicShapePlan_segments.cpp:1577` | `argTableStable = true` is only set when BOTH `shapeStable && addrStable` — requires 2 consecutive matching keys |

**Most likely root cause**: The decode segments contain GPU_COMPILER segments with `needsReplayInvariantTracking == true` (internal value-shape inputs). These segments require `argTableStable == true` for `segmentHasStablePointersForPlanPhase` to return true. But `argTableStable` starts as `false` and is only set to `true` after 2+ consecutive frozen executions with matching keys. If **any** key changes between frozen executions (e.g., KV cache position changes input addresses), `argTableStable` stays false, the plan never advances past SHAPES_FROZEN, and CUDA graph replay is never activated — resulting in 6.69 tok/s instead of ~90.

**Full output (7473 chars) written to:** `/home/agibsonccc/Documents/GitHub/deeplearning4j/.kompile/task-results/qwen-20260412-200703.md`
Use the `read` tool to access the full result if needed.

---

## investigate-decode-kv-scatter (qwen)

Subagent 'qwen' completed in 255.3s

## Summary
I now have a comprehensive understanding. Let me produce my findings.

---

## Findings: Decode Loop KV Cache Management Investigation

### Summary

The `<doctag>upsupsupsup...` pattern (correct first token, then degenerate repetition) is a classic symptom of the **KV cache not being updated** — the model keeps attending to the same KV state from prefill on every decode step. After tracing the full decode path, I've identified **three specific failure points** that could cause this.

---

### Finding 1: **The causal mask in padded mode is built ONCE and never updated** (MOST LIKELY CAUSE)

**Location:** `DecoderUtils.java`, lines 737-746 (the `isCausalMask` branch in `buildDecoderInputMap`)

In padded mode (`usePadded = true`), the causal mask is allocated once on the first decode step and cached in `reusableInputs`:

```java
if (canReuse && reusableInputs.containsKey(inputName)) {

... (10356 chars total) ...


### Recommended Debug Steps

1. **Add diagnostic print of present KV strides** after the attention op: Check if `present_key` / `present_value` outputs have `stride[3] != 1` (non-contiguous in the last dimension).

2. **Dump the KV scatter source data** before the scatter kernel: In `NativeDynamicShapePlan::scatterKvEntries`, add a diagnostic dump of the first few values from `presentKv` at position `lastPos`.

3. **Dump the static buffer after scatter:** Read back positions `[cachePos-2, cachePos+1]` from the static KV buffer and verify the scattered data is valid.

4. **Test with Java scatter (disable C++ scatter):** Set `-Dnd4j.dsp.nofreeze=true` to force Java-side scatter via the `DecoderUtils.scatterNewKvEntries` path which uses the `KvScatter` op (which may handle strides differently). If this produces correct output, it confirms the C++ scatter is the issue.

**Full output (10356 chars) written to:** `/home/agibsonccc/Documents/GitHub/deeplearning4j/.kompile/task-results/qwen-20260412-200618.md`
Use the `read` tool to access the full result if needed.

---

**Summary:** 3/3 subtasks completed successfully.