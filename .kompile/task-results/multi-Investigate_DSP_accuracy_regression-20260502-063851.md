# Multi-Task Results: Investigate DSP accuracy regression

**Subtasks:** 6

---

## slotexec-diff-analysis (claude)

Subagent 'claude' completed in 181.9s

## Summary
Now I have a thorough understanding of the changes. Let me compile my findings.

---

## Analysis: Accuracy-Critical Changes in `NativeDynamicShapePlan_slotexec.cpp`

Here are the changes between commit `9bb2680e2b` (last known-good) and the current working tree that could cause output arrays to contain wrong data or zeros. I've ranked them by risk.

---

### 1. **CRITICAL — Fused chain intermediate slots now get SEPARATE arrays instead of aliases** (lines 2254-2277)

**Old code** (line 1878 in old file):
```cpp
writeOutputSlot(chainOutputSlotIdx, output, "fused-chain-member");
```
All intermediate chain slots pointed to the **same** `output` NDArray as the fused chain head.

**New code** (lines 2261-2270):
```cpp

... (8592 chars total) ...


---

### Summary of most likely accuracy regression causes:

1. **#1 (fused chain intermediates)** — Almost certainly produces wrong data. Intermediate chain slots get freshly allocated but never-written arrays. Any op that reads from an intermediate fused chain slot (not just the head) gets zeros/garbage.

2. **#3 + #4 (conditional isDynamicShape + warmup window)** — Could cause slots to freeze with the wrong shape, producing wrong-shaped outputs that are truncated or padded.

3. **#6 + #7 (eager expansion + wrong shape allocation)** — Shape/buffer size mismatch could cause ops to write partial results.

**Full output (8592 chars) written to:** `/home/agibsonccc/Documents/GitHub/deeplearning4j/.kompile/task-results/claude-20260502-063516.md`
Use the `read` tool to access the full result if needed.

---

## segments-prezero-analysis (claude)

Subagent 'claude' completed in 397.6s

## Summary
Key finding: The `gpubackend` direct-exec path at line 3703 calls `prezeroSegmentOutputs` **unconditionally** — no guard. This is the capture/direct-execution path for GPU backends (Triton segments, etc.). The guard only exists in `executeSegmentSlotBySlot` at line 931.

Here is my complete analysis:

---

## Analysis: `prezeroSegmentOutputs` Guard Change

### 1. What `prezeroSegmentOutputs` Does

**Definition**: `_slotexec.cpp:1421` → delegates to `platformPrezeroSegmentOutputs`

**CUDA impl** (`_slotexec_cuda.cu:45-132`): Iterates all slots in the segment and zeros output buffers for qualifying slots. Has **6 filter conditions** (lines 62-69):
- Skip `frozenConstantSlot()` 
- Skip `!needsZeroedOutput` (op fully writes its output)
- Skip `isViewCapableOp`, `isIdentityOp`, `inPlaceFused`, `isFusedChainTail`
- Skip `FROZEN + isFullyWriting` (already known to overwrite completely)

Then batches qualifying buffers into a single `launchBatchMemset` kernel call.


... (5630 chars total) ...

This inconsistency means:
- Triton/GPU backend segments: always get prezero (**correct**)
- CUDA graph replay segments: prezero captured in graph (**correct**)  
- Slot-by-slot segments after step 2: **no prezero at all** (**potentially broken**)

### 7. Recommendation

**The guard should be removed.** The prezero function's internal filtering already handles the optimization correctly — it skips frozen fully-writing slots (the vast majority of decode ops). The outer guard adds negligible performance benefit (the function already returns quickly when all slots are filtered out) while creating correctness risk for the ~3 ops that genuinely need zeroed output buffers.

If the concern is memset overhead for the few remaining `needsZeroedOutput=true` slots, the correct fix is to add `autoregressive_decode` to the `OpTraitTable` with `OP_TRAIT_FULLY_WRITING` (if it truly overwrites all outputs), and/or to remove `DATADEP` from the `isFullyWriting` suppression for `gather`/`concat` (since data-dependence affects shape computation, not buffer write coverage).

**Full output (5630 chars) written to:** `/home/agibsonccc/Documents/GitHub/deeplearning4j/.kompile/task-results/claude-20260502-063851.md`
Use the `read` tool to access the full result if needed.

---

## plan-lifecycle-analysis (claude)

Subagent 'claude' completed in 248.4s

## Summary
Now I have the full picture. Here is my comprehensive analysis:

---

## Root Cause Analysis: All-Zero Logits in Autoregressive Decode

### Summary of Changes Since 9bb2680e2b

The diff introduces three categories of changes to NativeDynamicShapePlan.cpp:

1. **New `phaseShapeInferenceOnly` function** (317 new lines, 3461-3764) — shape pre-pass that allocates zero-initialized arrays
2. **New `markExternalInputVariable` function** (63 new lines, 2632-2694) — marks external inputs as variable and invalidates segments  
3. **New auto shape pre-pass trigger** in `execute()` (lines 1873-1897) — runs `phaseShapeInferenceOnly` before first execution
4. Diagnostic/corruption scanning additions (debug-only, harmless)
5. Changes to `llm_ops.cpp`: rope/silu/swish_mul refactoring

### Finding 1: `markExternalInputVariable` + `invalidateForRebuild` resets executeCount but NOT `compilationDone_`

**Severity: HIGH**
**Location:** `DspSegmentLifecycle.h:147-178`, `NativeDynamicShapePlan.h:1603`

... (6457 chars total) ...


### Recommended Investigation Priority

1. **`silu` / `swish_mul` in-place detection** (`llm_ops.cpp`) — Most likely root cause. These are called on every token in the LLM decode path. If the in-place branch misfires on CUDA, `sigmoid(x)` overwrites `x` before the multiply, producing `sigmoid(x) * sigmoid(x)` instead of `x * sigmoid(x)`. For large values this approaches 1.0, for negative values approaches 0.0. The net effect: output values are crushed toward 0 and 1, which after many layers could produce all-zero logits.

2. **`markExternalInputVariable` not resetting `compilationDone_`** — Could cause stale compiled backends to execute with wrong addresses after segment invalidation.

3. **`rms_norm_linear` reshape with `false`** — Could silently produce wrong results for non-contiguous inputs.

4. **`phaseShapeInferenceOnly` zero-initialized arrays** — Lower priority since the frozen path guard prevents this from running during decode.

**Full output (6457 chars) written to:** `/home/agibsonccc/Documents/GitHub/deeplearning4j/.kompile/task-results/claude-20260502-063622.md`
Use the `read` tool to access the full result if needed.

---

## rms-norm-linear-analysis (claude)

Subagent 'claude' completed in 232.5s

## Summary
Now I have the complete picture. Here's the analysis:

## Investigation Summary

### 1. Full rms_norm_linear op implementation

The op at `llm_ops.cpp:1092-1131`:
- For rank > 2 inputs: reshapes `x` [B,S,K] → `x2d` [M,K] and `output` [B,S,N] → `out2d` [M,N], runs `helpers::rmsNormLinear`, then optionally copies back.
- For rank == 2: passes directly to helper.

### 2. What `reshape(order, shape, false)` means

From `NDArray.hXX:3536-3573`:

- **`copyToNewBuff = true` (default, old code):** Always allocates a new buffer and copies data via `applyTransform(Assign)`. The returned NDArray is independent — writes to it are self-contained.
- **`copyToNewBuff = false` (new code):** Attempts a **zero-copy view** first. Calls `reshapeNoAlloc` to check if contiguous strides can be computed for the new shape. If YES → creates a view sharing the same `DataBuffer`. If NO (non-contiguous strides) → **falls back to copy** (`copyToNewBuff` is flipped to `true`).

### 3. The DSP regression risk — YES, this is dangerous

**The critical bug path:**

... (6471 chars total) ...

| Assign-back needed? | Only if out2d != output buffer AND output is the reshaped copy | Same | Same |

### Conclusion

The new code is **fixing a real bug** (the old code was broken for rank>2 — data wrote to a temporary and was lost). But the fix has a subtle correctness issue in the `!directWrite` assign-back path:

**`out2d->reshape(output->ordering(), *outShape, false)`** — When `out2d` is a temporary (non-view), reshaping it back to [B,S,N] with `false` will succeed as a view (since `out2d` is a freshly-allocated C-contiguous buffer). Then `output->assign(reshaped)` copies from reshaped → output. On CUDA, this requires proper stream synchronization between the `rmsNormLinear` kernel and the assign. The helper already calls `registerSpecialUse`, so `out2d`'s device buffer is marked authoritative, and `assign`'s `prepareUse` should handle the sync correctly.

**Net assessment:** The code should be functionally correct but adds unnecessary overhead. A simpler fix would be:
- If `directWrite` is false, just do `output->assign(out2d)` directly — no need for the intermediate reshape, since `out2d` [M,N] and `output` [B,S,N] have the same total elements and assign handles shape broadcasting/striding.

**Full output (6471 chars) written to:** `/home/agibsonccc/Documents/GitHub/deeplearning4j/.kompile/task-results/claude-20260502-063606.md`
Use the `read` tool to access the full result if needed.

---

## autoregressive-decode-analysis (claude)

Subagent 'claude' completed in 177.4s

## Summary
This is the smoking gun. Line 472:

```cpp
isFirstFrozenWarmup = shapesFrozen && executeCount == 0;
```

Here's the complete chain of events:

## Investigation Summary

### The `markExternalInputVariable` Regression Mechanism

**Timeline of what happens:**

1. **Before decode loop** (lines 443-450): `markExternalInputVariable()` is called for 6-8 external inputs

2. **First call** (e.g., `embeddingsExtIdx`): The `needsFullInvalidation` check at line 2654 evaluates based on whether `effectiveExternals_`, `placeholderStagingBuffers_`, `cachedVariableExtIndices_`, or `variableIndicesCached_` are populated. If the plan has already gone through warmup+capture (which it has — Java prefill runs the plan), **these are populated**, so `needsFullInvalidation = true`.

3. **Full invalidation fires** (lines 2665-2687):
   - Deletes `effectiveExternals_` and `placeholderStagingBuffers_`

... (4907 chars total) ...


### Key Concerns

1. **The `needsFullInvalidation` condition is fragile**: Only the first `markExternalInputVariable` call triggers full invalidation. The remaining 5-7 calls just set flags. But the first call already destroyed all CUDA graphs and reset execution count.

2. **Timing is correct but effect is destructive**: The calls happen before the decode loop (correct timing — after shapes are frozen, before execution). But the invalidation destroys the entire execution pipeline that was built during prefill.

3. **`argTableStable_` reset**: Set to `false` by `invalidateForRebuild`. This means the fast replay path that skips refresh+ext input sync is disabled until arg table stability is re-detected (after ~2-3 warmup executions with matching address keys).

4. **`planPhase_` is NOT demoted**: `markExternalInputVariable` doesn't touch `planPhase_`. It remains at `REPLAYING`. But `executeCount_` is 0. The `executeSteadyState` precondition catches this (`executeCount_ < 3`), but the full `execute()` then sees `isFirstFrozenWarmup = true` and enters warmup. After warmup completes, the plan has to re-capture graphs, re-compile, and re-stabilize — which takes 3+ executions.

**Full output (4907 chars) written to:** `/home/agibsonccc/Documents/GitHub/deeplearning4j/.kompile/task-results/claude-20260502-063511.md`
Use the `read` tool to access the full result if needed.

---

## cpu-dsp-path-analysis (claude)

Subagent 'claude' completed in 124.1s

## Summary
Now I have a comprehensive picture of all the changes. Here's my analysis:

---

## Analysis: DSP Accuracy Regression — Platform-Neutral Root Causes

I've identified **5 suspicious changes** between the known-good commit `9bb2680e2b` and the current code, ranked by likelihood of causing accuracy regression on BOTH CPU and CUDA.

### 1. **CRITICAL — `backfillCachedOutputShapes` early return + shape pre-pass interaction** (slotexec.cpp:62)

**The change:**
```cpp
// NEW — added at line 62:
if (slot.state_ >= NativeSlot::SlotState::SHAPE_CACHED) return;
```

**Why it's dangerous:** The new `phaseShapeInferenceOnly()` pre-pass (NativeDynamicShapePlan.cpp:1883-1897) runs before the first real execution. It sets `slot.state_ = SHAPE_CACHED` and populates `cachedOutputShapes` based on the **prefill** input shapes. Later, during **warmup** (`phaseWarmup`), states are correctly reset to `WARMUP` and `cachedOutputShapes.clear()` is called (line 2985). BUT — the pre-pass also runs when `!shapePrePassDone_ && !shapesFrozen_`, which is TRUE on the first execution of the unfrozen phase (before freeze). The pre-pass populates shape caches from prefill shapes. The early return guard at line 62 then prevents `backfillCachedOutputShapes` from ever updating those shapes if the actual execution produces different output shapes (e.g. different output dtypes after type promotion, or shapes that calculateOutputShape computed differently from the pre-pass). The pre-existing guard `if (!slot.shapeCache.cachedOutputShapes.empty()) return;` was sufficient — the new state-based guard is **redundant AND more aggressive** because `SHAPE_CACHED` persists even after cachedOutputShapes has been repopulated by different values.

... (6957 chars total) ...

| 3 | Persistent `nativeRangeSegments_` enter frozen replay | CPU only | High |
| 4 | `static_cast` UB on CUDA | CUDA only | Medium |
| 5 | OpenVINO cached promoted input | CPU only | Medium |

**The strongest candidates for a both-backend regression are #1 and #2.** The prezero skip is simple and clear-cut: if any op doesn't fully overwrite its output, stale data persists. The shape pre-pass interaction is more subtle but could cause wrong output shapes that cascade through the graph.

To confirm, I'd recommend:
1. **Revert the prezero skip** (restore unconditional `prezeroSegmentOutputs`) and test
2. **Revert the `backfillCachedOutputShapes` early-return** (remove line 62) and test
3. **Set `shapePrePassDone_ = true` in the constructor** to disable the auto pre-pass and test

**Full output (6957 chars) written to:** `/home/agibsonccc/Documents/GitHub/deeplearning4j/.kompile/task-results/claude-20260502-063418.md`
Use the `read` tool to access the full result if needed.

---

**Summary:** 6/6 subtasks completed successfully.