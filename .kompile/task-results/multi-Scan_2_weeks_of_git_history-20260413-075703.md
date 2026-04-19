# Multi-Task Results: Scan 2 weeks of git history

**Subtasks:** 3

---

## cuda-graph-replay-history (qwen)

Subagent 'qwen' completed in 143.2s

## Summary
Now I have all the data I need. Let me compile the comprehensive report.

---

## Report: CUDA Graph Replay / Frozen Execution Changes (Last 2 Weeks)

### Chronological Commit List

| Date | Commit | Summary | Stale Data Risk |
|------|--------|---------|-----------------|
| Apr 1 | `2fac9dfdbb` | Guard warmup array deletion during capture + refresh batch-zero pointers | **HIGH (was causing it, now fixed)** |
| Apr 1 | `85560ef1c0` | Revert lombok val→var migration, cleanup | Low |
| Apr 2 | `50909241d6` | Fix CUDA graph replay OOM: weight migration, shape/TAD cache clearing | **MEDIUM (OOM caused stale buffers)** |
| Apr 3 | `3bc6ba64e8` | Pool trimming, weight directReference, frozen slot promotion (executeCount_ >= 0) | **MEDIUM (was promoting too late)** |
| Apr 4 | `1a79df75a1` | Fix SIGSEGV: dereference stream pointer (double pointer fix) | **HIGH (crash, not stale data)** |
| Apr 5 | `05a9fc98ac` | Fix double-free, DataBuffer sharing guard (isSlotArrayShared) | **HIGH (double-free → corruption)** |
| Apr 5 | `bcbb21b0f6` | Remove ungated debug printf from replay hot path (25x slowdown) | Low (perf only) |
| Apr 5 | `7f42a20347` | Eliminate VALUE_DEPENDENT_OPS list: hash small input values in shape key | **MEDIUM (removed guard, later restored)** |
| Apr 5 | `c38210887b` | Restore hasIntLong for shapeStatic + keep value-hashing shape key | **MEDIUM (correctness fix)** |

... (11852 chars total) ...


1. **`634fabf1b7` (tl_graphExecutionActive)** — Directly caused stale pinned host copies to be created during non-capture execution, which replay would read instead of current device data. **This is the most likely root cause of any recent "correct during capture, wrong during replay" issues.**

2. **`af8a42b409` (phase transition address key)** — External inputs in the address key hash prevented phase advancement, meaning the plan never reached the stable replay path. **If you're seeing replay never happen or always falling back to slot-by-slot, this is the cause.**

3. **`2fac9dfdbb` (warmup array deletion)** — Use-after-free of warmup arrays during capture means restored pointers are dangling. **If you see corruption specifically on the first few replay steps after capture, this is the cause.**

4. **`83eccfcc77` (value-dependent ops capturable)** — If value-dependent ops were being captured (during the window between commits `7f42a20347` and `83eccfcc77`), the graph would bake in wrong allocation sizes.

5. **`1919a949ed` (broadcast_to sync)** — Stale host buffer reads in broadcast_to cascade wrong shapes to all downstream ops.

**Full output (11852 chars) written to:** `/home/agibsonccc/Documents/GitHub/deeplearning4j/.kompile/task-results/qwen-20260413-075642.md`
Use the `read` tool to access the full result if needed.

---

## frozen-execution-history (qwen)

Subagent 'qwen' completed in 117.2s

## Summary
Now I have comprehensive data. Let me compile the report.

---

## Report: Frozen Execution, View Recipe, and Segment Execution Changes (Last 2 Weeks)

### Overview

There have been **20 commits** touching the frozen execution path, view recipe handling, and segment execution in the last 2 weeks. These represent massive refactoring of the DSP (Dynamic Shape Plan) system with three major phases:

1. **Phase 1 (Apr 3-5):** CUDA graph replay fixes — dead constants, buffer protection, OOM fixes
2. **Phase 2 (Apr 5-9):** DSP refactoring — macro removal, slot unification, CPU backend support
3. **Phase 3 (Apr 9-13):** Execution correctness — permute view strides, frozen sync skip, phase transition fix

---

### 1. FROZEN EXECUTION PATH — Major Changes

#### `af8a42b409` — **Skip frozen steady-state sync + fix phase transition address key** (Apr 13, most recent)


... (8462 chars total) ...

### 5. STALE DATA RISK IN GRAPH REPLAY

The combination of changes in `af8a42b409` and `7ff51ae45c` creates a potential stale data scenario:

- **Skip sync** → GPU data may not be current
- **Skip nullify** → Output buffers retain previous step's data
- **Actuality counters not ticked** → Downstream ops don't know inputs changed
- **Permutation from device array** → If sync fails, wrong strides

The `DynamicShapePlanExecutor` Java-side stride matching (`createUninitialized(dtype, shape, strides, ordering)` + dup-to-contiguous safety net) is a good defensive measure but can't fix wrong data that was already computed on the C++ side with wrong strides.

**Full output (8462 chars) written to:** `/home/agibsonccc/Documents/GitHub/deeplearning4j/.kompile/task-results/qwen-20260413-075616.md`
Use the `read` tool to access the full result if needed.

---

## java-pipeline-history (qwen)

Subagent 'qwen' completed in 164.1s

## Summary
Now I have a comprehensive picture. Here's the report:

---

## Java-Side Pipeline Changes — Last 2 Weeks

### Commits Overview (chronological, oldest first)

| # | Commit | Summary |
|---|--------|---------|
| 1 | `6e046f6564` | DSP special-case/fallback audit: eliminate cascades, consolidate views, fix Where sync |
| 2 | `50909241d6` | Fix CUDA graph replay OOM: weight migration, shape/TAD cache clearing, pinned host tracking |
| 3 | `3bc6ba64e8` | CUDA graph replay improvements: pool trimming, weight directReference, MHA workspace, frozen slot promotion |
| 4 | `05a9fc98ac` | Fix DSP double-free crash, compilation errors, DataBuffer sharing guard |
| 5 | `83eccfcc77` | Restore outputShapeDependsOnInputValues check in isSlotCapturable |
| 6 | `f391338e05` | Add multi-model pipeline framework for samediff-llm |
| 7 | `21fb3759ad` | Fix LegacyScalarBoolOp for DSP |
| 8 | `2fac9dfdbb` | Fix CUDA graph replay error 700: guard warmup array deletion |
| 9 | `85560ef1c0` | Revert lombok val→var migration, upgrade lombok |
| 10 | `e9af4b6bb3` | Skip empty CUDA/Triton graphs to prevent spurious fingerprint mismatches |

... (10018 chars total) ...


3. **`47a24d3ce4` — 250 MB/step GPU memory leak**: tl_castCache in MmulHelper was growing unboundedly. Fixed by removing push_back calls and adding AutoFreeOnLaunch.

4. **`7ff51ae45c` — `clearNodeOutputsOnly()` vs `clearAllCaches()`**: The decode recompile path now clears only stale outputs, not model constants. `clearAllCaches()` progressively destroyed model weights across recompile cycles.

5. **`7ff51ae45c` — Causal mask per-step update**: In padded decode mode, the causal mask now updates content each step (attend to 0..cachePos, mask the rest). Previously reused stale step-1 mask.

6. **`7ff51ae45c` — KV position sync**: `syncSavedKvPosition()` keeps Java-side position in sync with C++ without double-incrementing, ensuring plan recompilation restores correct position.

7. **Uncommitted — TF32 Triton matmul**: Triton per-element matmul now truncates to TF32 precision to match cuBLAS. This affects numerical output but should improve consistency with cuBLAS baseline.

**Full output (10018 chars) written to:** `/home/agibsonccc/Documents/GitHub/deeplearning4j/.kompile/task-results/qwen-20260413-075703.md`
Use the `read` tool to access the full result if needed.

---

**Summary:** 3/3 subtasks completed successfully.