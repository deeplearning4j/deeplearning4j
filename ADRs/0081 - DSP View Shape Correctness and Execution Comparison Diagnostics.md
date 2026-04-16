# ADR 0081 - DSP View Shape Correctness and Execution Comparison Diagnostics

## Status
Accepted

## Context

The DSP execution path (ADR 0061) builds shape info for view-producing ops (permute, reshape, transpose, squeeze, expand-dims, strided-slice) at plan compile time. The slot execution path uses these precomputed `shapeInfo` buffers to stamp the output slot without rerunning the shape function. This is a significant performance win over the standard op-by-op path but is correctness-critical: any view with wrong strides silently produces wrong data layouts that compound through downstream ops.

Two correctness bugs were observed while investigating why SmolDocling produced degenerate output on the DSP path (stuck argmax, random tokens) while producing the expected "mythic heroes" passage on the standard path:

### Bug 1 — `buildPermutedViewShapeInfo` Ignored ONNX Permutation Input

ONNX `Transpose` stores its permutation in the second input tensor (`input[1]`) as a constant `int64` array. PyTorch-style SameDiff ops store permutation in `iArgs`. The DSP view builder only read `iArgs`. When `iArgs` was empty (which is always the case for ONNX-imported models), the builder fell through to the shape function's default code path, which assigned fresh C-contiguous strides instead of permuted strides.

The result: every `permute`-after-`reshape` chain in the ONNX-imported decoder produced a view with wrong strides. The permuted view was laid out as `[d0*s0 + d1*s1 + d2*s2 + ...]` using C-order strides over the original buffer, so downstream ops read the wrong elements. The error was silent — no shape mismatch, no NaN, just wrong numbers that compounded through 30 transformer layers.

### Bug 2 — Broadcastable Ops Could Emit Non-Contiguous Output Strides

`BroadcastableOp::calculateOutputShape` sometimes emitted output strides that were not contiguous in C or F order (inherited from one of the broadcast inputs). Downstream DSP ops assumed C-contiguous output layout when reading from slot buffers, so a non-contiguous broadcast output produced wrong offset calculations in the next op.

A third, related issue: `reshape('c', ..., false)` (the no-alloc variant) would fall through to an in-place reshape that assigned C-contiguous strides even when the underlying buffer was not contiguous. This produced a "view" whose strides didn't match the data layout — reads through the view returned garbage. The safety check was added to detect the non-contiguous case and fall back to a copy when no-alloc reshape can't be honored.

### Bug 3 — Infrastructure Sub-Bugs Exposed by the Investigation

Tracking down the view stride bugs surfaced several related issues that had to be fixed to verify the diagnosis:

- **`MmulHelper` still used `ews()`.** The matmul fast-path checked `ews() == 1` as a contiguity condition, which is invalid for views (see MEMORY / CLAUDE.md rule on EWS). Replaced with a stride-based contiguity check.
- **`FlashAttention` workspace buffers were not zeroed before use.** The workspace allocation path assumed zeroed memory but ran on top of pool-recycled buffers. Stale data contaminated attention output. Fix: explicit `nullify()` before use.
- **`numOutputs` under-allocation.** DSP was allocating only the "wired" output count (outputs actually consumed by downstream ops) while the shape function expected the full op output count. The under-allocation caused out-of-bounds writes in ops with unused outputs (e.g., `topk` returning indices when only values are wired).
- **`savedKvCurrentPos` not synced on plan recompilation.** The static KV cache's `currentPos` was cached in a frozen slot buffer; plan recompile reset the slot but left the cached value stale. Fix: explicit sync before recompile.
- **Negative permutation index normalization.** ONNX allows negative axis indices (`-1` for last dim); the view builder did not normalize them against rank.
- **Causal mask reuse in padded decode mode.** The causal mask from step 1 was being reused across decode steps even though the mask shape depends on the current decode step. Fix: regenerate per-step.

## Decision

### 1. `buildPermutedViewShapeInfo` Reads Permutation from Second Input

When `iArgs` is empty, the builder now reads the permutation from `inputs[1]` as an `int64` array. The second-input fallback is guarded by a rank check so it only fires when the second input is actually a permutation tensor (1-D, length == rank of input[0]) and not a broadcast operand.

### 2. `BroadcastableOp::calculateOutputShape` Always Produces Contiguous Output

The base class for broadcast-capable ops now stamps C-contiguous strides on the output shape info regardless of input strides. This matches the invariant that DSP slot buffers are contiguous unless explicitly declared as views.

### 3. `reshape('c', ..., false)` Safety Check

When `reshapeNoAlloc` fails (because the source is non-contiguous), the reshape falls back to a `dup()`-and-reshape path that produces a correct contiguous view. The previous behavior of returning a "view" with mismatched strides is eliminated.

### 4. `DynamicShapePlanExecutor` Matches C++ Output Strides

The Java executor now reads the C++ output stride from the plan shape info and, if the Java output array's strides don't match, calls `dup()` to produce a matching contiguous copy. This ensures Java and C++ agree on layout at every slot boundary.

### 5. Diagnostic Infrastructure

To make it possible to diagnose these bugs without adding ad-hoc `printf`, three diagnostic additions were made:

- **`DSP_DIAG` auto-enabled by `isDebugAndVerbose()`**. Previously the DSP diagnostic category system required explicit `-Dnd4j.dsp.diagnostics=FULL` activation. Now the debug-and-verbose flag (`-Dnd4j.verbose=true -Dnd4j.debug=true`) automatically enables the diagnostic framework at `full` level. No environment variables or extra flags needed for typical debugging sessions.
- **Per-slot input/output logging with `syncToHost`**. Each slot execution now logs its inputs and outputs (with `syncToHost` to force device→host copy) under a new `SLOT_IO` category. Enabling this produces a complete trace of every value flowing through the plan — invaluable for diff-ing against the standard path.
- **`PLAN_OUTPUT` diagnostic at the C++→Java boundary**. Every output returned from native DSP execution logs its full `shapeInfo` (rank, shape, strides, data type, offset). This makes view stride bugs immediately visible — the wrong-stride permute output shows up as the first slot with an unexpected stride pattern.

### 6. `SameDiff.compareExecutionPaths()` Differential Harness

A new API method `SameDiff.compareExecutionPaths(placeholders, outputs)` runs the same graph under both the standard op-by-op executor and the DSP executor, captures all intermediate slot values from both sides, and returns a `PathComparisonReport` that lists every slot where the two paths disagree beyond a configurable tolerance.

The report includes:
- Slot index and op name
- First divergent element index
- Maximum absolute difference
- L2 norm of the difference
- Standard path output (truncated)
- DSP path output (truncated)

This is the primary tool for diagnosing DSP-specific correctness regressions. Running it on SmolDocling produced a one-line answer: "slot 127 (permute) diverges starting at element 0 with maxDiff = 2.1e-1" — the permute view bug was identified in minutes.

### 7. Empty-Array Handling for Reductions

`sumNumber`, `maxNumber`, `minNumber`, `meanNumber`, and `prodNumber` previously crashed on empty arrays (zero-element tensors). They now return the neutral element (`0` for sum/mean, `-inf` for max, `+inf` for min, `1` for prod) and emit a diagnostic warning under the `REDUCE` category.

### 8. New Test Classes

Five new test classes exercise the fixed code paths:

- **`TestPermuteViewStrides`** — permute correctness validation against reference layouts.
- **`TestReshapeViewStrides`** — reshape view stride correctness, including non-contiguous source handling.
- **`TestDSPExecutionCorrectness`** — repeat_kv pattern (a common permute-chained pattern in GQA attention) DSP vs standard.
- **`TestAttentionDspVsStandard`** — full attention forward pass, DSP vs standard, layer-by-layer tolerance.
- **`TestModelComponentIsolation`** — per-component (embedding, attention, MLP, layer norm) DSP vs standard validation across whole-model inference.

The `TritonGraphBackendTest` was extended with a 3D GQA test that exercises the head-expansion code path without a direct `past` input (the absence of `past` previously caused a null-pointer path).

## Consequences

- **Correct output on the DSP path.** SmolDocling now produces the "mythic heroes are set apart from their contemporaries" passage and valid doctag output on the DSP execution path, matching the standard path exactly.
- **View bugs are no longer silent.** The `PLAN_OUTPUT` diagnostic exposes shape info at every slot boundary. Adding a new view-producing op now requires that its shape info appear correct in the diagnostic output — there is a ready-made validation harness.
- **Differential testing is the primary bug-finding tool.** `compareExecutionPaths()` is now the recommended first step when any DSP regression is reported. The report narrows the search to a specific slot in O(seconds).
- **ONNX imports now work correctly.** Any ONNX-imported model with `Transpose` ops (which is virtually all of them) now gets correct view strides on the DSP path. Previously these models worked on the standard path and silently corrupted on the DSP path.
- **Empty arrays no longer crash reductions.** Models that emit empty intermediate tensors (e.g., conditional branches that produce empty slices) now execute cleanly through reduction ops instead of crashing at the first reduction.
- **`ews()` usage reduced further.** The `MmulHelper` fast path is now stride-based. The MEMORY / CLAUDE.md rule to replace `ews()` with stride-based checks is now enforced in this file too.
- **Diagnostic code is reusable.** All diagnostic additions plug into the existing `DspDiagnostics` framework (ADR 0078). No ad-hoc `printf` was introduced. New categories integrate automatically with `--diag-*` flags in the benchmark scripts.

## Files Added/Modified

### Modified Files
- `libnd4j/include/graph/impl/NativeDynamicShapePlan_slotexec.cpp` — `buildPermutedViewShapeInfo` second-input fallback
- `libnd4j/include/ops/BroadcastableOp.cpp` — contiguous output stride enforcement
- `libnd4j/include/array/NDArray.hpp` — `reshape('c', ..., false)` safety check
- `libnd4j/include/helpers/MmulHelper.cu` — replace `ews()` with stride-based contiguity check
- `libnd4j/include/ops/declarable/helpers/cuda/flash_attention.cu` — nullify workspace buffers before use
- `libnd4j/include/helpers/ShapeUtils.cpp` — negative permutation index normalization, fusedAttentionCuda bias strides fix
- `libnd4j/include/graph/DspDiagnostics.h` — `SLOT_IO`, `PLAN_OUTPUT`, `REDUCE` category bits
- `nd4j/.../samediff/SameDiff.java` — `compareExecutionPaths()` + `PathComparisonReport`
- `nd4j/.../execution/DynamicShapePlanExecutor.java` — stride matching, cached2 reuse zeroing, constant caching
- `nd4j/.../generation/StaticKvCacheDecodeLoop.java` — causal mask per-step regeneration, `savedKvCurrentPos` sync
- `nd4j/.../internal/InferenceSession.java` — DSP_DIAG-based memory diagnostics, close cast placeholder copies
- `nd4j/.../model/import/onnx/cache/OnnxModelCache.java` — SDZ file locking + retry

### Added Files
- `platform-tests/.../TestPermuteViewStrides.java`
- `platform-tests/.../TestReshapeViewStrides.java`
- `platform-tests/.../TestDSPExecutionCorrectness.java`
- `platform-tests/.../TestAttentionDspVsStandard.java`
- `platform-tests/.../TestModelComponentIsolation.java`

## References

- ADR 0061 — DynamicShapePlan Execution (underlying mechanism)
- ADR 0062 — Java-Side Shape Inference (related shape infrastructure)
- ADR 0078 — DSP Diagnostic Framework Extensions (category system)
- MEMORY / CLAUDE.md — "EWS is invalid" rule (enforced here)
- Commit `7ff51ae45c` — original fix set
- Commit `114cfc16f9` — `MAX_AUTOTUNE` restoration and causal mask follow-up
