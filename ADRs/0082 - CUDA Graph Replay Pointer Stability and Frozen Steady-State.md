# ADR 0082 - CUDA Graph Replay Pointer Stability and Frozen Steady-State

## Status
Accepted

## Related ADRs
- [ADR 0089](0089%20-%20CUDA%20Graph%20Capture%20and%20Replay.md) — architectural specification this ADR patches

## Context

CUDA graph replay is the top-tier DSP execution mode for autoregressive decode: once a segment's kernel launch sequence is captured into a `cudaGraph_t`, subsequent decode steps invoke `cudaGraphLaunch` instead of re-issuing per-op launches, eliminating ~2 ms of Java-side overhead per step and a similar amount of CUDA launch latency. For SmolDocling on an RTX 4090, replay brings per-step latency down to ~11 ms, approaching the memory-bandwidth bound (~8 ms to load 5.3 GB of weights at ~650 GB/s).

Replay has three hard constraints that interact with the rest of the DSP pipeline in subtle ways:

1. **Captured pointer stability.** `cudaGraphLaunch` re-executes kernels with the same kernel argument bindings used at capture time. If any input/output pointer has moved (because the allocator handed out a different buffer, or because a fresh array replaced the captured slot), the launch will read or write the wrong memory.

2. **Cross-stream ordering.** Java-side `.assign()` and `syncToDevice` operations run on the default `LaunchContext` stream. Graph replay launches on a separate DSP execution stream. Writes on one stream are not visible to reads on another without explicit event-based synchronization. Without it, replay reads stale capture-time data from device memory that was just overwritten on the other stream.

3. **Frozen steady-state guarantees.** Once a plan enters the frozen steady-state (all shapes fixed, all constants identified, all pointers stable), it is safe to skip per-step synchronization, argument table refresh, and per-op `prepareSpecialUse`/`registerSpecialUse` calls. But the transition into frozen state is tricky: one wrong pointer that looks stable (because it happens to reuse the same address) breaks everything.

A sequence of regressions in the SmolDocling benchmark (manifesting as degenerate output — stuck argmax, repeated tokens like "unpaid × 10") made this constraint sprawl visible. The root causes were:

### Regression 1 — `argTableStable` Permanently False

`argTableStable` is a flag the replay path uses to skip arg table refresh and `EXT_INPUT_SYNC` when no pointer in the captured arg table has moved since the last refresh. In the pre-fix state, external input addresses were always changing (Java allocates a fresh placeholder array per decode step), which made `argTableStable` permanently `false`. Fast replay (argument refresh skip) never activated and the benefit of pointer-stability tracking was lost.

### Regression 2 — Phase Transition Address Key Included External Inputs

`computeSegmentShapeKey` hashes all slot addresses into a per-segment key used to detect whether the graph can transition from `SHAPES_FROZEN` to `POINTERS_STABLE`. Including external input addresses in the hash meant the key changed every step (Java allocates fresh placeholder arrays). The phase transition never fired, so the frozen steady-state optimizations never engaged.

### Regression 3 — Frozen Constants Detection Disabled

`detectFrozenConstants()` was disabled (early return at line 684) in the pre-fix state because it produced incorrect results on KV concat ops. KV concat reads from static KV buffers — those slots are Java-managed variables/constants, not produced by any plan op. The dependency propagation couldn't see the upstream and mis-classified KV concat outputs as frozen when they were in fact data-dependent, causing the frozen path to emit stale KV data into the graph.

### Regression 4 — Reused Fixed-Address Embedding Buffers Broke Graph Replay

`StaticKvCacheDecodeLoop` previously allocated fixed-address embedding and `inputId` buffers and used `.assign()` to update them each step. This kept the pointer identity stable across steps (required for graph replay) but created a new problem: `cudaGraphLaunch` replayed the captured embedding-read kernel against the same address, which now had fresh data. But the cross-stream ordering between the `.assign()` write (default stream) and the graph launch (DSP stream) was not enforced. The graph launch read stale capture-time data from the buffer, not the just-assigned fresh data. The output was consistent with the capture-time input, not the current-step input — hence the "stuck argmax" symptom.

### Regression 5 — `tl_graphExecutionActive` Left Set Outside Capture

Related to Bug 1 in ADR 0080, `tl_graphExecutionActive` was being set during all backend segment execution paths. Outside capture, this made `DataBuffer::syncToSpecial()` create pinned host mirrors of device buffers that the next replay step would copy back, silently corrupting replay outputs.

### Regression 6 — Gap Ops Captured Into Graphs

"Gap ops" are small utility ops (identity, nullify, cast) between segment boundaries that were being inadvertently captured into the segment graph. Their inputs are Java-managed and change each step, so capturing them into the replay graph meant the graph held references to capture-time data that was invalid on replay.

## Decision

### 1. `argTableStable` Tracks Internal-Only Pointer Changes

`argTableStable` is now set by a function that compares only internal slot pointers (slots produced by plan ops, not external inputs) against the captured snapshot. External input pointer changes are expected every step and are handled by the explicit arg table refresh + D2D copy path, which runs unconditionally before the replay launch. `argTableStable` only blocks when a plan-internal buffer has moved, which should only happen at phase transitions.

All eight `replayHandle.reset()` invalidation points in `TritonGraphBackend_kernel.cu` set `argTableStable = false` explicitly. The flag is re-set to `true` only at the end of a successful arg table refresh.

### 2. Phase Transition Key Excludes External Inputs

`computeSegmentShapeKey` now iterates only over internal plan-managed buffer addresses when building the phase transition hash. External inputs are explicitly skipped. The hash stabilizes on the second decode step (warmup captures initial addresses; step 2 onward matches), allowing the phase transition `SHAPES_FROZEN → POINTERS_STABLE` to fire on schedule.

### 3. `detectFrozenConstants` Pre-Pass for Java-Managed Slots

Before propagating frozen-ness through the op dependency graph, a pre-pass identifies all output slots that are not produced by any plan op (i.e., Java-managed variables or constants read from the host side) and marks them as `external-dependent`. This taint propagates forward: any op reading from a Java-managed slot is itself marked external-dependent and never frozen.

This restores the correct classification for KV concat ops (which read from static KV buffers) and allows the remaining truly-frozen ops to be optimized. `detectFrozenConstants` is re-enabled without the early-return guard.

### 4. Skip `prepareSpecialUse`/`registerSpecialUse` in Frozen Steady-State

In frozen steady-state (`executeCount_ >= 2` and `planPhase_ == POINTERS_STABLE`), all data is device-resident and the actuality flags on each buffer are correct — there is no need to re-sync on every op. The pre/post sync calls are skipped entirely. This eliminates ~5,486 `syncToDevice()` calls per decode step (2,743 ops × 2 calls), which on SmolDocling corresponds to a measurable steady-state latency reduction.

### 5. Cross-Stream CUDA Event Sync

A CUDA event is now recorded on the default `LaunchContext` stream at the end of each `.assign()` or `syncToDevice` operation that writes to a buffer subsequently read by graph replay. The DSP execution stream waits on that event before launching the graph. All three replay paths are covered:

- Triton composite replay
- Raw CUDA graph replay
- Frozen fast path

This is the minimum synchronization needed to ensure the data written on the default stream is visible to the graph launch on the DSP stream. `cudaDeviceSynchronize()` is **not** used — the event is scoped to the exact stream pair and does not serialize other work.

### 6. Fresh Embedding Buffer Per Step

`StaticKvCacheDecodeLoop` no longer reuses fixed-address embedding or `inputId` buffers. Each decode step allocates fresh buffers. This causes the kernel argument table to see address drift, which correctly invalidates the replay handle and triggers a graph re-evaluation. The slight additional allocation cost (~1 ms per step) is dominated by the correctness win; the fast-replay path still applies to all other stable-address slots.

### 7. `tl_graphExecutionActive` Scoped to Capture

As per ADR 0080, `tl_graphExecutionActive` is now set only inside the capture `begin`/`end` pair. Non-capture execution paths do not touch it. This eliminates stale pinned host mirrors during normal segment execution.

### 8. Gap Ops Excluded from Capture

Ops classified as "gap ops" (pre/post-segment utility ops — identity, nullify, cast) are now excluded from the segment capture boundary and execute fresh on every replay step. Their state is never baked into the replay graph. The exclusion is applied at segment-building time by checking the op category against a gap-op whitelist.

### 9. `batch-zero` and Decode Input Updates in Fast Path

The frozen fast path and raw CUDA graph replay paths previously lacked batch-zero and decode input update steps (these were only present in the Triton composite replay path). Both are now added to all three replay paths. This ensures output buffers are always zeroed before replay (required for correct accumulation in reduction ops) and that per-step decode metadata is fresh.

### 10. No `goto`, No Silent Fallback

All `goto` statements in DSP execution were removed. Error paths that previously fell through to slot-by-slot execution now throw exceptions with a diagnostic category (`FALLBACK`) describing what went wrong. This is a hard anti-workaround stance (see CLAUDE.md rule: "NEVER fall back to slot-by-slot execution"). Silent fallback hides real bugs under a veneer of correctness and destroys the configuration optionality that DSP is designed to support.

### 11. 14 Decode Loop Feature Isolation Tests

A new test class isolates individual `StaticKvCacheDecodeLoop` features one at a time to identify which specific feature triggers a graph replay divergence. Tests 9–22 cover:

- `clearNodeOutputsOnly`
- `reassignDevices`
- `suppressCrossDeviceRouting`
- `ensureExecutionDevice`
- `clearPlaceholders`
- `kvSetCloseableFalse`
- `outputDirect`
- Reusable embedding buffer vs. fresh allocation (this is the test that caught the bug)

Each test runs with exactly one feature enabled or disabled against a baseline and asserts output equivalence against the standard path. The harness is reusable for future divergence investigations.

## Consequences

- **Correct SmolDocling output restored.** The benchmark now produces the expected "mythic heroes" passage with proper doctag structure on all three replay paths (Triton composite, raw CUDA graph, frozen fast path).
- **Fast-replay engages.** `argTableStable` now goes `true` at step 3 (after warmup + one stability-check step) and stays `true` through steady-state. The benchmark shows ~4.71 → ~4.86 tok/s — a modest step but the fast path is now actually active; prior to the fix it was dead code.
- **Frozen steady-state sync skip engages.** After step 2 in frozen mode, per-op sync calls are skipped. On SmolDocling this is ~5,486 calls × negligible-per-call eliminated — measurable at the benchmark level but more importantly correct (previously the presence of these calls was masking the stale-pointer bug).
- **Phase transition fires on schedule.** `SHAPES_FROZEN → POINTERS_STABLE` now triggers at step 3 instead of never. The frozen-path-specific optimizations (constant caching, address key stability) engage for most of the benchmark.
- **Cross-stream sync is explicit.** The default-stream → DSP-stream event ordering is no longer implicit. Any future code that writes to a device buffer from the default stream and expects graph replay to see the write must record an event — the existing event infrastructure can be reused.
- **Fallback is noisy.** Removing `goto` and silent fallback means any error in DSP execution now throws with a FALLBACK category diagnostic. This is the desired behavior per CLAUDE.md: errors must be fixed, not hidden.
- **Feature isolation is the debugging workflow.** The 14 decode loop isolation tests form a template for future divergence investigations. When a new DSP feature causes divergence, the first step is to add it to the isolation test matrix and run the binary-search to the failing feature.
- **Reusable fixed-address buffers are a footgun.** This ADR establishes the rule: fresh buffer per step is the default; fixed-address buffers are allowed only when the write is demonstrably on the same stream as the subsequent graph launch and the cross-stream event is recorded.

## Files Added/Modified

### Modified Files
- `libnd4j/include/graph/gpu/TritonGraphBackend_kernel.cu` — `argTableStable` internal-only tracking, 8 invalidation points
- `libnd4j/include/graph/impl/NativeDynamicShapePlan_gpubackend.cpp` — phase transition address key, frozen sync skip, cross-stream event sync on all three replay paths
- `libnd4j/include/graph/impl/NativeDynamicShapePlan_slotexec.cpp` — `detectFrozenConstants` pre-pass for Java-managed slots
- `libnd4j/include/graph/impl/NativeDynamicShapePlan.cpp` — `computeSegmentShapeKey` external input exclusion
- `libnd4j/include/graph/impl/NativeDynamicShapePlan_cudagraph.cu` — batch-zero + decode input updates in raw graph replay path
- `libnd4j/include/graph/impl/NativeDynamicShapePlan_batchzero.cu` — batch-zero in frozen fast path
- `libnd4j/include/graph/impl/NativeDynamicShapePlan_segments.cpp` — gap op exclusion from capture
- `nd4j/.../generation/StaticKvCacheDecodeLoop.java` — fresh embedding buffer per step, remove reusable fixed-address path
- `libnd4j/include/array/DataBuffer.h`, `cuda/DataBuffer.cu` — `tl_dspExecutionStream` routing (from ADR 0079 follow-up), event record helpers

### Added Files
- `platform-tests/.../CudaGraphReplayDivergenceTest.java` — `testFastReplayCorrectness`, `testFastReplayAcrossPlanLifecycles`
- `platform-tests/.../StaticKvCacheDecodeLoopFeatureIsolationTest.java` — 14 decode loop feature isolation tests (9–22)
- `platform-tests/.../TestDSPFrozenReplayCorrectness.java` — frozen path correctness matrix
- `platform-tests/.../TestCudaGraphReplayRegression.java` — end-to-end replay regression suite

## References

- ADR 0061 — DynamicShapePlan Execution (baseline mechanism)
- ADR 0071 — Triton Graph Backend (capture mechanism)
- ADR 0080 — Triton Fusion Replay Correctness (related `tl_graphExecutionActive` fix)
- ADR 0079 — NativeDynamicShapePlan Structural Refactoring (enables the per-segment state tracking this relies on)
- Commit `af8a42b409` — skip frozen steady-state sync + phase transition address key
- Commit `7d92133d39` — re-enable `detectFrozenConstants` with Java-managed slot handling
- Commit `b997c15894` — cross-stream CUDA event sync
- Commit `316c23fce8` — `argTableStable` internal-only tracking
- Commit `f8e83ff4c9` — fresh embedding buffer, Triton accuracy restoration
- Commit `b64d99b38b` — 14 decode loop feature isolation tests
