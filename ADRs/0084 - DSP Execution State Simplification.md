# ADR 0084 - DSP Execution State Simplification

## Status
Accepted

## Context

Following the structural refactoring in ADR 0079 and the replay correctness work in ADR 0082, a second
round of targeted simplification was applied to the DSP execution pipeline. The goals were to reduce
debugging complexity, eliminate invisible coupling, and make phase management bugs detectable at
assertion time rather than as silent downstream corruption.

The following problems were identified:

### Problem 1 — Redundant `ExecutionPhase` Enum

`NativeDynamicShapePlan.h` defined an `ExecutionPhase` enum (5 values: `WARMUP`, `COMPILING`,
`COMPILED`, `REPLAYING`, `SLOT_BY_SLOT`) that was a per-segment concept, set at the same transition
points as `SegmentLifecycleState`. The two enums were parallel representations of the same
information. Maintaining both added translation overhead and created divergence risk: any future
lifecycle state that was added to one enum had to be mirrored in the other. JNI callers required
integer codes from `ExecutionPhase`, further anchoring the redundancy.

### Problem 2 — Dead `SlotState` Values

`NativeSlot::SlotState` had 6 values: `UNINITIALIZED`, `WARMUP`, `SHAPE_CACHED`, `COMPILED`,
`FROZEN`, `FROZEN_CONSTANT`. Searching the entire codebase showed:

- `UNINITIALIZED` appeared only as the default initializer. No code ever branched on it, compared
  against it, or used it as a transition target.
- `COMPILED` had zero references anywhere after the structural refactor.

Both were dead weight. Their presence made the enum larger than necessary, added cognitive load when
reading slot state checks, and gave the impression that 6 states were semantically distinct when only
4 were ever exercised. The Java-side `SlotState.java` mirrored the C++ enum, so it also carried the
dead values.

### Problem 3 — File-Scope TLS as Invisible Coupling

`NativeDynamicShapePlan_gpubackend.cpp` declared two `static thread_local` variables at file scope:

```cpp
static thread_local cudaEvent_t tl_crossStreamEvent = nullptr;
static thread_local std::unordered_map<uintptr_t,uint64_t> tl_prevVariableFingerprints;
```

Thread-local state at file scope is invisible to callers and has thread lifetime, not
execution lifetime. This created two coupling problems:

1. `syncCrossStream()` reached into module-level TLS to obtain the event. Any code path that called
   `syncCrossStream` in a thread where `tl_crossStreamEvent` had not been properly initialized for
   the current execute() invocation would silently use a stale or null event.
2. `tl_prevVariableFingerprints` accumulated entries across execute() calls on the same thread.
   Because its lifetime was the thread rather than the execution, entries from a prior execute()
   (including entries from a different plan if the same thread ran multiple plans) were visible
   during the current execute(), causing false positives in fingerprint comparison.

The monolithic structure of the original file had hidden these issues because in practice a single
thread executed a single plan sequentially. After the structural refactor (ADR 0079), the separation
between per-plan state and per-thread state became visible as a correctness risk.

### Problem 4 — No Structured Execution Trace

When a DSP execution error occurred (wrong results, hang at capture, replay divergence), the
diagnostic path was to enable `DSP_DIAG` at `FULL` level and re-run. That surfaced log lines but no
structured, queryable history of what happened. Reproducing the sequence of events that led to a
bug required either large log buffers or adding ad-hoc diagnostic points. There was no lightweight,
always-on trace that could be inspected post-mortem without a re-run.

### Problem 5 — Lifecycle Transitions Unchecked

`SegmentLifecycleState` transitions in `_gpubackend.cpp` were unchecked: code could call
`markCaptured()` on a segment in `NEEDS_WARMUP` state and the transition would silently succeed.
In practice, the callers were correct, but the lack of precondition assertions meant that bugs like
the latent shape-change recompile path omitting `markWarmupDone` (which had existed in the codebase
before this work) were invisible until they caused downstream execution hangs or corrupted replay.

## Decision

### 1. Remove `ExecutionPhase` Enum, Replace with Display Methods

The `ExecutionPhase` enum was deleted from `NativeDynamicShapePlan.h`. All query logic that
previously depended on it now goes through two methods on `GraphSegmentExec`:

- **`displayPhaseName() const → const char*`** — returns the old enum name as a string for
  diagnostics and logging. Maps `SegmentLifecycleState` values to the prior `ExecutionPhase`
  names so existing log parsers and diagnostic tools continue to work:

  | `SegmentLifecycleState` | `displayPhaseName()` |
  |------------------------|---------------------|
  | `NEEDS_WARMUP`         | `"WARMUP"`           |
  | `NEEDS_COMPILE`        | `"COMPILING"`        |
  | `CAPTURE_PENDING`      | `"COMPILED"`         |
  | `CAPTURED`             | `"COMPILED"`         |
  | `REPLAYING`            | `"REPLAYING"`        |
  | `FAILED`               | `"SLOT_BY_SLOT"`     |
  | `OOM_DEFERRED`         | `"OOM_DEFERRED"`     |

- **`getExecutionPhaseCode() const → int`** — returns the old integer ordinal for JNI
  compatibility. JNI callers (`getPlanSegmentExecutionPhase()` in `NativeOps_dsp.cpp` and
  `NativeOps_dsp.cu`) are updated to call `getExecutionPhaseCode()`.

The `DSP_SET_SEG_PHASE` macro in `DspPhaseUtils.h` is updated to be diagnostic-only (it emits a
`PHASE_TRANSITION` diagnostic event but makes no state assignment). All actual state assignments go
through the `SegmentLifecycle::mark*()` functions, which are the single source of truth.

This change eliminates the two-enum synchronization problem entirely: there is now one state
(`SegmentLifecycleState`) and two derived views of it (string name, integer code) for external
consumers.

### 2. Flatten `SlotState` from 6 to 4 Values

`NativeSlot::SlotState` is reduced to four values:

```cpp
enum class SlotState : uint8_t {
  WARMUP          = 0,  // Initial + invalidation state
  SHAPE_CACHED    = 1,  // Shape cache populated, view status determined
  FROZEN          = 2,  // Shapes frozen, context reuse enabled
  FROZEN_CONSTANT = 3,  // Output never changes, skip execution entirely
};
```

The removed values and their disposition:

- `UNINITIALIZED`: Removed. Default-init of `NativeSlot` now uses `SlotState::WARMUP`, which is
  equivalent in behavior since no code ever tested for `UNINITIALIZED`.
- `COMPILED`: Removed. Zero references existed anywhere in the codebase.

All existing comparison expressions (`>=`, `<`, `>`, `==`) against `WARMUP`, `SHAPE_CACHED`,
`FROZEN`, and `FROZEN_CONSTANT` continue to be correct with the new ordinals.

The convenience accessors on `NativeSlot` that encode the comparison semantics are retained:

```cpp
bool shapeCacheValid()    const { return state_ >= SlotState::SHAPE_CACHED; }
bool frozenContextReady() const { return state_ >= SlotState::FROZEN; }
bool frozenConstantSlot() const { return state_ == SlotState::FROZEN_CONSTANT; }
```

`SlotState.java` is updated to match: the 4-value enum replaces the prior 6-value version. The
`getNativeCode()` / `fromNativeCode()` / `isAtLeast()` accessors are preserved. The Javadoc
describes the same state progression as the C++ comment.

### 3. Move TLS to `PlanExecutionContext`

`tl_crossStreamEvent` and `tl_prevVariableFingerprints` are removed as file-scope `static
thread_local` variables from `_gpubackend.cpp` and become fields on `PlanExecutionContext`:

```cpp
struct PlanExecutionContext {
  cudaEvent_t       crossStreamEvent = nullptr;
  std::unordered_map<uintptr_t, uint64_t> prevVariableFingerprints;
  // ... other per-execute() fields ...
};
```

`PlanExecutionContext` is heap-allocated by `platformBeginExecution()` at the start of each
`execute()` call and freed by `platformEndExecution()` at the end. Its lifetime is exactly one
`execute()` invocation.

`syncCrossStream()` in `_gpubackend.cpp` now accepts `cudaEvent_t syncEvent` as an explicit
parameter rather than reading from TLS:

```cpp
static void syncCrossStream(cudaStream_t dspStream, cudaEvent_t syncEvent,
                             const char* tag, int segIdx = -1);
```

All three call sites (composite replay, raw CUDA graph capture path, frozen fast path) pass
`execCtx->crossStreamEvent` explicitly. The `PlanExecutionContext` pointer is obtained via
`activeExecutionContext()` (a method on `NativeDynamicShapePlan` that returns the context as
`void*`, cast to `PlanExecutionContext*` in `.cpp`/`.cu` files that include
`PlanExecutionContext.h`).

This change ensures that `crossStreamEvent` and `prevVariableFingerprints` are always initialized
at the start of each execute, are scoped to exactly that execute, and are freed when the execute
ends — regardless of which thread runs the execute or whether the same thread has run prior
executes.

`PlanExecutionContext.h` is intentionally not included by `NativeDynamicShapePlan.h`. The main
header keeps `void*` signatures for `platformBeginExecution`/`platformEndExecution`/`activeExecutionContext`.
Only `.cpp`/`.cu` implementation files include `PlanExecutionContext.h`. This prevents the new
header from triggering ccache invalidation across all consumers of the main plan header.

### 4. Add `DspExecutionTrace` Ring Buffer

A new header-only file `DspExecutionTrace.h` introduces a lock-free ring buffer of structured
execution events. The design is:

- **512-event capacity** (power-of-2 for mask-based modulo).
- **48 bytes per event** (`DspTraceEvent`), no heap allocation at record time, no strings.
- **Lock-free**: sequence number is a `std::atomic<uint32_t>`, incremented with
  `memory_order_relaxed`. Writing to the slot is not synchronized with other writers; the ring
  buffer is a best-effort diagnostic tool, not a correctness mechanism. Under concurrent access, a
  recent event may be partially overwritten, which is acceptable.

12 event kinds cover the key decisions in DSP execution:

| Kind                  | Description |
|-----------------------|-------------|
| `SEGMENT_DISPATCH`    | Segment dispatched to a backend |
| `SLOT_WRITTEN`        | Output slot array written/replaced |
| `BUFFER_REPLACED`     | Buffer pointer changed for a slot |
| `EXT_INPUT_SYNCED`    | External input synced to device |
| `GRAPH_CAPTURED`      | CUDA graph capture completed |
| `GRAPH_REPLAYED`      | CUDA graph replay launched |
| `PHASE_TRANSITION`    | Plan or segment phase changed |
| `ARRAY_FREED`         | Plan-owned array freed |
| `SHAPE_KEY_CHANGED`   | Segment shape key changed (triggers rebuild) |
| `CAPTURE_ABORTED`     | CUDA graph capture aborted |
| `ERROR_OCCURRED`      | Error during execution |
| `LIFECYCLE_TRANSITION`| Segment lifecycle state transition |

The trace is wired into `NativeDynamicShapePlan`:

- Constructor allocates a `DspExecutionTrace` and stores it as `trace_`.
- Destructor frees it.
- `getTrace()` and `dumpTrace()` are public methods for test and diagnostic access.

Instrumentation points added:

- `writeOutputSlot` — records `SLOT_WRITTEN` with slot index and buffer address.
- `advancePlanPhase` — records `PHASE_TRANSITION` with old and new phase.
- Segment dispatch (in `_cuda.cu` and `_gpubackend.cpp`) — records `SEGMENT_DISPATCH`.
- Capture begin/end — records `GRAPH_CAPTURED` or `CAPTURE_ABORTED`.
- Replay launch — records `GRAPH_REPLAYED`.
- Error paths — records `ERROR_OCCURRED` with error detail code.

Convenience macros `DSP_TRACE_SEGMENT_DISPATCH`, `DSP_TRACE_SLOT_WRITTEN`, etc. are defined
adjacent to the class. All macros are no-ops when the `trace_` pointer is null.

The ring buffer complements the DSP diagnostic framework (ADR 0078): `DSP_DIAG` events are
category-filtered and go to a separate ring buffer that feeds the printable plan report;
`DspExecutionTrace` events are structured value types that can be inspected programmatically
(e.g., by test code or a crash handler) to reconstruct the last 512 execution decisions without
enabling verbose logging.

### 5. Enforce `SegmentLifecycleState` Transition Preconditions

All `SegmentLifecycle::mark*()` functions in `_gpubackend.cpp` now assert their precondition via
a `SLS_ASSERT_FROM` macro:

```cpp
#define SLS_ASSERT_FROM(exec, expected, targetName)                        \
  if ((exec).lifecycleState != (expected)) {                               \
    THROW_EXCEPTION("SegmentLifecycle::%s called from wrong state %s "    \
                    "(expected %s); use invalidateForRebuild to reset\n",  \
                    (targetName), slsName((exec).lifecycleState),          \
                    slsName(expected));                                     \
  }
```

The macro expands to `((void)0)` in release builds (controlled by `NDEBUG`). In debug builds
(the default for developer and CI builds), any illegal transition throws with a message naming the
transition function, the actual state, and the expected state.

Valid transitions enforced:

| From              | To                 | Function                |
|-------------------|--------------------|-------------------------|
| `NEEDS_WARMUP`    | `NEEDS_COMPILE`    | `markWarmupDone`        |
| `NEEDS_COMPILE`   | `CAPTURE_PENDING`  | `markCompiled`          |
| `CAPTURE_PENDING` | `CAPTURED`         | `markCaptured`          |
| `CAPTURED`        | `REPLAYING`        | `markReplaying`         |
| `CAPTURE_PENDING` | `OOM_DEFERRED`     | `markOomDeferred`       |
| any               | `FAILED`           | `markFailed` (terminal) |
| `FAILED`/`REPLAYING` | `NEEDS_WARMUP`  | `invalidateForRebuild`  |

A latent bug was found and fixed during this work: the shape-change recompile path (triggered when
a segment's shape key changes between execute() calls) was calling `markCompiled()` without first
calling `markWarmupDone()`. The segment was still in `NEEDS_WARMUP` when the compile path ran,
so `SLS_ASSERT_FROM` at the start of `markCompiled` would have thrown. The fix is a
`markWarmupDone()` call inserted before `markCompiled()` in the recompile path.

Additionally, the `markReplaying()` call site is now guarded with:

```cpp
if (seg.exec.lifecycleState == SLS::CAPTURED) {
  SegmentLifecycle::markReplaying(seg.exec);
}
```

This ensures `markReplaying` only fires on the first replay after capture; subsequent replays
leave the segment in `REPLAYING` state (the steady-state). Without this guard, the assertion
would fire on every replay after the first.

## Consequences

- **Single source of truth for segment phase.** `SegmentLifecycleState` is the only per-segment
  phase state. `ExecutionPhase` no longer exists as a separate concern. JNI callers get integer
  codes and string names via derived-view methods that are trivially verifiable against the enum.

- **Slot state enum is correct by inspection.** With only 4 values (the 4 that are actually used),
  reading slot state comparisons is unambiguous. Dead states cannot appear in traces or breakpoints.

- **`crossStreamEvent` lifetime matches execution lifetime.** The event is allocated before the
  execute, valid throughout, and freed after. There is no window where a stale event from a prior
  execute on the same thread is inadvertently used.

- **Variable fingerprints are scoped per execution.** `prevVariableFingerprints` is empty at the
  start of every execute. Cross-execute contamination of fingerprint comparisons is structurally
  impossible.

- **Post-mortem trace without re-run.** When a replay divergence or phase regression occurs, the
  last 512 execution events are available via `plan->dumpTrace()` without enabling verbose
  diagnostic logging. Tests that detect divergence can call `getTrace()` to inspect the exact
  event sequence that led to the error.

- **Illegal transitions throw.** In debug builds, calling `markCompiled()` on a segment in
  `NEEDS_WARMUP` state (or any other invalid transition) throws with a clear message. This surfaces
  phase management bugs at the transition site rather than as mysterious stale-data or hang symptoms
  several steps later.

- **Recompile path bug fixed.** The shape-change recompile path that was skipping `markWarmupDone`
  now correctly advances through `NEEDS_WARMUP → NEEDS_COMPILE` before calling `markCompiled`.
  Without the assertion, this bug was silent; the segment's `lifecycleState` would be inconsistent
  with the actual phase after recompile.

- **No header rebuild cascade.** `PlanExecutionContext.h` is not included by
  `NativeDynamicShapePlan.h`. The `DspExecutionTrace.h` is included only once in the main plan
  header (one new include). All other implementation changes are in `.cpp`/`.cu` files. ccache
  invalidation is minimal.

## Files Added/Modified

### Added Files
- `libnd4j/include/graph/DspExecutionTrace.h` — lock-free ring buffer with 12 event kinds,
  48-byte `DspTraceEvent`, 512-event capacity, convenience macros
- `libnd4j/include/graph/PlanExecutionContext.h` — per-execute() context struct consolidating
  CUDA stream handles, cross-stream event, variable fingerprints, derived booleans, timing,
  phase dispatch flags, sync tracking, `SyncLevel` enum
- `libnd4j/include/graph/CaptureStateGuard.h` — RAII guard for CUDA graph capture scoping

### Modified Files
- `libnd4j/include/graph/NativeDynamicShapePlan.h` — remove `ExecutionPhase` enum; add mapping
  comment; reduce `SlotState` to 4 values; add `DspExecutionTrace*` member, `getTrace()`,
  `dumpTrace()` methods; add `activeExecutionContext()` method returning `void*`
- `libnd4j/include/graph/impl/NativeDynamicShapePlan_gpubackend.cpp` — remove TLS variables
  `tl_crossStreamEvent` and `tl_prevVariableFingerprints`; change `syncCrossStream` signature
  to accept `cudaEvent_t` parameter; update all three call sites to pass
  `execCtx->crossStreamEvent`; add `SLS_ASSERT_FROM` in each `SegmentLifecycle::mark*` function;
  fix recompile path `markWarmupDone` omission; guard `markReplaying` on `CAPTURED` state
- `libnd4j/include/graph/impl/NativeDynamicShapePlan.cpp` — add trace instrumentation at
  `writeOutputSlot`, `advancePlanPhase`; allocate/free trace in constructor/destructor;
  include `PlanExecutionContext.h`; implement `platformBeginExecution`/`platformEndExecution`
- `libnd4j/include/graph/impl/NativeDynamicShapePlan_cuda.cu` — add trace instrumentation at
  segment dispatch and replay sites; include `PlanExecutionContext.h`
- `libnd4j/include/graph/impl/DspPhaseUtils.h` (or `.cpp`) — update `DSP_SET_SEG_PHASE` to
  diagnostic-only (emit trace event, no state assignment)
- `libnd4j/include/graph/impl/NativeOps_dsp.cpp`,
  `libnd4j/include/graph/gpu/NativeOps_dsp.cu` — update `getPlanSegmentExecutionPhase()` JNI
  functions to call `seg.exec.getExecutionPhaseCode()`
- `nd4j/.../execution/SlotState.java` — reduce to 4-value enum (`WARMUP`, `SHAPE_CACHED`,
  `FROZEN`, `FROZEN_CONSTANT`); update Javadoc; retain `getNativeCode()`, `fromNativeCode()`,
  `isAtLeast()` accessors

## References

- ADR 0061 — DynamicShapePlan Execution (baseline mechanism, `SegmentLifecycleState` origin)
- ADR 0078 — DSP Diagnostic Framework Extensions (`DSP_DIAG` categories, ring buffer architecture)
- ADR 0079 — NativeDynamicShapePlan Structural Refactoring (`GraphSegmentExec`, `NativeSlot`
  sub-struct extraction that made this simplification feasible)
- ADR 0082 — CUDA Graph Replay Pointer Stability and Frozen Steady-State (cross-stream event
  infrastructure that `PlanExecutionContext.crossStreamEvent` consolidates)
- ADR 0083 — Thread-Local Cast Cache Leak Prevention (related TLS lifetime issue pattern)
