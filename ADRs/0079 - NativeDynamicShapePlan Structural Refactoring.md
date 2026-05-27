# ADR 0079 - NativeDynamicShapePlan Structural Refactoring

## Status
Accepted

## Context

The `NativeDynamicShapePlan` C++ class (the core of DSP execution, see ADR 0061) had grown organically into an unmanageable structure:

- **1,662-line header** with ~120 members and ~80 methods.
- **12 implementation files totaling ~18K lines**, with a single `NativeDynamicShapePlan_gpubackend.cpp` file of ~5.7K lines covering compile, warmup, replay, frozen handling, and segment dispatch.
- **`NativeSlot` with 150+ fields** — per-op state, fusion metadata, shape caches, control flow info, and legacy op support all intermingled in a single flat struct.
- **`GraphSegment` with 200+ fields** — immutable segment definition (start/end slot, capturability, shape key, contract) mixed with mutable execution state (execution count, replay handle, compilation failure flags).
- **Four parallel state machines** (`PlanPhase`, `ExecutionPhase`, `SlotState`, and a cluster of booleans like `shapesFrozen_`, `isStableReplay`, `isFusedChainHead`) with no single source of truth.
- **Platform dispatch `#ifdef` sprawl** across five files, making any cross-platform change a fragile multi-file edit.

Beyond readability, the structural problems caused concrete bugs:

- **Macro indirection made grepping impossible.** `DSP_SLOT_WRITE(si, value, tag)` and `DSP_NEW_ARRAY(args)` were thin wrappers over `writeOutputSlot(...)` and `new NDArray(...)` that added no abstraction value but hid call sites from code search.
- **Aliased members diverged.** `slotArrayCache_` was defined as a separate pointer but always initialized equal to `outputSlots_`. Code alternately used both names, creating ghost update paths when one was modified without the other.
- **Inline header methods invalidated ccache.** `setShapesFrozen()` was a 130-line method defined inline in the header. Touching it triggered multi-hour rebuilds of every translation unit that included `NativeDynamicShapePlan.h`.
- **Partial Phase 3/4 migration** had left `PlanDefinition` and `ExecutionState` in the tree as placeholders with duplicate state, never plumbed in.

The refactor had to proceed without breaking DSP execution on CPU or CUDA, without regressing the existing Triton composite + CUDA graph replay paths, and without invalidating ccache more than necessary (header changes were held to a minimum).

## Decision

The refactor was sequenced over a baseline audit and five focused commits to make each structural change reviewable in isolation.

### 1. Baseline Audit

The first commit (`eb4a1d035e`) captured the state before refactoring: 110 files changed, ~18K insertions, ~3K deletions, with no behavioral changes. This made subsequent diffs reviewable against a known good reference point.

### 2. `NativeSlot` Sub-Struct Extraction

The 150+ field `NativeSlot` struct was decomposed into eight focused sub-structs, each with a single responsibility:

| Sub-struct | Responsibility |
|------------|----------------|
| `SlotIdent` | Op identity: op hash, op pointer, op name |
| `SlotWiring` | Input/output wiring: `numInputs`, `inputSourceIndices`, output count |
| `SlotArgs` | Frozen op arguments: `iArgs`, `tArgs`, `bArgs`, `dArgs`, `sArgs` |
| `SlotFlags` | Execution flags: data-dependence, fusion head, closable, etc. |
| `FusedChain` | Fused elementwise chain metadata |
| `ControlFlowInfo` | Control flow type, branch targets |
| `LegacyOpInfo` | Legacy op type, legacy op number |
| `ShapeCache` | Per-slot shape cache and static analysis (`cachedShapeKey`) |

Access patterns changed mechanically: `slot.opHash` → `slot.ident.opHash`, `slot.iArgs` → `slot.args.iArgs`, `slot.isDataDependent` → `slot.flags.isDataDependent`, and so on. Thirty-two files were updated with ~2,400 field reference changes. Zero remaining old-style `NativeSlot` field references in any `.cpp`/`.cu` file — the grep check was used as the acceptance gate.

### 3. `GraphSegment` Split into Def + Exec

`GraphSegment` was split into two composed structs reflecting the immutable/mutable boundary:

- **`GraphSegmentDef`** (immutable after plan compile): `startSlot`, `endSlot`, `isCapturable`, `hasValueDepOps`, `shapeKey`, `backendOverride`, `selectedBackend`, `contract`.
- **`GraphSegmentExec`** (mutable during warmup/replay): `executionCount`, `compilationFailed`, `replayHandle`, `isStableReplay`, per-replay counters.

`GraphSegment` now composes both: `seg.def.startSlot`, `seg.exec.executionCount`. This separation makes it trivial to identify which fields can be safely cached, hashed, or serialized (the `def` half) versus which require live mutation guards (the `exec` half). Thirty-plus files were updated with field path changes.

### 4. Unify `outputSlots_`/`slotArrayCache_`

`slotArrayCache_` was removed as a distinct member. Since it was already aliased to `outputSlots_` at all initialization sites, merging them closed the ghost-update path. A `#define slotArrayCache_ outputSlots_` was left in place as a backward-compat bridge for external code that might still reference the old name; new code uses `outputSlots_` directly.

### 5. Remove `DSP_SLOT_WRITE` / `DSP_NEW_ARRAY` Macros

All `DSP_SLOT_WRITE(si, value, tag)` call sites were rewritten to `writeOutputSlot(si, value, tag)`, and all `DSP_NEW_ARRAY(args)` call sites to `new NDArray(args)`. The macros were deleted from headers. The refactor exposed real call sites to grep and IDE symbol navigation; no behavior changes.

### 6. Extract `setShapesFrozen()` to `.cpp`

The 130-line inline `setShapesFrozen()` method was moved from `NativeDynamicShapePlan.h` to `NativeDynamicShapePlan.cpp`. The header now only declares the method. This isolates the freeze logic from ccache invalidation on every consumer of the header and makes breakpoints/symbolication work correctly during debugging.

### 7. Post-Refactor Stabilization (commit `22432f0e49`)

The refactor exposed several latent bugs and new compilation errors that were fixed in a follow-up commit:

- **`DspVerifyUtils.h`**: `reduceNumber()` returns `NDArray*`, fix pointer dereference and add `delete`.
- **`TritonIRBuilder_sections.cpp`**: `epilogueOps` was replaced by `matmulEpilogueCount` during the refactor — fix references.
- **`NativeDynamicShapePlan_segments.cpp`**: `bufferOffset()` replaced with the correct `offset()` `NDArray` method.
- **`NativeDynamicShapePlan_gpubackend.cpp`**: re-declare `isStableReplay` after the refactor removed its original declaration site.
- **Execution hang at warmup**: `phaseWarmup` now sets `segment.exec.executionCount = 1` after warmup so graph capture can proceed — otherwise the segment was stuck at count 0 and capture never triggered.
- **`computeSegmentShapeKey` hang**: add a central frozen cache at function entry (return cached key immediately when `shapesFrozen_ && cachedShapeKey != 0`). Replace per-element `arr->e<T>()` in the shape key hash with direct host buffer `memcpy`. Remove the call from `phaseWarmup` entirely (it was extremely expensive and unnecessary there).
- **Mode preservation**: `StaticKvCacheDecodeLoop` previously hardcoded `MAX_AUTOTUNE` on KV recompile, overriding user-selected `SLOT_BY_SLOT` to `TRITON`. Fix by preserving execution mode across KV recompile. `DynamicShapePlanExecutor` reuses `configuredGraphExecutionMode` on plan recompile.
- **Constant protection**: `InferenceSession` was force-closing constant `DataBuffer`s during session cleanup, destroying model constants needed by DSP execution. Remove the force-close; add `planExecutorClosed` variable declaration for proper lifecycle tracking. `DynamicShapePlanExecutor` now caches 973 small constant values at compile time and calls `syncToDevice` for frozen placeholder inputs.

### 8. In-Flight Phase 2 (uncommitted)

Phase 2 is in progress in the working tree: further simplification of `NativeDynamicShapePlan_gpubackend.cpp` (~2.9K lines removed) and introduction of two new small utility headers — `DspPhaseUtils.h` and `DspVerifyUtils.h` — to isolate phase-transition logic and golden-comparison helpers from the monolith. This is tracked as a separate ADR target once stabilized.

### 9. Shape-Keyed Plan Cache (April 2026)

The structural refactor exposed a second class of bug that the monolith hid: the prior "one plan per `SameDiff`" lifecycle assumed slots were effectively mutable — the same `NativeDynamicShapePlan` was reused across calls with different placeholder shapes, and each call rewrote the affected slots' NDArrays to match the new shape. Once slots became well-defined structs (step 2 above) and `SlotBufferOwnership` gained explicit ownership tracking, the mutation path was visibly incorrect: a slot's array held references from prior dispatch that became dangling after a reshape.

The follow-on work implements a shape-keyed `NativePlanCache` (C++-owned) that turns slot mutation into a compile-time error. The full design is documented in ADR 0061's *Shape-Keyed Plan Cache and Deferred Dispatch* section. The refactor-relevant changes are:

- **`NativeDynamicShapePlan_slotexec.cpp`** — four sites that previously rewrote a slot's `NDArray*` on shape mismatch now `THROW_EXCEPTION(...)`. Slot immutability is a post-condition of the refactor, not a pre-condition of it.
- **New files** — `libnd4j/include/graph/NativePlanCache.h` and `impl/NativePlanCache.cpp` (LRU cache over `std::unique_ptr<NativeDynamicShapePlan>`), keyed on `(outputSetHash, std::vector<sd::LongType*>)` where the pointers come from `ConstantShapeHelper`.
- **`NativeOps_dsp.{cpp,cu}`** — new JNI entrypoints `createNativePlanCache`, `freeNativePlanCache`, `clearNativePlanCacheHandle`, `dispatchNativePlan`. `dispatchNativePlan` is the replacement for the old compile-time native bind; it both installs and looks up plans keyed by current shape signature.
- **`DynamicShapePlanExecutor.java`** — `compileNativePlan(...)` no longer calls a native compile; it serializes bytes, sorts outputs, captures placeholder keys, and stashes per-handle settings. The new `redispatchForCurrentShapes(...)` runs per execute, calls `dispatchNativePlan`, and applies per-handle settings lazily via `configuredHandleAddresses: Set<Long>`.
- **JavaCPP signature parity** — `NativeOps.java`'s default `dispatchNativePlan` stub must match the JavaCPP-generated descriptor exactly (`sd::LongType` → `long`; pointer-of-pointers → `Pointer`). A mismatch resolves to the throwing default. See ADR 0061 for the confirmation procedure via `javap -p`.
- **`DspConfig`** — two new properties (`planCacheBudgetFraction`, `planCacheMaxPlans`) bound the cache. Reading from `Environment` → `DspConfig` keeps the configuration surface consistent with other DSP tunables.

This work is the lifecycle counterpart to the structural refactor: step 2 made slots struct-typed so their identity was crisp; step 9 enforces that the identity, once established, is immutable for the plan's lifetime.

### 10. Disk Plan Persistence (May 2026)

Building on the shape-keyed plan cache (step 9), serialized plan bytes are now persisted to disk at `~/.kompile/cache/dsp/dsp_plan_cache/`, keyed by FNV-1a hash of the `DynamicShapePlan.serialize()` output. This eliminates plan recompilation across JVM restarts — the exact serialized bytes that `fromSerializedPlan()` consumes are loaded from disk instead of recomputed from the Java DAG.

The disk cache mirrors the Triton kernel cache architecture: atomic writes (temp file + rename), override directory for pre-seeded deployments, `.meta` sidecar for version-based invalidation, and model identity index files for cross-JVM lookup without plan recompilation.

New `DspConfig` fields (`planCacheDiskEnabled`, `planCacheDiskDir`, `planCacheDiskForceRecompile`, `planCacheOverrideDir`) follow the same `initFromEnvironment()` pattern established in step 9 for the in-memory plan cache. Full design in ADR 0061's *Disk Plan Persistence* section.

Key file: `nd4j/.../samediff/execution/DspPlanDiskCache.java`.

## Consequences

- **Grep-ability restored.** Removing macro indirection and ghost aliases means `grep writeOutputSlot` / `grep outputSlots_` now return authoritative call-site lists. The acceptance gate for each refactor step was "zero remaining old-style field references," enforced by grep.
- **Header stability.** `setShapesFrozen()` is no longer in the header; header touches that used to trigger 30+ minute rebuilds are now isolated to the `.cpp`. Further inline method extractions are the default going forward.
- **Reviewable diffs.** Each refactor commit is one structural change. `NativeSlot` extraction (2,400 field references) was mechanical enough to verify by grep; the `GraphSegment` split (30+ files) was similarly straightforward. No mixed behavioral + structural changes made it into the refactor commits — all behavioral fixes are in commit `22432f0e49` and clearly labeled.
- **Latent bugs surfaced.** The hang at `computeSegmentShapeKey`, the execution count off-by-one at warmup, and the `InferenceSession` constant-close bug were all hidden by the monolithic structure. The refactor made them visible by forcing each code path through well-defined struct boundaries.
- **No performance regression.** DSP execution paths — Triton composite, raw CUDA graph, frozen fast path — all produce the same outputs and latency profile as pre-refactor baseline. The VLM benchmark (SmolDocling, RTX 4090) remained at its prior tokens/sec after post-refactor stabilization.
- **Mode preservation is now explicit.** The hardcoded `MAX_AUTOTUNE` override was a silent behavior that only manifested when a user requested `SLOT_BY_SLOT` and got Triton-compiled code anyway. All recompile paths now read `configuredGraphExecutionMode` from the plan instead of hardcoding.

## Files Added/Modified

### Modified Files (refactor commits)
- `libnd4j/include/graph/NativeDynamicShapePlan.h` — top-level class, `setShapesFrozen()` declaration
- `libnd4j/include/graph/impl/NativeDynamicShapePlan.cpp` — `setShapesFrozen()` implementation
- `libnd4j/include/graph/NativeSlot.h` — 8 sub-structs, composition
- `libnd4j/include/graph/GraphSegment.h` — `GraphSegmentDef` + `GraphSegmentExec` split
- `libnd4j/include/graph/impl/NativeDynamicShapePlan_gpubackend.cpp` — field path updates
- `libnd4j/include/graph/impl/NativeDynamicShapePlan_segments.cpp` — field path updates, `offset()` fix
- `libnd4j/include/graph/impl/NativeDynamicShapePlan_slotexec.cpp` — `writeOutputSlot()` call sites
- `libnd4j/include/graph/impl/NativeDynamicShapePlan_cuda.cu`, `_cudagraph.cu`, `_batchzero.cu`, `_cuda_stubs.cpp` — field path updates
- `libnd4j/include/graph/impl/NativePlanCompiler.cpp` — segment def field access
- `libnd4j/include/graph/impl/SlotBufferOwnership.cpp` — sub-struct access
- `libnd4j/include/graph/gpu/TritonGraphBackend_*.{cpp,cu}` — field path updates
- `libnd4j/include/graph/gpu/TritonIRBuilder_sections.cpp` — `matmulEpilogueCount` fix
- `nd4j/.../execution/DynamicShapePlanExecutor.java` — mode preservation, constant caching, frozen sync
- `nd4j/.../internal/InferenceSession.java` — drop force-close of constants, add `planExecutorClosed`
- `nd4j/.../generation/StaticKvCacheDecodeLoop.java` — preserve mode across KV recompile

### Removed
- `DSP_SLOT_WRITE` and `DSP_NEW_ARRAY` macros (globally)
- `slotArrayCache_` as a distinct member (aliased via `#define` for backward compat)
