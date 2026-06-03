# ADR 0094 - DSP Buffer Coloring, Pooling, and Passivation

## Status
Implemented

Proposed by: Adam Gibson (June 2026)

## Context

A single DSP plan for a large LLM (28-layer Qwen, ~1000+ slots) allocates one physical GPU buffer per output slot. Most intermediate buffers have non-overlapping lifetimes — slot A's output is consumed and dead before slot B even produces — yet each gets its own `cudaMalloc`. For a 1000-slot plan, this means ~1000 separate GPU allocations for intermediates, 10-20x more than the live working set at any point during execution.

Multiple plans for the same model (prefill vs decode, different sequence lengths) compound this waste. Each plan holds its own copy of identically-shaped intermediate buffers, even when only one plan is active at a time.

The existing `ArrayCacheMemoryMgr` (ADR 0063) operates at the Java level and handles per-op buffer reuse across SameDiff steps. The `CudaMemoryPool` (ADR 0060) operates at the device allocation level. Neither addresses **intra-plan buffer sharing** (slots within a single compiled plan sharing physical memory) or **cross-plan buffer reuse** (inactive plans releasing buffers for active plans to acquire).

## Decision

We implement a three-tier memory reduction architecture that composes: coloring reduces what a plan **needs**, pooling shares what plans **have**, passivation releases what plans **aren't using**.

| Tier | What | Scope | Mechanism |
|---|---|---|---|
| **Coloring** | Non-overlapping slots share buffers within a plan | Per-plan | Greedy interval graph coloring |
| **Pooling** | Plans share physical buffers across plans via a global pool | Cross-plan | Per-device `(numElements, dtype)`-keyed pool |
| **Passivation** | Plans not in use release everything back to pool | Cache-level | LRU eviction with three rounds |

### Tier 1: Buffer Coloring (`DspBufferColorMap`)

Buffer coloring is analogous to register allocation in compilers. The "registers" are physical GPU buffers; the "variables" are slot outputs with known live ranges.

#### Liveness Analysis

`NativePlanCompiler` persists `SlotLivenessData` on `PlanDefinition` during plan compilation. For each output slot, it records:
- `producerStep`: the execution step index where the slot's value is produced
- `lastConsumerStep`: the execution step index where the slot's value is last consumed

This data is immutable and shared across all per-thread plan instances that use the same `PlanDefinition`.

#### Coloring Algorithm

`DspBufferColorMap::compute()` runs after `SHAPES_FROZEN`:

1. **Eligibility filter**: Exclude views (`VIEW_OF_SLOT`), view parents (slots with `viewRefCount > 0`), requested output slots (user needs their dedicated buffer), and non-`SLOT_OWNED` slots.
2. **View-extended liveness**: Extend a slot's effective `lastConsumerStep` to cover all its `VIEW_OF_SLOT` children, since the parent's buffer must live as long as any view of it.
3. **Shape grouping**: Group eligible slots by `(shape, dtype, device)` — only slots with identical memory footprint can share a buffer.
4. **Greedy interval coloring**: Within each shape group, sort by `producerStep` and assign colors greedily. Two slots get the same color only if their extended live ranges do not overlap (checked by `assertNoOverlap()`).

#### Apply / Eject

- `apply()` replaces per-slot buffers with shared color buffers acquired from `DspBufferPool`. The "master" slot (first assigned to a color) keeps its buffer; all other slots in the same color get a new `NDArray` wrapping the master's `DataBuffer`.
- `eject()` undoes coloring gracefully: acquires fresh dedicated buffers from the pool, copies current data from shared buffers to the new dedicated buffers (preserving correctness mid-execution), and releases old shared buffers back to the pool.

Ejection triggers: validation inconsistency, shape change (plan demotion to `SLOT_BY_SLOT`), OOM during capture, segment invalidation, or manual disable.

#### Safety Invariants

All checked by `validate()` (called every execute in debug mode):
- Same-color slots share the same `dataBuffer()` address
- No overlapping live ranges share a `dataBuffer()`
- `VIEW_OF_SLOT` parents are never colored
- Uncolored `SLOT_OWNED` slots retain their original buffer
- `numColors` matches distinct `DataBuffer` count among colored slots

### Tier 2: Buffer Pool (`DspBufferPool`)

A per-device singleton pool of reusable `DataBuffer` objects, keyed by `(numElements, dtype)`.

```
Plan A releases buffer [1024 × FLOAT] → pool
Plan B acquires buffer [1024 × FLOAT] → gets Plan A's buffer (zero-copy)
```

- `acquire(numElements, dtype)`: If the pool has a matching buffer, return it. Otherwise allocate fresh.
- `release(buffer, numElements, dtype)`: Return buffer to pool for reuse. Double-release detection via `pooledSet_`.
- `trim(targetFreeBytes)`: Free pooled buffers until at least `targetFreeBytes` have been freed. Called by the plan cache under memory pressure.

Thread safety: all public methods are mutex-guarded. The pool is per-device so contention is limited to plans on the same GPU.

Device selection: `DspBufferPool::forCurrentDevice()` uses `AffinityManager::currentDeviceId()` — no platform guards needed at call sites.

#### How coloring uses the pool

- `DspBufferColorMap::apply()` calls `pool.acquire()` for the master buffer of each color
- `DspBufferColorMap::eject()` calls `pool.release()` for shared buffers, `pool.acquire()` for dedicated replacements
- Plans call `pool.release()` when they release intermediates (passivation)
- Plans call `pool.acquire()` when they re-warm (de-passivation)

### Tier 3: Passivation (in `NativePlanCache`)

When memory pressure is detected, the plan cache passivates LRU unpinned plans before full eviction.

#### Four-Round Eviction

`evictIfOverBudgetLocked()` proceeds:

1. **Hard count cap**: Full eviction of oldest unpinned plans if count exceeds limit (unchanged from prior behavior).
2. **Passivation round**: For LRU unpinned, non-passivated plans, call `plan->passivate()`. This releases GPU intermediates back to `DspBufferPool` but keeps plan metadata (slots, segments, wiring) alive in heap (~10-50 MB). Plan stays in cache.
3. **Pool trim round**: If still over budget, call `DspBufferPool::forCurrentDevice().trim(overshoot)` to free pooled buffers.
4. **Full eviction round**: If still over budget, delete LRU plans entirely (existing behavior).

#### Plan Passivation/Reactivation

- `passivate()`: Calls `releaseGpuIntermediates()`, sets `passivated_ = true`. Returns bytes freed.
- `reactivate()`: Clears `passivated_` flag. The existing execute warmup path handles buffer re-allocation on the next cache hit.
- On cache hit for a passivated plan: `reactivate()` is called automatically in `getOrInsert()`.

### Lifecycle Integration

| Transition | Coloring | Pool |
|---|---|---|
| Warmup → `SHAPES_FROZEN` | `colorMap_.compute()` | — |
| First frozen execution | `colorMap_.apply(pool)` | `pool.acquire()` for master buffers |
| Shape change → `SLOT_BY_SLOT` | `colorMap_.eject(pool)` | `pool.release()` shared + `pool.acquire()` dedicated |
| Passivation (LRU cache pressure) | `colorMap_.eject(pool)` | `pool.release()` all intermediates |
| Re-activation (cache hit) | recompute on next freeze | `pool.acquire()` on re-warm |
| Plan destruction | — | `pool.release()` all plan-owned buffers |

### JNI Introspection

Exposed via `DspHandle.java`:

| Method | Returns |
|---|---|
| `bufferColoringApplied()` | bool |
| `bufferColoringNumColors()` | int |
| `bufferColoringBytesSaved()` | long |
| `slotColor(int slotIdx)` | int |
| `bufferPoolPooledBytes(int deviceId)` | long (static) |
| `bufferPoolPooledCount(int deviceId)` | int (static) |
| `bufferPoolTotalAcquired(int deviceId)` | long (static) |
| `bufferPoolTotalReused(int deviceId)` | long (static) |

`DspPlanAssertions.assertColoringConsistent()` validates applied/numColors consistency.

### DSP Diagnostics

New `COLORING` category in `DspDiagnostics.h`:

| Event | Level | Content |
|---|---|---|
| `COLORING_COMPUTE_DONE` | summary | `%d slots → %d colors, saving %zuMB` |
| `COLORING_APPLY_DONE` | summary | `consolidated=%d, freed %zuMB` |
| `COLORING_EJECTED` | summary | `reason=%s, restored %d slots` |
| `POOL_ACQUIRE` | full | `shape=%s dtype=%s reused=%d` |
| `POOL_RELEASE` | full | `shape=%s dtype=%s pooledBytes=%zu` |
| `POOL_TRIM` | summary | `freed %zuMB, %d buffers remaining` |
| `CACHE_PASSIVATE` | summary | `plan=%p freed %zuMB to pool` |
| `CACHE_REACTIVATE` | summary | `plan=%p acquired %zuMB from pool` |

## Files

| File | Change |
|---|---|
| `libnd4j/include/graph/DspBufferColorMap.h` | New — `SlotLivenessData`, `DspBufferColorMap` |
| `libnd4j/include/graph/impl/DspBufferColorMap.cpp` | New — coloring, apply, eject, validate |
| `libnd4j/include/graph/DspBufferPool.h` | New — per-device buffer pool |
| `libnd4j/include/graph/impl/DspBufferPool.cpp` | New — acquire, release, trim, `forCurrentDevice()` |
| `libnd4j/include/graph/DspDiagnostics.h` | Added `COLORING` category |
| `libnd4j/include/graph/PlanDefinition.h` | Added `SlotLivenessData*` member |
| `libnd4j/include/graph/impl/PlanDefinition.cpp` | Delete `slotLiveness_` in destructor |
| `libnd4j/include/graph/impl/NativePlanCompiler.cpp` | Persist live ranges (Step 6b) |
| `libnd4j/include/graph/NativeDynamicShapePlan.h` | Added `DspBufferColorMap`, passivation members |
| `libnd4j/include/graph/impl/NativeDynamicShapePlan.cpp` | Wire coloring/pool into lifecycle |
| `libnd4j/include/graph/impl/NativePlanCache.cpp` | Four-round eviction with passivation |
| `libnd4j/include/dsp/NativeOpsDsp.h` | 8 JNI introspection declarations |
| `libnd4j/include/legacy/cuda/NativeOps_dsp.cu` | JNI implementations |
| `libnd4j/include/legacy/cpu/NativeOps_dsp.cpp` | JNI implementations |
| `nd4j/.../execution/DspHandle.java` | Java accessors |
| `nd4j/.../execution/DspPlanAssertions.java` | `assertColoringConsistent()` |
| `nd4j/.../nativeblas/NativeOps.java` | 8 default interface methods |
| `platform-tests/.../DspBufferColoringTest.java` | 5 tests |

## Verification

- CUDA build: SUCCESS (zero compilation errors)
- `DspBufferColoringTest`: 5/5 pass
- DSP regression gate: 1590 tests, 0 failures, 0 errors
