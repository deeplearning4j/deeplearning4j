# ADR 0083 - Thread-Local Cast Cache Leak Prevention

## Status
Accepted

## Context

`MmulHelper.cu` (CUDA matmul fast path) uses thread-local caches `tl_castCacheA` and `tl_castCacheB` to hold intermediate `NDArray` copies produced when an input has to be cast to a different dtype before the cuBLAS call (e.g., FP32 input → HALF operand for a TF32 matmul, or FLOAT32 × HALF mixed-type path). The cache is thread-local to avoid cross-thread contention and is intended to be reused across matmul calls within the same inference session.

The leak: `resetCastCacheIndices()` resets the *index* into the cache to 0 at the start of each matmul, but it does not reset the cache *contents*. Each new cast call appends via `push_back()` instead of overwriting the slot at the current index. On every matmul step, the cache grows by one entry per cast (up to two per matmul — one for A, one for B). The cached `NDArray` holds a device buffer allocation; growing without bound, the cache accumulates ~1.2 MB per matmul.

On SmolDocling (RTX 4090), this compounds to:

- ~1.2 MB × 211 matmuls per decode step = ~250 MB of GPU memory leaked per step.
- `nvidia-smi` confirmed: 250 MB/step growth before the fix, 0 MB/step growth after the fix.
- At ~100 tokens the model exhausts 24 GB VRAM and crashes with OOM.

The leak was masked in short tests (< 10 tokens) that never accumulated enough waste to OOM. It only manifested in full-length VLM benchmarks, where the tokens-per-OOM was ~100.

Three call sites had the same bug pattern:

1. `mmulMxM` — FP32×FP32→HALF TF32 path — both A and B cached.
2. `mmulMxM` — FLOAT32×HALF mixed-type path — A cached.
3. `mmulMxM` — HALF×FLOAT32 mixed-type path — B cached.
4. `mmulMxV` — FP32→HALF vector path — A and B cached.

## Decision

### 1. Replace `push_back()` with Indexed Overwrite

All four call sites now use the index from `resetCastCacheIndices()` to assign into the cache at the current position. If the current index is within the existing vector size, overwrite the existing slot (releasing the previous `NDArray` via RAII). If the index is at or beyond the vector size, extend the vector by one. This caps cache size at the maximum number of casts needed in any single matmul — the cache is reused across calls instead of growing.

### 2. Auto-Free-on-Launch for Captured Graph Handles

Independent but related: `CudaGraphHandle::AutoFreeOnLaunch` is now enabled by default. When a graph handle is destroyed, any device allocations captured in the graph are freed as part of the launch cleanup. Previously this was opt-in, which meant captured allocations survived the graph's lifetime and counted as leaks under `nvidia-smi`.

### 3. `selectTargetDevice` Delegates to `DeviceMemoryManager`

`CudaExecutioner::selectTargetDevice` previously had its own device selection logic that made independent decisions from `DeviceMemoryManager`. This produced divergent device choices in multi-GPU scenarios and contributed to cross-device leak accounting errors (allocations tracked on one device were freed on another). It now delegates to `DeviceMemoryManager::selectTargetDevice`, which is the single source of truth for memory pressure routing.

### 4. Close Cast Placeholder Copies After DSP Execution

`InferenceSession` now explicitly closes the cast placeholder copies it creates during DSP placeholder binding, after DSP execution completes. These were previously leaking because they were not part of the slot buffer lifecycle. A DSP_DIAG-based memory diagnostic was added in `writeOutputSlot` to expose future leaks of this class.

### 5. Per-Phase GPU Memory Tracking in Decode Loop

`StaticKvCacheDecodeLoop` now records GPU memory before and after each phase (warmup, prefill, decode step, KV recompile) and writes the deltas to the DSP diagnostic ring buffer under the `MEMORY` category. This makes future GPU memory regressions immediately visible — a per-step delta > 0 in steady state is a leak.

### 6. Pool Trim After State Reset

`BenchmarkConfigApplier` now calls `Nd4j.getMemoryManager().trimPools()` after `resetModelState` to release transient buffers held by the pool. Without this, benchmark reconfiguration accumulated GPU memory across configurations.

### 7. Leak Isolation Test

`TestDspValidation` was extended with memory leak isolation tests that run a fixed number of matmul iterations and assert that GPU memory is bounded by a fixed ceiling. The test failed reliably before the fix and passes after.

## Consequences

- **~250 MB/step leak eliminated.** `nvidia-smi` confirms 0 MB/step growth after the fix on SmolDocling. 1,000-token benchmarks now complete without OOM; pre-fix, they crashed at ~100 tokens.
- **Memory ceiling is knowable.** With the leak gone, the steady-state GPU memory usage for SmolDocling sits at ~5.3 GB (constants) + ~1 MB × token_count (KV cache growth). A 1,000-token decode uses ~6.3 GB total — well under the 24 GB RTX 4090 budget.
- **Per-phase memory tracking is built in.** Future leak investigations start with the DSP_DIAG `MEMORY` category output — no ad-hoc printf. Any phase with a non-zero steady-state delta is a leak candidate.
- **Device selection is unified.** `selectTargetDevice` goes through one code path. Multi-GPU benchmarks now account memory correctly per-device; pre-fix, they double-counted or lost allocations.
- **The `push_back` anti-pattern is documented.** Any thread-local cache that uses `reset-index-but-keep-contents` semantics must use indexed assignment, not `push_back`. This pattern is now explicitly called out in the code and should be the reviewed whenever a new thread-local cache is introduced.
- **Leak tests run in CI.** The memory leak isolation test is added to the DSP validation suite and runs on every merge. Regressions in any of the four fixed call sites — or in new call sites that reintroduce the pattern — will fail the test.

## Files Added/Modified

### Modified Files
- `libnd4j/include/helpers/MmulHelper.cu` — replace `push_back` with indexed assignment at 4 cast sites
- `libnd4j/include/graph/gpu/CudaGraphHandle.h` — `AutoFreeOnLaunch` default `true`
- `libnd4j/include/execution/cuda/CudaExecutioner.cu` — delegate `selectTargetDevice` to `DeviceMemoryManager`
- `nd4j/.../internal/InferenceSession.java` — close cast placeholder copies, DSP_DIAG memory diagnostics in `writeOutputSlot`
- `nd4j/.../generation/StaticKvCacheDecodeLoop.java` — per-phase GPU memory tracking
- `nd4j/.../benchmark/BenchmarkConfigApplier.java` — trim pools after `resetModelState`

### Added Tests
- `platform-tests/.../TestDspValidation.java` — memory leak isolation tests with fixed ceilings

## References

- ADR 0060 — CUDA Async Memory Pool (underlying allocator behavior)
- ADR 0063 — ArrayCacheMemoryMgr Buffer Reuse (related cache pattern — also subject to the reset-index-vs-contents distinction)
- ADR 0065 — Multi-GPU Memory Management (`selectTargetDevice` delegation)
- ADR 0070 — GC Pressure Optimization (related memory tracking)
- Commit `47a24d3ce4` — original fix
