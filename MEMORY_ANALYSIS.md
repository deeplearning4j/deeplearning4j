# Memory Usage Analysis: TestSmolDoclingOptimizedPipeline Benchmark

## Overview
Analysis of GPU and host memory progression during VLM decode benchmark run that encountered out-of-memory conditions while loading vision encoder embeddings. Benchmark terminated after ~89.5 seconds due to a multi-GPU memory allocation failure cascading into a use-after-free issue.

## System Configuration
- **GPU 0 (RTX 4090?)**: 24084 MB total capacity, 75% threshold = 18063 MB
- **GPU 1 (smaller GPU)**: 7851 MB total capacity, 75% threshold = 5888 MB
- **Pinned Host Memory**: 8589934592 bytes (8.0 GB) limit
- **Setup**: Dual-GPU system with non-peer capability between devices

## Memory Progression Timeline

### Phase 1: Model Loading
**Timeframe**: Model deserialization (line 1677-2520)

- **Variables loaded**: 2297 + 75 array constants = 2372 total
- **Operations in graph**: 1953 ops initially, 2743 final ops after optimization
- **Batch sync to GPU**: 75 constant arrays synced to device in batch mode
- **Result**: Model constants loaded (~5.3 GB based on prior analysis)

### Phase 2: Early Execution (Pre-OOM)
**Timeframe**: Graph execution begins through first OOM attempt (line 24778)

- **Pool state at first cudaMallocAsync failure**:
  - Pool used: 14480 MB
  - Pool reserved: 14528 MB
  - Reclaimable: 47 MB
  - Device memory (MemoryCounter): 14434 MB
  - Free CUDA memory: 95 MB

- **Requested allocation size**: 12582920 bytes (12.00 MB)
- **Device 0 at capacity**: 14434 MB / 24084 MB = 59.9% used before OOM occurs

### Phase 3: OOM Cascade and Failover Pattern
**Timeframe**: Lines 24778-24920+ (repeated OOM→failover cycles)

#### Failover Pattern Observed:
1. **Primary allocation fails** on device 0 (cudaMallocAsync out of memory)
2. **Trim & recover attempt**: cudaFree() yields 70-95 MB recovery
3. **Pool still exhausted**: 14480-14486 MB used out of 14528 MB reserved
4. **Non-peer failover**: Falls back to device 1 (peer device) which has 5.9 GB free
5. **Host fallback**: "Pinned host fallback: 1193 MB" used at line 27602

**Memory counter progression at OOM points:**

| Attempt | Device[0] | Host | Pool Used | Pool Reserved |
|---------|-----------|------|-----------|---------------|
| 1st OOM | 14434 MB | 13516 MB | 14480 MB | 14528 MB |
| 2nd OOM | 14449 MB | 13531 MB | 14483 MB | 14528 MB |
| 3rd OOM | 14464 MB | 13546 MB | 14486 MB | 14528 MB |
| Steady state | ~14481-14485 MB | 13556 MB | 14486 MB | 14528 MB |
| Final (line 27601) | 15658 MB | 14647 MB | 14489 MB | 14528 MB |

**Observations:**
- Pool stays **pegged at 14486-14489 MB** (96-99% of 14528 MB reserved)
- MemoryCounter device[0] grows from 14434 → 15658 MB (~1.2 GB increase)
- Host memory grows from 13516 → 14647 MB (~1.1 GB increase)
- Pinned host fallback activated: 1193 MB

### Phase 4: Critical Failure (Vision Encoder Embeddings)
**Timeframe**: Line 27602-27625 (final allocation failure)

**Context**: Vision encoder embedding reshape operation
- Input shape: `[3, 512, 512]` (786,432 elements = 3.0 MB for FLOAT)
- Output shape: `[1, 1, 3, 512, 512]` (same 786,432 elements, reshaped as view)
- **Issue**: Input array **CLOSED** before reshape_no_copy execution

**Error sequence:**
1. **Line 27602**: Final pinned host fallback: 1193 MB
2. **Line 27604**: Pool still shows 14489 MB / 14528 MB (reclaimable=38 MB)
3. **Line 27605**: MemoryCounter device[0] = 15658 MB (exceeds reserved pool!)
4. **Line 27608**: Non-peer failover triggered for small input migration
5. **Line 27609**: Multi-GPU routing decision: device 0 has only 70 MB free, routes to device 1
6. **Line 27610**: "Async transfer failed"
7. **Line 27617-27620**: ContextBuffers._reductionPointer cudaMallocAsync fails during `dbMigrate()`
8. **Line 27625**: **"Input argument at index 0 was closed before call. shape=[3, 512, 512]"**

## Key Memory Issues Identified

### Issue 1: Device Memory Exhaustion (15.6+ GB on 24 GB GPU)
- **Cause**: MemoryCounter shows 15658 MB allocated on device 0
- **Pool reserved**: Only 14528 MB, but MemoryCounter shows 15.6+ GB used
- **Gap**: ~1.1 GB unaccounted for (likely host buffers synced to device)
- **Implication**: Either:
  - Pool memory accounting is not catching all allocations
  - Non-pool allocations are accumulating (workspace, temp arrays)
  - Host buffers being synced to device without pool tracking

### Issue 2: Pinned Host Memory Pressure (1.2 GB growth, 8 GB limit)
- **Initial**: 13516 MB on host
- **Final**: 14647 MB on host
- **Growth**: +1131 MB
- **Pinned host fallback**: 1193 MB activated
- **Implication**: Host-side arrays not being closed/GC'd, accumulating as plan executes

### Issue 3: Vision Encoder Array Lifetime Violation
- **Symptom**: reshape_no_copy receives closed input array
- **Array ID**: 8324, shape=[3,512,512], dtype=FLOAT
- **Allocated**: Line 27608-27609 as non-peer failover on device 1
- **Closed**: By line 27625, before reshape operation
- **Root cause**: Likely InferenceSession or DSP plan cleanup prematurely closing arrays that are still referenced in OpContext

### Issue 4: Multi-GPU Cascade on OOM
When device 0 runs out of memory:
1. Non-peer failover routes to device 1
2. Async transfer of input requires _reductionPointer on device 0
3. _reductionPointer cudaMallocAsync **also fails** (no GPU context available)
4. Cascade: single allocation failure → multi-device context allocation failure → input closed

## Memory Pool Saturation Root Cause

### Observed Behavior:
- Device 0 pool **remains saturated at 14486 MB** across 40+ OOM cycles
- Trim recovers 0-81 MB but cannot make progress
- MemoryCounter keeps growing (14434 → 15658 MB)

### Hypotheses:

**H1: Deferred-Close Accumulation (Most Likely)**
- Arrays marked for close via `setCloseable(false)` → `setConstant(true)` not being freed
- DSP plan execution creates thousands of intermediates
- Deferred close queue backs up, memory not returned to pool
- Each step allocates new arrays, previous step's are still pending close

**H2: View-Based Leak**
- reshape_no_copy, slice, gather operations create views
- Views share DataBuffer with parent
- Parent closed while views still active → DataBuffer deallocation races
- Child views left pointing to freed/reused GPU memory

**H3: Host Buffer Synchronization**
- H2D transfers for shape functions triggered on large arrays
- Host buffers replicated to GPU during `ensureAvailableOn()` calls
- Sync not being reversed; GPU replicas accumulate
- `MemoryCounter device[0]` counts both primary and replicated host copies

## Evidence for View-Based Lifecycle Issue

From error output (lines 27648-27649):
```
Input[0]:
  IsView: true
  IsAttached: false
```

And later (line 27625):
```
DIAGNOSTIC: inputsFromOp OpContext path for op 'null' (reshape_no_copy).
Input at index 0 is CLOSED in OpContext
```

**This indicates**:
1. Vision encoder output is a **view** (reshape_no_copy on prior concat/gather)
2. View's **parent array closed** while view still active in DSP OpContext
3. When reshape_no_copy op tried to execute, OpContext validation failed
4. InferenceSession cleanup destroying array → DataBuffer freed → memory returned to pool
5. But OpContext still has reference to view (pointer now stale)

## Memory Budget Analysis

### Total Capacity: 24084 MB (GPU 0)

**Breakdown at Failure:**
- Model constants: ~5.3 GB (from prior analysis)
- Pool reserved: 14.5 GB (60% of total)
- Pool used: 14.5 GB (95-98% of reserved)
- MemoryCounter (actual allocations): 15.6 GB
- **Unaccounted**: ~5.6 GB gap (14084 - 5300 - 14500 + 15658 ≈ 7 GB in pool + device allocations)

**Issue**: Pool is managing only ~14.5 GB of a 24 GB GPU. The remaining ~9.5 GB is:
- Triton compilation cache
- CUDA context overhead
- Other runtime allocations

The DSP vision encoder execution requires:
- 5.3 GB model constants
- ~1-2 GB intermediate activations per layer
- 3-stage inference (text, images, decoder) → 3-5 GB intermediates
- Total: ~8-10 GB needed, but pool already had 5.3 GB constants → only ~9 GB available for exec

**Pool exhaustion was inevitable** with current allocation patterns.

## Recommended Fixes

### Immediate (High Priority):
1. **Fix view-based array lifecycle** — ensure views don't outlive parents
   - InferenceSession should NOT close arrays still referenced in OpContext
   - DSP plan cleanup should track OpContext references before deallocation
   - See MEMORY.md #5 (in_progress)

2. **Reduce deferred-close backlog** — force close at step boundaries
   - Don't rely on automatic deallocation during plan execution
   - Call `closeable.close()` after each DSP step completes
   - Avoid accumulating thousands of pending-close arrays

### Medium (Configuration):
3. **Increase pool reserved percentage** — currently 60% of total
   - Change from `0.60 * total` to `0.75 * total`
   - Gives DSP more room for intermediate allocations
   - Trade: Less CUDA runtime headroom

4. **Implement progressive GC** during vision encoder
   - After each transformer block, garbage collect intermediate layers
   - Use `Nd4j.getMemoryManager().gcIfHeapPressured()` at layer boundaries
   - Reduces peak memory footprint

### Long-term:
5. **Profiling-guided memory budget** — characterize actual needs
   - Run vision encoder alone, measure peak allocation
   - Parameterize pool reserve by vision/text/decoder workload
   - Currently: one-size-fits-all 60% may not be optimal

## Conclusion

The benchmark OOM was triggered by a **perfect storm**:

1. **Cascading view-based lifecycle issue** (primary) — array parent closed while views still active
2. **Pool saturation** (secondary) — deferred-close accumulation + large DSP graph intermediates
3. **Multi-GPU complexity** (tertiary) — non-peer failover requires ContextBuffers, which also OOMs

The immediate fix is **not** to increase pool size, but to ensure array lifetimes are properly managed in the DSP + OpContext ecosystem. See task #5 in MEMORY.md for ongoing work on this issue.
