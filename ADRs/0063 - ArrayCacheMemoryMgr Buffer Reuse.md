# ADR: ArrayCacheMemoryMgr Buffer Reuse

## Status

Implemented

Proposed by: Adam Gibson (January 2025)

Discussed with: Development Team

## Context

Autoregressive generation workloads allocate and deallocate thousands of intermediate arrays per decode step. Each step in a VLM decoder (e.g., SmolDocling) executes ~4400 ops, many producing intermediate arrays that are consumed and discarded within the same step. Without buffer reuse, every intermediate requires a fresh GPU allocation and eventual deallocation.

The original `ArrayCacheMemoryMgr` had several design issues:

**Lock Contention**: A global lock protected all cache operations, serializing allocation across threads during multi-GPU execution.

**Linear Scan Lookup**: Finding a suitable cached buffer required iterating through all cached arrays to find one with matching data type and sufficient capacity. For large caches (1000+ entries), this O(n) scan added measurable overhead.

**Growth Factor Leak**: The `DEFAULT_GROWTH_FACTOR` of 1.05 (5% headroom) created buffers where `data().length() > length()`. The `BaseNDArray.closeable()` method returned `false` for such arrays (treating them as views), which caused DSP release paths gated on `closeable()` to permanently leak these oversized buffers.

**No Size-Based Eviction**: All cached arrays were treated equally regardless of size, causing large stale buffers to crowd out smaller frequently-reused ones.

## Decision

We rewrite ArrayCacheMemoryMgr with a capacity-indexed TreeMap architecture, LRU eviction, and fixes for the closeable gate leak.

### Architecture

```
┌───────────────────────────────────────────────────────────────┐
│                    ArrayCacheMemoryMgr                          │
│                                                                │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ capacityIndex: Map<DataType, TreeMap<Long, Deque<Array>>>│  │
│  │  Key: buffer element count                               │  │
│  │  Value: queue of arrays with that exact capacity          │  │
│  │  Lookup: ceilingEntry(requiredElements) → O(log n)       │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ lruMap: LinkedHashMap<IdentityKey, INDArray>              │  │
│  │  Insertion order tracking for LRU eviction                │  │
│  │  IdentityHashMap semantics (reference equality)           │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ Memory Budget                                             │  │
│  │  maxCacheCapacity = maxMemFrac * totalMemory (default 25%)│  │
│  │  currentCacheSize tracks live cache memory usage          │  │
│  │  PER_SLOT_EVICTION_THRESHOLD = 64KB (large-first evict)  │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ Diagnostics                                               │  │
│  │  exactHits, capacityHits, fullMisses                      │  │
│  │  releasesSkipped (views), overallocCount                  │  │
│  └──────────────────────────────────────────────────────────┘  │
└───────────────────────────────────────────────────────────────┘
```

### Capacity-Based Lookup

The TreeMap per DataType enables O(log n) buffer lookup:

```java
public INDArray tryAllocateFromCapacityCache(DataType dtype, long[] shape) {
    long requiredElements = ArrayUtil.prod(shape);
    long maxElements = (long)(requiredElements * largerArrayMaxMultiple); // default 2.0x

    TreeMap<Long, ArrayDeque<INDArray>> typeCache = capacityIndex.get(dtype);
    // Find smallest buffer >= requiredElements
    Map.Entry<Long, ArrayDeque<INDArray>> entry = typeCache.ceilingEntry(requiredElements);

    if (entry != null && entry.getKey() <= maxElements) {
        INDArray cached = entry.getValue().poll();
        // Reshape to requested shape (no allocation — same underlying buffer)
        return cached.reshape(shape);
    }
    return null; // Cache miss — allocate fresh
}
```

The `largerArrayMaxMultiple` (default 2.0) controls how much larger a cached buffer can be before it's rejected. A buffer with 2x the needed elements is acceptable; 3x would waste too much memory.

### LRU Eviction

When the cache exceeds its memory budget, entries are evicted in insertion order (oldest first):

```java
if (currentCacheSize > maxCacheCapacity) {
    // Evict large arrays first (> 64KB)
    Iterator<Map.Entry<IdentityKey, INDArray>> it = lruMap.entrySet().iterator();
    while (it.hasNext() && currentCacheSize > maxCacheCapacity) {
        INDArray victim = it.next().getValue();
        if (victim.data().length() * victim.data().getElementSize() > PER_SLOT_EVICTION_THRESHOLD) {
            evict(victim);
            it.remove();
        }
    }
}
```

Large arrays (>64KB) are evicted first because they contribute most to memory pressure and are less likely to be reused (autoregressive decode tends to reuse fixed-size KV cache buffers, not large intermediates).

### Growth Factor and Closeable Gate Fix

The growth factor (default 1.05) creates buffers with 5% extra capacity for better reuse across slightly-varying shapes. However, this triggered a critical leak:

**Problem Chain**:
1. `allocateWithHeadroom()` creates buffer with `requiredElements * 1.05`
2. `data().length() > length()` → `BaseNDArray.closeable()` returns `false`
3. DSP release paths gated on `closeable()` → oversized buffers NEVER freed

**Fix**:
1. DSP output slots allocate via `Nd4j.create(dt, shape)` directly, bypassing the growth factor
2. All DSP release/cache/free paths gate on `isConstant()` instead of `closeable()`
3. `closeable()` uses `length() < data.length()` check instead of `isView()`

**Rule**: NEVER gate buffer release on `closeable()` — use `isConstant()` instead. The `closeable()` method conflates "buffer is a view" with "buffer should not be freed", which is incorrect when growth factors create non-view buffers with mismatched lengths.

### Memory Fractions

Cache capacity is configured as a fraction of total GPU memory:

- `maxMemFrac`: Maximum cache size as fraction of total memory (default 0.25 = 25%)
- `smallArrayThreshold`: Arrays smaller than 1024 elements get separate treatment
- All thresholds are ratio-based, not fixed sizes, ensuring correct behavior across GPUs with different memory sizes

## Consequences

### Advantages

**Cache Hit Rate**: 70-85% hit rate in autoregressive decoding. Most intermediate arrays have stable shapes across decode steps, making capacity-based lookup effective.

**Allocation Savings**: ~20% reduction in peak GPU memory usage from buffer reuse. Fewer `cudaMallocAsync` calls reduce driver overhead.

**O(log n) Lookup**: TreeMap-based capacity index provides logarithmic lookup time vs. linear scan in the previous implementation.

**No More Closeable Leak**: Gating on `isConstant()` instead of `closeable()` eliminates the permanent leak from growth-factor-enlarged buffers. VLM decode memory growth drops from ~40MB/step to ~1MB/step.

### Disadvantages

**Cache Staleness**: Cached buffers from early decode steps (with small KV cache) are useless for later steps (with large KV cache). The LRU eviction handles this naturally but wastes the initial cache warming.

**Memory Overhead**: The cache itself consumes up to 25% of GPU memory for buffer storage. This is memory that could otherwise be used for model weights or batch processing.

**Growth Factor Complexity**: The 5% headroom improves reuse but complicates buffer lifecycle management. The fix (gating on `isConstant()`) works but adds a subtle invariant that future developers must maintain.

## References

- ArrayCacheMemoryMgr.java in nd4j-api
- DynamicShapePlanExecutor.java (DSP integration)
- BaseNDArray.closeable() (closeable gate analysis)
