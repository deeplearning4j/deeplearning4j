# ADR: CUDA Async Memory Pool

## Status

Implemented

Proposed by: Adam Gibson (January 2025)

Discussed with: Development Team

## Context

LibND4J's original CUDA memory management used direct `cudaMalloc`/`cudaFree` calls for every allocation and deallocation. This approach has several problems that become critical during inference workloads like autoregressive token generation:

**Allocation Latency**: Each `cudaMalloc` call requires a round-trip to the CUDA driver to negotiate memory from the OS. This takes 100-1000 microseconds per call, which dominates execution time when models execute thousands of small ops per decode step (e.g., 1962 ops for a vision encoder frame).

**Fragmentation**: Frequent allocate/free cycles create memory fragmentation in the driver's free list. Over thousands of decode steps, the driver cannot satisfy requests despite having sufficient total free memory, leading to premature OOM failures.

**No Cross-Stream Reuse**: Memory freed on one CUDA stream cannot be reused by allocations on another stream without explicit synchronization, creating invisible waste in multi-stream execution pipelines.

**Multi-GPU Complexity**: When a single GPU runs out of memory, the framework had no systematic fallback strategy. Allocations would simply fail, crashing the application even when other GPUs or host memory had capacity available.

## Decision

We implement a CUDA async memory pool based on `cudaMallocAsync`/`cudaFreeAsync` (CUDA 11.2+), with a multi-device failover strategy and stream-aware pool management.

### Architecture

```
┌──────────────────────────────────────────────────────────┐
│                    CudaMemoryPool                         │
│                                                          │
│  ┌────────────────┐  ┌─────────────────────────────────┐ │
│  │ Pool Per Device │  │   Failover Chain                │ │
│  │                 │  │                                 │ │
│  │  cudaMallocAsync│  │  1. trimPool + retry            │ │
│  │  cudaFreeAsync  │  │  2. Peer-accessible GPUs        │ │
│  │                 │  │  3. Non-P2P GPUs (host-staged)  │ │
│  │  poolUsed       │  │  4. Pinned host memory          │ │
│  │  poolReserved   │  │                                 │ │
│  └────────────────┘  └─────────────────────────────────┘ │
│                                                          │
│  ┌────────────────────────────────────────────────────┐  │
│  │ Stream-Aware Trimming                              │  │
│  │  dirtyFreeStreams_[device] tracks pending frees     │  │
│  │  trimPool() syncs only dirty streams, not device   │  │
│  └────────────────────────────────────────────────────┘  │
│                                                          │
│  ┌────────────────────────────────────────────────────┐  │
│  │ P2P Access Matrix                                  │  │
│  │  peerAccessEnabled_[i][j] initialized at startup   │  │
│  │  Gates direct vs. host-staged cross-device access  │  │
│  └────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────┘
```

### Pool Allocation (Fast Path)

The primary allocation path uses `cudaMallocAsync` with the device's memory pool:

```cpp
void* CudaMemoryPool::allocate(size_t size, int deviceId, cudaStream_t stream) {
    void* ptr;
    cudaError_t err = cudaMallocAsync(&ptr, size, stream);
    if (err == cudaSuccess) {
        poolUsed[deviceId] += size;
        return ptr;
    }
    // Never fall back to cudaMalloc — go to failover
    return allocateFailover(size, deviceId);
}
```

The CUDA driver maintains an internal pool of reserved memory. Freed blocks are returned to the pool (not the OS) and reused by subsequent allocations on the same stream without driver round-trips. This reduces allocation latency from ~100-1000 microseconds to ~0.1 microseconds.

**Critical Rule**: When `cudaMallocAsync` fails, we NEVER fall back to `cudaMalloc`. Direct allocations bypass pool statistics tracking (`poolUsed`/`poolReserved`), causing the pool to lose track of memory and leading to cascading allocation failures. All failures go through `allocateFailover()`.

### Proactive Soft Limit (CUDA)

The pool supports a proactive soft limit that triggers failover **before** `cudaMallocAsync` fails. This prevents the cascade of trim-retry-fail cycles that occur when a device is near capacity:

```cpp
void* CudaMemoryPool::allocate(size_t size, int deviceId, cudaStream_t stream) {
    int softLimit = softLimitPercent_.load(std::memory_order_relaxed);
    if (softLimit > 0) {
        size_t freeMem, totalMem;
        cudaMemGetInfo(&freeMem, &totalMem);
        double usagePercent = 100.0 * (1.0 - (double)freeMem / (double)totalMem);
        if (usagePercent >= (double)softLimit) {
            // Proactively route to another device before OOM
            return allocateFailover(size, deviceId);
        }
    }
    // Normal fast path...
}
```

**Configuration**:
- `SD_CUDA_SOFT_LIMIT_PERCENT` environment variable (0-100, 0=disabled)
- `Environment::setCudaSoftLimitPercent(int)` programmatic API
- `NativeOps::setMemoryPoolSoftLimitPercent(int)` JNI API from Java

**Key design decisions**:
- Uses `cudaMemGetInfo` (real driver query) rather than pool statistics, because pool `poolReserved` includes reusable blocks and overstates pressure.
- All log output is guarded by `isVerbose()` since this check runs on every allocation in the hot path.
- When the soft limit triggers and the failover device also exceeds the soft limit, the allocation proceeds on that device anyway (one-level proactive check, not recursive).

### Multi-Device Failover Strategy

When the current device cannot satisfy an allocation:

1. **Trim + Retry**: Release reserved-but-unused memory from the pool (`trimPool`), then retry on the same device. This handles the case where the pool has reserved excess memory that can be returned.

2. **Peer-Accessible Devices**: Try allocation on GPUs with P2P access enabled. Cross-device memory access is transparent via NVLink/PCIe peer mappings — no data copies needed for compute access.

3. **Non-P2P Devices**: Try GPUs without P2P access, sorted by free memory. Data must be staged through host memory (D2H + H2D) for cross-device access, but this is preferable to OOM.

4. **Pinned Host Memory**: Last resort. `cudaMallocHost` allocates page-locked host memory accessible from all GPUs via PCIe. Slower than device memory but prevents application crashes.

### Stream-Aware Pool Trimming

Pool trimming requires synchronization to ensure all pending `cudaFreeAsync` calls complete before releasing memory back to the OS. Naive device-wide synchronization (`cudaDeviceSynchronize`) would block ALL GPU work on ALL streams.

Instead, we track which streams have pending frees:

```cpp
// In free():
dirtyFreeStreams_[device].insert(stream);

// In trimPool():
auto dirtyStreams = std::move(dirtyFreeStreams_[device]);
for (auto& stream : dirtyStreams) {
    cudaStreamSynchronize(stream);
}
cudaMemPoolTrimTo(pool, 0);
```

This syncs only the streams that actually had frees, leaving compute-bound streams running uninterrupted.

### Pool Statistics

The pool tracks two key metrics per device:

- **poolUsed**: Sum of live allocation sizes. Incremented on allocate, decremented on free.
- **poolReserved**: Total memory reserved by the CUDA driver (available via `cudaMemPoolGetAttribute`).

`cudaMemGetInfo` reports free memory as `totalDeviceMemory - poolReserved - otherAllocations`. This means low reported free memory does NOT indicate OOM — the pool may have ample reusable capacity within its reserved space. This distinction is critical for load balancing and device selection decisions.

### Release Threshold

The pool's release threshold is set to 75% of device total memory:

```cpp
cudaMemPoolSetAttribute(pool, cudaMemPoolAttrReleaseThreshold, threshold_75pct);
```

This means the pool retains up to 75% of device memory for reuse, even when allocations drop. The remaining 25% stays available for non-pool uses (CUDA contexts, cuDNN workspaces, display buffers).

## Consequences

### Advantages

**Allocation Performance**: Pool reuse reduces allocation latency by ~1000x (0.1μs vs 100μs). For the VLM vision encoder (1962 ops/frame), this saves ~200ms per frame.

**Memory Efficiency**: Pool-managed reuse reduces peak GPU memory by ~20%. Freed blocks are immediately available for reuse without OS round-trips. VLM decode shows ~1MB/step growth vs. ~40MB/step without pooling.

**OOM Resilience**: The four-stage failover chain eliminates hard OOM crashes. Non-P2P GPUs and pinned host memory serve as overflow capacity, degrading performance gracefully instead of crashing.

**Stream Isolation**: Dirty-stream tracking ensures compute streams are never blocked by pool management. This is critical for multi-stream execution in DynamicShapePlan.

### Disadvantages

**CUDA 11.2+ Requirement**: `cudaMallocAsync`/`cudaFreeAsync` are not available on older CUDA versions. Systems with CUDA < 11.2 must fall back to direct allocation.

**Pool Reservation Opacity**: `cudaMemGetInfo` reports misleadingly low free memory because pool reservations are counted as "used" by the driver. This requires all memory budget calculations to account for pool statistics, not just driver-reported free memory.

**Stream Mismatch Risk**: Memory freed on one stream can only be reused by allocations on the same stream (without explicit synchronization). If DSP frees on the execution stream but C++ allocates on the null stream, the pool cannot reuse freed memory, leading to excessive OOM recoveries. This must be addressed by ensuring stream consistency between Java and C++ execution paths.

## References

- CUDA Memory Management documentation (cudaMallocAsync/cudaFreeAsync)
- CUDA 11.2 release notes on stream-ordered memory allocation
- CudaMemoryPool.h, CudaMemoryPool.cu in libnd4j/include/memory/cuda/
- CoreConfig.h/cpp — `_cudaSoftLimitPercent` field and `SD_CUDA_SOFT_LIMIT_PERCENT` env var
- Environment.h — `cudaSoftLimitPercent()` / `setCudaSoftLimitPercent()` forwarding
- ADR 0065 — Multi-GPU Memory Management (failover chain that soft limit triggers into)
