# ADR: Multi-GPU Memory Management

## Status

Implemented

Proposed by: Adam Gibson (January 2025)

Discussed with: Development Team

## Context

Multi-GPU systems present several memory management challenges that single-GPU code does not encounter:

**Device Selection**: When loading a large model, which GPU should receive the constants? Naively choosing the GPU with the most free memory is wrong — CUDA memory pool reservations reduce reported free memory without actually consuming it. A GPU with 2GB "free" may have 20GB of reusable pool capacity.

**Cross-Device Access**: When an array on GPU 0 is needed by an op on GPU 1, the data must be transferred. With NVLink/PCIe peer access, GPU 1 can read GPU 0's memory directly. Without peer access (common for consumer GPUs), data must be staged through host memory (D2H + H2D).

**OOM Failover**: When the primary GPU runs out of memory, the framework should failover to other GPUs rather than crash. But failover to non-P2P GPUs requires host-staged transfers for every access, potentially causing 100x slowdowns if used for compute-intensive ops.

**View Replication**: Arrays that are views of other arrays cannot be replicated directly — they reference a parent buffer at an offset. Replicating requires first creating a contiguous copy, then transferring the copy to the target device.

**Memory Leaks**: Cross-device operations create intermediate arrays that must be freed on the correct device. Device context switches between allocation and deallocation can cause frees to target the wrong device, leaking memory silently.

## Decision

We implement a comprehensive multi-GPU memory management system with total-memory-based device selection, multi-stage failover, stream-safe cross-device migration, and P2P-aware compute budget allocation.

### Device Selection Strategy

```java
public static int selectBestGpu() {
    int bestDevice = 0;
    long bestTotal = 0;
    for (int i = 0; i < numDevices; i++) {
        long totalMem = Nd4j.getDeviceProperties(i).getTotalGlobalMem();
        if (totalMem > bestTotal) {
            bestTotal = totalMem;
            bestDevice = i;
        }
    }
    return bestDevice;
}
```

**Total memory is the primary metric, not free memory.** Rationale:
- CUDA memory pool reservations reduce reported free memory without blocking allocation
- The GPU with the largest total memory can accommodate the largest model
- Free memory is misleading during startup when pools haven't been populated yet

### Multi-Stage Failover

When `CudaMemoryPool::allocate()` fails on the current device:

```
Stage 1: trimPool(currentDevice) → retry on same device
    ↓ (still fails)
Stage 2: Try peer-accessible GPUs, sorted by free memory
    ↓ (all P2P GPUs full)
Stage 3: Try non-P2P GPUs, sorted by free memory
    ↓ (all GPUs full)
Stage 4: cudaMallocHost → pinned host memory (accessible from all GPUs)
    ↓ (host memory full)
Stage 5: Raise OOM error
```

**Critical**: Stage 2 sorts candidates peer-first, then by free memory. Non-P2P GPUs are included but ranked lower because host-staged transfers add latency.

**Critical**: `allocateFailover()` tracks `actualMigrateDevice` — the device where memory was actually allocated. This is stored in the DataBuffer so that `migrate()` and `free()` target the correct device.

### P2P Access Detection and Compute Budget

At startup, the system probes all device pairs for P2P access:

```cpp
for (int i = 0; i < numDevices; i++) {
    for (int j = 0; j < numDevices; j++) {
        int canAccess;
        cudaDeviceCanAccessPeer(&canAccess, i, j);
        peerAccessEnabled_[i][j] = (canAccess != 0);
        if (canAccess) cudaDeviceEnablePeerAccess(j, 0);
    }
}
```

**Non-P2P compute budget**: 0% by default. Non-P2P secondary GPUs are NOT assigned compute ops because every data access requires a host-staged round-trip, causing 100x slowdowns. A 30% budget was tested and caused emergency reclaim cycles on every OOMing op.

Non-P2P GPUs are still valuable for **memory spillover** — constants and infrequently-accessed buffers can be placed there via `allocateFailover`, with host-staged transfers occurring only when the data is actually needed.

Override via: `-Dnd4j.dsp.nonP2pBudgetFraction=0.3` (not recommended for most workloads).

### Cross-Device Replication

`CudaAffinityManager.replicateToDevice()` handles the complexities of cross-device array transfer:

```java
public INDArray replicateToDevice(int targetDevice, INDArray source) {
    if (source.isView()) {
        // Views can't be replicated directly — create contiguous copy first
        int sourceDevice = getDeviceForArray(source);
        switchDevice(sourceDevice);          // Switch to source device for dup
        INDArray contiguous = source.dup();  // Contiguous copy on source device

        // Transfer contiguous copy to target device
        INDArray result = transferToDevice(targetDevice, contiguous);
        contiguous.close();  // CRITICAL: close intermediate to prevent leak
        return result;
    }
    return transferToDevice(targetDevice, source);
}
```

**Critical Fix**: The intermediate `contiguous` array from `dup()` was previously leaked because GC-based cleanup is broken (PhantomRef strong reference cycle). Explicit `contiguous.close()` after transfer prevents ~30MB/step leaks for VLM KV cache views.

### Device-Safe Free

`CudaMemoryPool::free()` saves and restores the current device to ensure frees target the correct device:

```cpp
void CudaMemoryPool::free(void* ptr, size_t size, int deviceId) {
    int savedDevice;
    cudaGetDevice(&savedDevice);
    if (savedDevice != deviceId) {
        cudaSetDevice(deviceId);
    }
    cudaFreeAsync(ptr, stream);
    poolUsed[deviceId] -= size;
    if (savedDevice != deviceId) {
        cudaSetDevice(savedDevice);
    }
}
```

Without device save/restore, frees during failover would target the wrong device's pool, causing double-free crashes or silent memory leaks.

### ConstantHelper Pinned Host Fallback

When `CudaMemoryPool::allocate()` returns a buffer on the wrong device (non-P2P), shape buffers used by `ConstantHelper` fall back to pinned host memory:

```cpp
void* ConstantHelper::replicatePointer(void* src, size_t size) {
    void* ptr = CudaMemoryPool::allocate(size, targetDevice);
    if (getDeviceForPointer(ptr) != targetDevice && !peerAccessEnabled_[current][target]) {
        CudaMemoryPool::free(ptr);
        cudaMallocHost(&ptr, size);  // Pinned host — accessible from ALL devices
    }
    return ptr;
}
```

Pinned host memory is slower than device memory but accessible from all GPUs via PCIe, avoiding the need for per-access host staging.

### ContextBuffers Device-Aware Initialization

`ContextBuffers::initialize()` trims the pool BEFORE allocating reduction and allocation pointers:

```cpp
void ContextBuffers::initialize() {
    CudaMemoryPool::trimPool(currentDevice);  // Free reserved-but-unused memory
    // Now allocate reduction buffer, allocation pointer, etc.
    cudaMallocAsync(&_reductionBuffer, REDUCTION_SIZE, stream);
}
```

This prevents the paradox where pool reservations from previous contexts block allocation of new context buffers.

## Consequences

### Advantages

**Zero-Crash OOM Handling**: Four-stage failover ensures allocations never fail unless ALL GPUs AND host memory are exhausted. The framework degrades gracefully from fast device memory to slower host memory.

**Efficient Device Selection**: Total-memory-based selection correctly identifies the best GPU for model loading regardless of pool reservation state.

**Transparent Cross-Device Access**: `replicateToDevice` handles views, device switching, and intermediate cleanup. Callers don't need to understand P2P topologies.

**Non-P2P GPU Utilization**: Secondary GPUs without NVLink still serve as memory overflow capacity, extending effective GPU memory beyond the primary device.

### Disadvantages

**Non-P2P Latency**: Host-staged transfers (D2H + H2D) for non-P2P devices add significant latency. This is acceptable for infrequent constant access but prohibitive for per-op intermediate transfers.

**Complexity**: Device save/restore, P2P detection, failover tracking, and view handling add substantial complexity. Bugs in any of these paths cause silent memory leaks or cross-device corruption.

**Pool Statistics Fragmentation**: When allocations span multiple devices, pool statistics per device may not reflect actual memory pressure. Global memory budgeting requires aggregating across all devices.

## References

- CudaMemoryPool.h, CudaMemoryPool.cu
- CudaAffinityManager.java
- DeviceMemoryManager.java
- ConstantHelper.cu
- ContextBuffers.cu
- ADR 0060 - CUDA Async Memory Pool
