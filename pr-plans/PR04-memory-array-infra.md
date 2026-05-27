# PR04: Memory Management & Array Infrastructure

**Estimated files:** ~64
**Merge layer:** 1
**Complexity:** High — core data structures, CUDA memory pool, workspace system
**Reviewers:** Core C++ team, memory safety review

## Description

NDArray, DataBuffer, memory allocator, workspace system, CUDA memory pool,
indexing utilities. These are the foundational data structures used by
every op and every execution path. Changes here affect correctness and
performance across the project.

## Files (64)

### Array headers (~16)
- `libnd4j/include/array/AllocationLogger.h`
- `libnd4j/include/array/ArrayOptions.h`
- `libnd4j/include/array/ArrayOptions.hXX`
- `libnd4j/include/array/ConstantShapeBuffer.h`
- `libnd4j/include/array/DataBuffer.h`
- `libnd4j/include/array/DataBufferLifecycleTracker.h`
- `libnd4j/include/array/DataType.h`
- `libnd4j/include/array/DataTypeUtils.h`
- `libnd4j/include/array/InteropDataBuffer.h`
- `libnd4j/include/array/NDArray.h`
- `libnd4j/include/array/NDArray.hXX`
- `libnd4j/include/array/NDArrayFactory.h`
- `libnd4j/include/array/NDArrayHelpers.hXX`
- `libnd4j/include/array/NDArrayLifecycleTracker.h`
- `libnd4j/include/array/PointerWrapper.h`
- `libnd4j/include/array/ShapeDescriptor.h`
- `libnd4j/include/array/ShapeList.h`
- `libnd4j/include/array/TadCalculator.h`
- `libnd4j/include/array/TadPack.h`

### Array CPU implementations (3)
- `libnd4j/include/array/cpu/DataBuffer.cpp`
- `libnd4j/include/array/cpu/NDArray.cpp`
- `libnd4j/include/array/cpu/NDArrayLambda.hpp`

### Array CUDA implementations (8)
- `libnd4j/include/array/cuda/CudaPointerDeallocator.cu`
- `libnd4j/include/array/cuda/DataBuffer.cu`
- `libnd4j/include/array/cuda/ExtraArgumentsCuda.cu`
- `libnd4j/include/array/cuda/NDArray.cu`
- `libnd4j/include/array/cuda/NDArray_core.cu`
- `libnd4j/include/array/cuda/NDArrayLambda.cu`
- `libnd4j/include/array/cuda/NDArray_print.cu`
- `libnd4j/include/array/cuda/NDArray_repeat.cu`
- `libnd4j/include/array/cuda/NDArray_tile.cu`
- `libnd4j/include/array/cuda/NDArray_triangular.cu`

### Array shared implementations (11)
- `libnd4j/include/array/impl/ConstantHolder.cpp`
- `libnd4j/include/array/impl/ConstantShapeBuffer.cpp`
- `libnd4j/include/array/impl/DataBuffer.cpp`
- `libnd4j/include/array/impl/DataTypeUtils.cpp`
- `libnd4j/include/array/impl/ExtraArguments.cpp`
- `libnd4j/include/array/impl/InteropDataBuffer.cpp`
- `libnd4j/include/array/impl/NDArrayFactory.cpp`
- `libnd4j/include/array/impl/NDArrayList.cpp`
- `libnd4j/include/array/impl/PrimaryPointerDeallocator.cpp`
- `libnd4j/include/array/impl/ShapeDescriptor.cpp`
- `libnd4j/include/array/impl/TadPack.cpp`

### Data type validation (2)
- `libnd4j/include/array/DataTypeConversions.h`
- `libnd4j/include/array/DataTypeValidation.cpp`
- `libnd4j/include/array/DataTypeValidation.h`

### Memory subsystem headers (5)
- `libnd4j/include/memory/DeviceWorkspaceManager.h`
- `libnd4j/include/memory/MemoryUtils.h`
- `libnd4j/include/memory/MultiBackendWorkspace.h`
- `libnd4j/include/memory/Workspace.h`

### Memory CPU implementations (3)
- `libnd4j/include/memory/cpu/DeviceWorkspaceManager.cpp`
- `libnd4j/include/memory/cpu/MultiBackendWorkspace.cpp`
- `libnd4j/include/memory/cpu/Workspace.cpp`

### Memory CUDA implementations (4)
- `libnd4j/include/memory/cuda/CudaMemoryPool.cu`
- `libnd4j/include/memory/cuda/CudaMemoryPool.h`
- `libnd4j/include/memory/cuda/MultiBackendWorkspace.cu`
- `libnd4j/include/memory/cuda/Workspace.cu`

### Memory shared implementations (3)
- `libnd4j/include/memory/impl/MemoryTracker.cpp`
- `libnd4j/include/memory/impl/MemoryUtils.cpp`
- `libnd4j/include/memory/impl/padded_allocator.cpp`

### Memory JNI bridge (1)
- `libnd4j/include/memory/MultiBackendWorkspaceJni.cpp`

### Indexing (2)
- `libnd4j/include/indexing/NDIndexUtils.h`
- `libnd4j/include/indexing/impl/NDIndexUtils.cpp`

### ADRs (6 — only those actually changed in the diff)
- `ADRs/0028 - Offset centralization.md` — Centralize offset storage into NDArray, introduce OpaqueNDArray
- `ADRs/0057 - Multi-Backend Workspace System.md` — Multi-device workspace tracking with MSI-style coherence (duplicate 0057, needs renumbering)
- `ADRs/0060 - CUDA Async Memory Pool.md` — cudaMallocAsync-based pooling with multi-GPU OOM failover
- `ADRs/0063 - ArrayCacheMemoryMgr Buffer Reuse.md` — Capacity-indexed TreeMap with LRU eviction, closeable-gate leak fix
- `ADRs/0065 - Multi-GPU Memory Management.md` — Total-memory device selection, multi-stage OOM failover, P2P budgeting
- `ADRs/0070 - GC Pressure Optimization.md` — Heap-pressure-aware conditional GC, PhantomReference cycle fix

## Review Focus

- CUDA memory pool changes (CudaMemoryPool.cu) — affects OOM failover
- DataBuffer lifecycle tracking — affects shutdown/dealloc ordering
- Workspace allocation changes — affects DSP memory
