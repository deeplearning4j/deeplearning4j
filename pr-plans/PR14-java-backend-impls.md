# PR14: Java Backend Implementations (CUDA + CPU)

**Estimated files:** ~64
**Merge layer:** 4
**Complexity:** High — backend-specific allocators, executioners
**Reviewers:** Backend team, CUDA specialists

## Description

CUDA and CPU backend Java implementations: allocators, memory managers,
executioners, data buffer factories, TAD managers, affinity managers,
workspace implementations, and backend registration.

## CUDA Backend (40 files)

### Core
- `nd4j-cuda/src/main/java9/module-info.java`
- `nd4j-cuda/src/main/java/org/nd4j/linalg/jcublas/JCublasBackend.java`
- `nd4j-cuda/src/main/java/org/nd4j/linalg/jcublas/CudaEnvironment.java`
- `nd4j-cuda/src/main/resources/nd4j-jcublas.properties`

### JITA allocator
- `AllocationPoint.java`
- `AtomicAllocator.java`
- `CudaDeallocator.java`

### TAD/affinity/memory
- `BasicTADManager.java`
- `DeviceTADManager.java`
- `CudaAffinityManager.java`
- `CudaDeviceContextProvider.java`
- `Configuration.java`
- `CudaZeroHandler.java`
- `MemoryHandler.java`
- `CudaMemoryManager.java`
- `CudaWorkspace.java`

### BLAS
- `JcublasLapack.java`
- `JcublasLevel1.java`

### Data buffers (13 types)
- `BaseCudaDataBuffer.java`
- `CudaFloatDataBuffer.java`
- `CudaDoubleDataBuffer.java`
- `CudaHalfDataBuffer.java`
- `CudaBfloat16DataBuffer.java`
- `CudaIntDataBuffer.java`
- `CudaLongDataBuffer.java`
- `CudaByteDataBuffer.java`
- `CudaUByteDataBuffer.java`
- `CudaBoolDataBuffer.java`
- `CudaShortDataBuffer.java`
- `CudaUInt16DataBuffer.java`
- `CudaUInt32DataBuffer.java`
- `CudaUInt64DataBuffer.java`
- `CudaUtf8Buffer.java`
- `CudaDataBufferFactory.java`

### NDArray/executioner
- `JCublasNDArrayFactory.java`
- `JCublasNDArray.java`
- `CudaExecutioner.java`
- `CudaOpContext.java`
- `CudaOpContextDeallocator.java`
- `CudaContext.java`

## CPU Backend (24 files)

### nd4j-cpu-backend-common (16)
- `module-info.java`
- `CpuLapack.java`
- `CpuLevel1.java`
- `BaseCpuDataBuffer.java`
- `CpuDeallocator.java`
- `LongBuffer.java`
- `ConstantBuffersCache.java`
- `CpuAffinityManager.java`
- `CpuMemoryManager.java`
- `CpuNDArrayFactory.java`
- `CpuTADManager.java`
- `DirectShapeInfoProvider.java`
- `CpuOpContext.java`
- `CpuOpContextDeallocator.java`
- `NativeOpExecutioner.java`
- `CpuWorkspace.java`

### nd4j-native (8)
- `CpuBackend.java`
- `CpuEnvironment.java`
- `CpuStatisticsProvider.java`
- `nd4j-native.properties`
- Native-image config JSONs (4)

## Review Focus

- CudaExecutioner changes — affects all CUDA op dispatch
- AtomicAllocator — CUDA memory lifecycle
- CudaMemoryManager — OOM handling, failover
- CpuWorkspace/CudaWorkspace — workspace allocation strategy
