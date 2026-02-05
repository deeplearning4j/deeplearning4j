# ADR 0059 - Multi-Backend Op Execution System

## Status
Proposed

## Context

ND4J/libnd4j currently supports multiple compute backends (CPU, CUDA, ZLUDA, etc.), but traditionally only one backend is active at a time. The backend is selected at build time and runtime based on classpath configuration. This creates limitations:

1. **No dynamic device routing**: Once a backend is loaded, all ops run on that backend
2. **No automatic data transfer**: Arrays on different devices require manual management
3. **No runtime backend switching**: Cannot leverage both CPU and GPU dynamically
4. **Op helper compilation is exclusive**: CPU and CUDA op helpers are linked separately

For true multi-device operation (e.g., running LLM inference across multiple GPUs, or offloading to CPU when GPU memory is constrained), we need:
- Runtime loading of multiple backends
- Automatic routing of ops based on input data location
- Transparent cross-device data transfer

## Decision

We implement a **Multi-Backend Op Execution System** with these components:

### 1. MultiBackendNativeOpsHolder

Location: `nd4j/nd4j-backends/nd4j-api-parent/nd4j-api/src/main/java/org/nd4j/nativeblas/MultiBackendNativeOpsHolder.java`

Loads and manages multiple `NativeOps` implementations simultaneously:
- Attempts to load both `Nd4jCpu` and `Nd4jCuda` at runtime
- Maps device types to their NativeOps implementations
- Falls back gracefully when a backend isn't available

```java
// Enable multi-backend mode
MultiBackendNativeOpsHolder.enableMultiBackend();

// Get ops for a specific device
NativeOps cpuOps = MultiBackendNativeOpsHolder.getInstance()
    .getOpsForDeviceType(DeviceType.CPU);
NativeOps gpuOps = MultiBackendNativeOpsHolder.getInstance()
    .getOpsForDeviceType(DeviceType.CUDA_GPU);
```

### 2. BackendRoutingStrategy

Location: `nd4j/nd4j-backends/nd4j-api-parent/nd4j-api/src/main/java/org/nd4j/linalg/api/ops/executioner/BackendRoutingStrategy.java`

Interface defining op routing logic:
- `selectTargetDevice(Op op)` - Determine optimal execution device
- `getNativeOpsForDevice(DeviceDescriptor)` - Get appropriate NativeOps
- `ensureOnDevice(INDArray, DeviceDescriptor)` - Transfer array to target device

Default implementation (`DefaultBackendRoutingStrategy`) routes ops based on:
- Device location of input arrays (majority vote)
- Available device memory
- User-configured device preferences

### 3. DeviceAwareOpExecutioner Enhancements

Location: `nd4j/nd4j-backends/nd4j-api-parent/nd4j-api/src/main/java/org/nd4j/linalg/api/ops/executioner/DeviceAwareOpExecutioner.java`

Enhanced wrapper that:
- Maintains a map of `DeviceType -> OpExecutioner`
- Routes each op execution to the appropriate backend executioner
- Automatically transfers inputs to target device before execution

```java
// Register multiple backend executioners
DeviceAwareOpExecutioner.getInstance()
    .registerBackendExecutioner(DeviceType.CPU, cpuExecutioner);
DeviceAwareOpExecutioner.getInstance()
    .registerBackendExecutioner(DeviceType.CUDA_GPU, cudaExecutioner);

// Now ops automatically route to the correct backend
INDArray gpuArray = DeviceAwareNd4j.createOnGpu(new long[]{1000, 1000});
INDArray cpuArray = Nd4j.create(new float[]{1, 2, 3});

// This routes to GPU (gpu array is larger) and auto-transfers cpuArray
INDArray result = gpuArray.add(cpuArray);
```

### 4. Existing Infrastructure Leveraged

The system builds on existing components:

- **DeviceMemoryManager**: Tracks per-device memory allocation and caps
- **DeviceRoutingConfiguration**: Configures routing policies and preferences
- **OpExecutionDelegator**: Handles automatic data transfer between devices
- **HybridDataBuffer**: Dual-buffer architecture for CPU/GPU coherence

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                    User Code                                 │
│           INDArray result = a.add(b);                       │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│              DeviceAwareOpExecutioner                       │
│  1. Determine target device from inputs                     │
│  2. Transfer inputs to target device                        │
│  3. Select appropriate backend executioner                  │
│  4. Execute op                                              │
└─────────────────────────┬───────────────────────────────────┘
                          │
              ┌───────────┴───────────┐
              │                       │
              ▼                       ▼
┌─────────────────────┐   ┌─────────────────────┐
│  NativeOpExecutioner │   │   CudaExecutioner   │
│     (CPU Backend)    │   │   (CUDA Backend)    │
└──────────┬──────────┘   └──────────┬──────────┘
           │                         │
           ▼                         ▼
┌─────────────────────┐   ┌─────────────────────┐
│      Nd4jCpu        │   │      Nd4jCuda       │
│   (nd4jcpu.so)      │   │   (nd4jcuda.so)     │
└─────────────────────┘   └─────────────────────┘
```

## Build System Considerations

The current build system creates separate native libraries:
- `nd4jcpu.so` - CPU op helpers only
- `nd4jcuda.so` - CUDA op helpers only

For multi-backend to work at runtime, both libraries must be:
1. Built separately (current approach)
2. Available on the Java classpath
3. Loaded dynamically via `MultiBackendNativeOpsHolder`

This avoids the complexity of:
- Merging both backends into one library (symbol conflicts)
- Modifying C++ build to compile both helper sets together

## Usage

### Enabling Multi-Backend

```java
// Simple: Enable device routing (loads available backends)
DeviceAwareNd4j.enableDeviceRouting();

// Advanced: Configure routing behavior
DeviceRoutingConfiguration config = DeviceRoutingConfiguration.builder()
    .defaultPolicy(DeviceRoutingPolicy.PREFER_GPU)
    .autoTransferEnabled(true)
    .gpuMemoryCapFraction(0.9)
    .build();
DeviceAwareNd4j.enableDeviceRouting(config);
```

### Creating Device-Specific Arrays

```java
// Create on specific GPU
INDArray gpuArray = DeviceAwareNd4j.createOnGpu(new long[]{1000, 1000});

// Create on CPU
INDArray cpuArray = DeviceAwareNd4j.createOnCpu(new long[]{1000, 1000});

// Auto-routed based on available memory
INDArray autoArray = DeviceAwareNd4j.createRouted(new long[]{1000, 1000});
```

### Cross-Device Operations

```java
// Automatic transfer - cpuArray copied to GPU, op runs on GPU
INDArray result = gpuArray.add(cpuArray);

// Explicit transfer
DeviceAwareNd4j.transferToGpu(cpuArray);
```

## Consequences

### Positive
- True multi-device operation without manual data management
- Graceful fallback when backends unavailable
- Leverages existing infrastructure
- No changes to core op implementations required
- Preserves backward compatibility (opt-in feature)

### Negative
- Some runtime overhead for device detection per op
- Requires both native libraries on classpath for full multi-backend
- Initial data transfer latency for cross-device ops
- Memory overhead for dual-buffer arrays

### Trade-offs
- Chose runtime library loading over unified build (simpler, no symbol conflicts)
- Chose wrapper pattern over modifying core executioners (less invasive)
- Chose automatic transfer over explicit-only (convenience vs control)

## Related ADRs
- ADR 0055 - Kernel Selection and Dynamic Loading
- ADR 0057 - Multi-Backend Workspace System
- ADR 0058 - Multi-Backend Kernel Selection and Management

## Future Work

1. **C++ DataTransferManager exposure**: Expose native P2P transfer functions to Java via JNI
2. **Unified kernel registry**: Runtime selection of CPU vs CUDA op helpers per-op
3. **Multi-GPU P2P transfers**: Direct GPU-to-GPU transfers without host staging
4. **Prefetch optimization**: Predictive data transfer based on execution graph analysis
