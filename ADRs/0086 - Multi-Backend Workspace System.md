# Multi-Backend Workspace System

## Status

Implemented

Proposed by: Adam Gibson (31 Dec 2024)

Discussed with: Claude AI Assistant

## Context

Modern deep learning workloads increasingly require execution across multiple compute devices (CPUs, GPUs, TPUs) simultaneously. The existing workspace system (ADR-0024) provides excellent memory reuse within a single device context, but lacks support for:

1. **Multi-device memory management** - Tracking memory allocations across CPU and GPU simultaneously
2. **Memory coherence** - Ensuring data consistency when the same logical buffer exists on multiple devices
3. **Cross-device transfers** - Efficient data movement between devices with proper synchronization
4. **Device-aware allocation policies** - Routing allocations to optimal devices based on workload characteristics

This ADR extends the workspace concept to support hybrid multi-backend execution with proper memory coherence tracking.

## Decision

We introduce a **Multi-Backend Workspace System** that extends the existing workspace architecture with:

1. **Device abstraction layer** - Unified interface for different compute devices
2. **MSI-like coherence protocol** - Cache coherence for multi-device memory
3. **Device-aware workspace configuration** - Per-device allocation policies
4. **Native C++ implementation** - High-performance native code with opaque Java bindings
5. **Automatic resource cleanup** - Java Cleaner-based deallocation

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         Java Layer                               │
├─────────────────────────────────────────────────────────────────┤
│  Nd4j.getMultiBackendWorkspaceManager()                         │
│       │                                                          │
│       ▼                                                          │
│  MultiBackendWorkspaceManager (Singleton)                        │
│       │                                                          │
│       ├── Thread-local workspace registry                        │
│       ├── Global workspace registry                              │
│       └── Default configuration                                  │
│              │                                                   │
│              ▼                                                   │
│  MultiBackendWorkspace (Interface)                               │
│       │                                                          │
│       ├── DefaultMultiBackendWorkspace (Pure Java)               │
│       └── NativeMultiBackendWorkspace (JNI → C++)                │
│                     │                                            │
├─────────────────────┼────────────────────────────────────────────┤
│                     ▼           Native Layer                     │
│  ┌─────────────────────────────────────────────┐                │
│  │   MultiBackendWorkspace (C++)                │                │
│  │       │                                      │                │
│  │       ├── Per-device Workspace instances     │                │
│  │       ├── CoherenceState tracking            │                │
│  │       └── Transfer management                │                │
│  └─────────────────────────────────────────────┘                │
│                     │                                            │
│                     ▼                                            │
│  ┌─────────────────────────────────────────────┐                │
│  │   DeviceWorkspaceManager (C++)               │                │
│  │       │                                      │                │
│  │       ├── Thread-local workspace map         │                │
│  │       ├── Global workspace registry          │                │
│  │       └── Lifecycle management               │                │
│  └─────────────────────────────────────────────┘                │
└─────────────────────────────────────────────────────────────────┘
```

## Memory Coherence Protocol

The system implements an MSI-like (Modified-Shared-Invalid) cache coherence protocol:

| State | Description | Can Read | Can Write |
|-------|-------------|----------|-----------|
| **EXCLUSIVE** | Only copy, unmodified | Yes | Yes |
| **SHARED** | Multiple copies exist, all valid | Yes | No |
| **MODIFIED** | Only copy, has been written | Yes | Yes |
| **INVALID** | Data is stale/not present | No | No |

### State Transitions

```
┌──────────────────────────────────────────────────────────────┐
│                                                               │
│    ┌─────────┐    write     ┌──────────┐                     │
│    │ SHARED  │─────────────▶│ MODIFIED │                     │
│    └────┬────┘              └────┬─────┘                     │
│         │                        │                            │
│    read │                        │ other device reads         │
│    (no  │                        │ (transfer + invalidate)    │
│    copy)│                        │                            │
│         ▼                        ▼                            │
│    ┌─────────┐   allocate   ┌─────────┐                      │
│    │EXCLUSIVE│◀─────────────│ INVALID │                      │
│    └─────────┘              └─────────┘                      │
│         │                        ▲                            │
│         │   other device         │                            │
│         │   reads (share)        │ invalidate()               │
│         └────────────────────────┘                            │
│                                                               │
└──────────────────────────────────────────────────────────────┘
```

## Description

### Device Abstraction

```cpp
// C++ Device Types
enum class DeviceType : int {
    CPU = 0,
    CUDA_GPU = 1,
    ROCM_GPU = 2,
    TPU = 3,
    UNKNOWN = 99
};

// Device Descriptor
struct DeviceDescriptor {
    DeviceType deviceType;
    int deviceIndex;
    std::string deviceId;  // e.g., "cpu:0", "gpu:0"
};
```

```java
// Java Device Types
public enum DeviceType {
    CPU("cpu", true),
    CUDA_GPU("cuda", true),
    ROCM_GPU("rocm", false),
    TPU("tpu", false),
    // ...
}
```

### Configuration

The `DeviceAwareWorkspaceConfiguration` extends the base `WorkspaceConfiguration`:

```java
DeviceAwareWorkspaceConfiguration config = DeviceAwareWorkspaceConfiguration.builder()
    .initialSize(10 * 1024 * 1024)           // 10 MB initial
    .maxSize(100 * 1024 * 1024)              // 100 MB max
    .crossDeviceMirroring(true)              // Keep copies on multiple devices
    .asyncTransfers(true)                     // Use async device transfers
    .transferPolicy(TransferPolicy.ON_DEMAND) // Transfer when needed
    .deviceSelectionPolicy(DeviceSelectionPolicy.FIRST_AVAILABLE)
    .preferredDeviceTypes(Arrays.asList(DeviceType.CUDA_GPU, DeviceType.CPU))
    .build();
```

#### Transfer Policies

| Policy | Description |
|--------|-------------|
| `ON_DEMAND` | Transfer data when accessed on different device |
| `EAGER` | Pre-transfer to all configured devices at allocation |
| `NONE` | Never transfer - fail if wrong device accesses |

#### Device Selection Policies

| Policy | Description |
|--------|-------------|
| `FIRST_AVAILABLE` | Use first available device of preferred type |
| `MOST_FREE_MEMORY` | Use device with most available memory |
| `ROUND_ROBIN` | Cycle through devices |
| `EXPLICIT` | Use explicitly specified device |

### Usage Examples

#### Basic Usage

```java
// Get the workspace manager
MultiBackendWorkspaceManager manager = Nd4j.getMultiBackendWorkspaceManager();

// Create a GPU-preferred workspace
MultiBackendWorkspace ws = manager.createWorkspace(
    DeviceAwareWorkspaceConfiguration.gpuPreferred(1024 * 1024),
    "training_workspace"
);

// Use with try-with-resources
try (MultiBackendWorkspace scoped = ws) {
    scoped.notifyScopeEntered();

    // Allocate memory
    PagedPointer ptr = scoped.alloc(256, DataType.FLOAT, true);

    // Memory is automatically managed
} // Scope exits, memory can be reused

// Destroy when done
manager.destroyWorkspace(ws);
```

#### Cross-Device Mirroring

```java
// Create a mirrored workspace (data on both CPU and GPU)
MultiBackendWorkspace ws = Nd4j.createMirroredWorkspace(10 * 1024 * 1024, "mirrored_ws");

// Allocate - data available on both devices
PagedPointer ptr = ws.allocOnDevice(1024, DataType.FLOAT, gpuDevice, true);

// Transfer to CPU for host-side processing
ws.transferTo(gpuDevice, cpuDevice);

// Check coherence state
CoherenceState state = ws.getCoherenceState(cpuDevice, 0);
// state == CoherenceState.SHARED (both have valid copies)
```

#### Using BackendRegistry

```java
BackendRegistry registry = BackendRegistry.getInstance();

// Create workspace via registry
MultiBackendWorkspace ws = registry.createWorkspace(
    registry.gpuPreferredWorkspaceConfig(1024 * 1024),
    "registry_workspace"
);

// Get memory statistics
Map<String, Long> perDeviceMemory = registry.getWorkspaceMemoryPerDevice();
```

#### Static Nd4j API

```java
// Convenience methods on Nd4j class
MultiBackendWorkspace gpuWs = Nd4j.createGpuPreferredWorkspace(1024 * 1024, "gpu_ws");
MultiBackendWorkspace cpuWs = Nd4j.createCpuOnlyWorkspace(1024 * 1024, "cpu_ws");
MultiBackendWorkspace mirroredWs = Nd4j.createMirroredWorkspace(1024 * 1024, "mirrored_ws");

// Get-or-create pattern
DeviceAwareWorkspaceConfiguration config = DeviceAwareWorkspaceConfiguration.builder()
    .initialSize(1024 * 1024)
    .build();
MultiBackendWorkspace ws = Nd4j.getOrCreateMultiBackendWorkspace(config, "my_workspace");
```

### Native Implementation

The C++ implementation provides high-performance workspace management:

```cpp
// Create workspace
MultiBackendWorkspaceConfig config;
config.initialSize = 1024 * 1024;
config.primaryDevice = DeviceDescriptor(DeviceType::CPU, 0);

MultiBackendWorkspace* ws = new MultiBackendWorkspace(config, "native_ws");

// Allocate
void* ptr = ws->allocateBytes(256);

// Scope management
ws->scopeIn();
// ... allocations ...
ws->scopeOut();  // Offsets reset, memory reusable

// Cleanup
ws->destroy();
delete ws;
```

#### C-Style API for Bindings

```cpp
// Opaque handle type
typedef MultiBackendWorkspace* MultiBackendWorkspaceHandle;

// C API functions
MultiBackendWorkspaceHandle createMultiBackendWorkspace(
    sd::LongType initialSize,
    int primaryDeviceType,
    int primaryDeviceIndex);

void destroyMultiBackendWorkspace(MultiBackendWorkspaceHandle handle);

void* mbwAllocateBytes(MultiBackendWorkspaceHandle handle, sd::LongType numBytes);

void mbwScopeIn(MultiBackendWorkspaceHandle handle);
void mbwScopeOut(MultiBackendWorkspaceHandle handle);

int mbwGetCoherenceState(MultiBackendWorkspaceHandle handle, int deviceType, int deviceIndex);
void mbwMarkModified(MultiBackendWorkspaceHandle handle, int deviceType, int deviceIndex);
```

### Deallocation Strategy

The system uses multiple deallocation mechanisms:

1. **Explicit destruction**: `workspace.destroyWorkspace()` or `manager.destroyWorkspace(ws)`
2. **Scope-based cleanup**: Memory reused when scope exits
3. **Thread cleanup**: `manager.destroyAllWorkspacesForCurrentThread()`
4. **Java Cleaner**: Automatic cleanup when workspace becomes unreachable

```java
// NativeMultiBackendWorkspace uses Java Cleaner
private static final Cleaner cleaner = Cleaner.create();

private static class NativeCleanerAction implements Runnable {
    private final long handle;

    @Override
    public void run() {
        if (handle != 0) {
            nativeDestroy(handle);  // Call C++ destructor
        }
    }
}

// Registered on construction
this.cleanable = cleaner.register(this, new NativeCleanerAction(nativeHandle));
```

### File Structure

#### Java Files

| File | Description |
|------|-------------|
| `DeviceAwareWorkspaceConfiguration.java` | Extended configuration with device awareness |
| `MultiBackendWorkspace.java` | Interface for multi-backend workspaces |
| `MultiBackendWorkspaceManager.java` | Thread-local workspace management |
| `DefaultMultiBackendWorkspace.java` | Pure Java implementation |
| `NativeMultiBackendWorkspace.java` | JNI-backed implementation |

#### C++ Files

| File | Description |
|------|-------------|
| `MultiBackendWorkspace.h` | Main workspace class and C API |
| `MultiBackendWorkspace.cpp` | CPU implementation |
| `DeviceWorkspaceManager.h` | Global manager |
| `DeviceWorkspaceManager.cpp` | Manager implementation |
| `MultiBackendWorkspaceJni.cpp` | JNI bridge |

## Consequences

### Advantages

* **Multi-device support**: Seamless memory management across CPU, GPU, and other accelerators
* **Memory coherence**: Automatic tracking of data validity across devices
* **Efficient transfers**: On-demand or eager transfer policies based on workload
* **Backward compatible**: Existing workspace code continues to work unchanged
* **Native performance**: C++ implementation with minimal JNI overhead
* **Automatic cleanup**: Multiple deallocation strategies prevent leaks
* **Thread safety**: All operations are thread-safe with proper locking

### Disadvantages

* **Increased complexity**: More configuration options and concepts to understand
* **Memory overhead**: Coherence tracking requires additional metadata
* **Transfer latency**: Cross-device transfers add latency when data moves
* **Learning curve**: Developers must understand coherence states and device policies

### Migration Path

Existing code using `WorkspaceConfiguration` and `MemoryWorkspace` continues to work. To adopt multi-backend features:

1. Replace `WorkspaceConfiguration` with `DeviceAwareWorkspaceConfiguration`
2. Use `MultiBackendWorkspaceManager` instead of `MemoryWorkspaceManager`
3. Add device-specific allocation calls where needed
4. Configure transfer policies based on workload characteristics

## Testing

Comprehensive tests are provided:

* **C++ tests**: `libnd4j/tests_cpu/layers_tests/MultiBackendWorkspaceTests.cpp`
* **Java tests**: `platform-tests/.../workspace/MultiBackendWorkspaceTests.java`

Tests cover:
- Workspace creation/destruction
- Allocation patterns
- Scope management
- Coherence state transitions
- Device management
- Statistics tracking
- C-style API
- Manager operations
- Integration with BackendRegistry and Nd4j static APIs

## Related ADRs

* **ADR-0024 - Workspaces**: Original workspace concept this extends
* **ADR-0055 - Kernel Selection and Dynamic Loading**: Device routing for operations
