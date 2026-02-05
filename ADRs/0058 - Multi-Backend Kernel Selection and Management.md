# ADR: Multi-Backend Kernel Selection and Management

## Status
Accepted

## Context

The deeplearning4j/nd4j framework supports multiple execution backends (CPU, CUDA, oneDNN, cuDNN, MPS, etc.) through a "platform helper" system. Each operation can have multiple implementations optimized for different hardware. However, the existing infrastructure had several limitations:

### Previous State

1. **Simple dispatch logic**: Operations used the first available helper without considering which might be fastest
2. **No runtime configuration**: Users couldn't easily enable/disable specific backends per operation
3. **Limited visibility**: No way to query which kernels are available for an operation
4. **No auto-tuning**: No mechanism to benchmark and select the best kernel at runtime
5. **No Java API**: Kernel configuration required C++ changes or environment variables only
6. **Coarse granularity**: Could only enable/disable helpers globally, not per-operation

### Requirements

1. Allow users to configure kernel selection at multiple levels (global, category, per-op)
2. Provide auto-tuning capability to select best kernel based on runtime benchmarks
3. Expose kernel management through a fluent Java API at the SameDiff level
4. Support environment variable configuration for quick overrides
5. Cache performance data to avoid repeated benchmarking
6. Integrate with the existing MultiPlatformDispatcher and PlatformHelper systems

## Decision

Implement a comprehensive multi-backend kernel selection and management system with three key components:

### 1. KernelDispatchHelper (C++)

Enhanced dispatch logic that integrates with auto-tuning and environment configuration:

```cpp
namespace sd::ops::platforms {

class KernelDispatchHelper {
public:
    // Check if a helper should be used for this operation
    static bool shouldUseHelper(DeclarableOp* op, sd::graph::Context& context);

    // Dispatch with auto-tuning support
    static std::pair<bool, Status> dispatchWithAutoTune(
        DeclarableOp* op,
        sd::graph::Context& context
    );
};

}  // namespace sd::ops::platforms
```

The dispatch logic follows this priority:

1. If `SD_KERNEL_FORCE_ENGINE` is set, use only that engine
2. If an engine is disabled via `SD_KERNEL_DISABLE_ENGINES`, skip it
3. If auto-tuning is enabled (`SD_KERNEL_AUTOTUNE=1`), benchmark all usable helpers and select best
4. Otherwise, use the first usable helper

### 2. KernelManager (C++ with JNI bindings)

Centralized registry for kernel discovery and configuration:

```cpp
namespace sd::ops {

struct KernelInfo {
    std::string name;
    samediff::Engine engine;
    bool enabled;
    int priority;
};

struct OpKernelInfo {
    std::string opName;
    LongType opHash;
    std::vector<KernelInfo> availableKernels;
    samediff::Engine preferredEngine;
};

class KernelManager {
public:
    static KernelManager& getInstance();

    // Discovery
    std::vector<OpKernelInfo> getAllOperationsWithKernels();
    OpKernelInfo getOpKernelInfo(const std::string& opName);
    std::vector<OpKernelInfo> searchOperations(const std::string& pattern);

    // Enable/Disable
    void enableKernel(const std::string& opName, samediff::Engine engine);
    void disableKernel(const std::string& opName, samediff::Engine engine);
    bool isKernelEnabled(const std::string& opName, samediff::Engine engine);

    // Engine preferences
    void setPreferredEngine(const std::string& opName, samediff::Engine engine);
    samediff::Engine getPreferredEngine(const std::string& opName);
    void setGlobalPreferredEngine(samediff::Engine engine);

    // Global configuration
    void disableEngineGlobally(samediff::Engine engine);
    void enableEngineGlobally(samediff::Engine engine);

    // State management
    void resetToDefaults();
    std::string getConfigurationSummary();
};

}  // namespace sd::ops
```

### 3. KernelConfiguration (Java Fluent API)

User-friendly configuration builder exposed through SameDiff:

```java
public class KernelConfiguration {

    public enum Preset {
        CPU_OPTIMIZED,
        GPU_OPTIMIZED,
        INTEL_OPTIMIZED,
        APPLE_SILICON_OPTIMIZED,
        MAXIMUM_COMPATIBILITY,
        AUTO_TUNE
    }

    public enum OperationCategory {
        CONVOLUTIONS("conv*", "depthwise*", "separable*"),
        POOLING("*pool*", "max_pool*", "avg_pool*"),
        NORMALIZATION("*norm*", "batch_norm*", "layer_norm*"),
        ACTIVATIONS("relu*", "sigmoid", "tanh", "softmax*"),
        LINEAR_ALGEBRA("matmul", "gemm", "dot", "tensordot"),
        ATTENTION("attention*", "dot_product_attention*"),
        // ... more categories
    }

    // Global settings
    public KernelConfiguration preferCuda();
    public KernelConfiguration preferOneDnn();
    public KernelConfiguration preferCpu();
    public KernelConfiguration disableEngine(Engine engine);
    public KernelConfiguration cpuOnly();

    // Category-based configuration
    public CategoryConfiguration forConvolutions();
    public CategoryConfiguration forLinearAlgebra();
    public CategoryConfiguration forAttention();
    // ... returns builder for category-specific settings

    // Operation-specific configuration
    public OperationConfiguration forOperation(String opName);
    public PatternConfiguration forPattern(String pattern);

    // Presets
    public KernelConfiguration usePreset(Preset preset);

    // Apply changes
    public KernelConfiguration apply();
}
```

## Implementation

### Core Components

```
libnd4j/include/
├── ops/declarable/
│   ├── KernelManager.h           # Kernel registry and configuration
│   └── impl/
│       └── KernelManager.cpp     # Implementation with C-style JNI API
├── helpers/
│   ├── KernelSelectionEnvironment.h   # Environment variable handling
│   ├── KernelAutoTuner.h              # Runtime benchmarking
│   ├── KernelPerformanceRegistry.h    # Performance data cache
│   └── impl/
│       ├── KernelSelectionEnvironment.cpp
│       ├── KernelAutoTuner.cpp
│       └── KernelPerformanceRegistry.cpp

nd4j/nd4j-backends/nd4j-api-parent/nd4j-api/src/main/java/org/nd4j/
├── linalg/api/ops/executioner/
│   └── KernelManager.java        # Java wrapper for native KernelManager
└── autodiff/samediff/config/
    └── KernelConfiguration.java  # Fluent configuration builder
```

### Environment Variables

| Variable | Description | Example |
|----------|-------------|---------|
| `SD_KERNEL_AUTOTUNE` | Enable auto-tuning | `1` or `true` |
| `SD_KERNEL_VERBOSE` | Enable verbose logging | `1` or `true` |
| `SD_KERNEL_FORCE_ENGINE` | Force specific engine | `cuda`, `onednn`, `cpu` |
| `SD_KERNEL_DISABLE_ENGINES` | Disable engines (comma-separated) | `onednn,cuda` |
| `SD_KERNEL_CACHE_PATH` | Path to performance cache | `/tmp/kernel_cache.json` |
| `SD_KERNEL_WARMUP_RUNS` | Warmup iterations for benchmarking | `5` |
| `SD_KERNEL_BENCHMARK_RUNS` | Benchmark iterations | `10` |

### Integration with DeclarableOp

The `DeclarableOp::execute()` method now uses the new dispatch system:

```cpp
Status DeclarableOp::execute(Context* block) {
    // ... validation and setup ...

    // Try to dispatch to platform helper with auto-tuning
    auto dispatchResult = platforms::KernelDispatchHelper::dispatchWithAutoTune(this, *block);
    if (dispatchResult.first) {
        status = dispatchResult.second;
        hasHelper = true;
    } else {
        // Fall back to native implementation
        status = this->validateAndExecute(*block);
    }

    // ... cleanup ...
}
```

### Auto-Tuning Flow

```
                    ┌─────────────────────────────────────┐
                    │  Operation Execution Request         │
                    └─────────────────────────────────────┘
                                    │
                                    ▼
                    ┌─────────────────────────────────────┐
                    │  Check KernelSelectionEnvironment    │
                    │  - Forced engine?                    │
                    │  - Disabled engines?                 │
                    └─────────────────────────────────────┘
                                    │
                         ┌──────────┴──────────┐
                         ▼                     ▼
              ┌──────────────────┐   ┌──────────────────┐
              │  Forced Engine   │   │  Normal Path      │
              │  Use if usable   │   └──────────────────┘
              └──────────────────┘            │
                                              ▼
                              ┌─────────────────────────────┐
                              │  Get All Usable Helpers      │
                              │  Filter by:                  │
                              │  - Engine not disabled       │
                              │  - Helper.isUsable(context)  │
                              └─────────────────────────────┘
                                              │
                         ┌────────────────────┼────────────────────┐
                         ▼                    ▼                    ▼
              ┌──────────────────┐ ┌──────────────────┐ ┌──────────────────┐
              │  0 helpers       │ │  1 helper        │ │  2+ helpers      │
              │  → Native exec   │ │  → Use it        │ │  → Auto-tune?    │
              └──────────────────┘ └──────────────────┘ └──────────────────┘
                                                                  │
                                              ┌───────────────────┴───────────────────┐
                                              ▼                                       ▼
                                   ┌──────────────────┐                    ┌──────────────────┐
                                   │  Auto-tune OFF   │                    │  Auto-tune ON    │
                                   │  → Use first     │                    │  → Check cache   │
                                   └──────────────────┘                    └──────────────────┘
                                                                                     │
                                                                    ┌────────────────┴────────────────┐
                                                                    ▼                                ▼
                                                         ┌──────────────────┐             ┌──────────────────┐
                                                         │  Cache hit       │             │  Cache miss      │
                                                         │  → Use cached    │             │  → Benchmark all │
                                                         └──────────────────┘             │  → Cache result  │
                                                                                          └──────────────────┘
```

## Usage Examples

### Java - Basic Configuration

```java
SameDiff sd = SameDiff.create();

// Configure kernels for this model
sd.kernelConfiguration()
    .preferCuda()                              // Use CUDA when available
    .disableEngine(Engine.ONEDNN)              // Don't use oneDNN
    .forConvolutions().useCudnn()              // Use cuDNN for convolutions
    .and()
    .forLinearAlgebra().useOneDnn()            // Use oneDNN for matmul
    .and()
    .apply();
```

### Java - Using Presets

```java
// Quick setup for common scenarios
sd.kernelConfiguration()
    .usePreset(Preset.GPU_OPTIMIZED)
    .apply();

// Or for Intel CPUs
sd.kernelConfiguration()
    .usePreset(Preset.INTEL_OPTIMIZED)
    .apply();
```

### Java - Querying Available Kernels

```java
KernelManager km = KernelManager.getInstance();

// Find all convolution operations with their available kernels
List<OpKernelInfo> convOps = km.searchOperations("conv*");
for (OpKernelInfo op : convOps) {
    System.out.println(op.getOpName() + ":");
    for (KernelInfo kernel : op.getAvailableKernels()) {
        System.out.println("  - " + kernel.getEngine() +
                          " (enabled: " + kernel.isEnabled() + ")");
    }
}
```

### Java - Fine-grained Control

```java
// Disable oneDNN only for batch normalization
sd.kernelConfiguration()
    .forOperation("batchnorm")
        .disableEngine(Engine.ONEDNN)
        .and()
    .forPattern("conv*")
        .useEngine(Engine.CUDNN)
        .and()
    .apply();
```

### Environment Variables

```bash
# Force CUDA for all operations
export SD_KERNEL_FORCE_ENGINE=cuda

# Enable auto-tuning with caching
export SD_KERNEL_AUTOTUNE=1
export SD_KERNEL_CACHE_PATH=/tmp/nd4j_kernel_cache.json

# Disable oneDNN and MPS
export SD_KERNEL_DISABLE_ENGINES=onednn,mps

# Enable verbose logging for debugging
export SD_KERNEL_VERBOSE=1
```

### C++ - Direct Usage

```cpp
#include <ops/declarable/KernelManager.h>
#include <helpers/KernelSelectionEnvironment.h>

// Query available kernels
auto& km = KernelManager::getInstance();
auto convInfo = km.getOpKernelInfo("conv2d");
for (const auto& kernel : convInfo.availableKernels) {
    std::cout << kernel.name << " (" << kernel.engine << ")\n";
}

// Configure kernel selection
km.setPreferredEngine("matmul", samediff::ENGINE_ONEDNN);
km.disableKernel("batchnorm", samediff::ENGINE_CUDA);

// Check environment configuration
if (KernelSelectionEnvironment::isAutoTuneEnabled()) {
    std::cout << "Auto-tuning is active\n";
}
```

## Consequences

### Positive

1. **User control**: Users can now fine-tune kernel selection for their specific hardware and workloads
2. **Performance optimization**: Auto-tuning automatically selects the best kernel based on runtime benchmarks
3. **Debugging**: Easy to disable specific backends when troubleshooting
4. **Portability**: Same model can be configured differently for different deployment targets
5. **Visibility**: Users can query which kernels are available and their status
6. **Fluent API**: Intuitive Java configuration through SameDiff
7. **Backward compatible**: Existing code continues to work with default behavior

### Negative

1. **Complexity**: Additional layer of abstraction in the execution path
2. **Memory overhead**: Caching performance data requires some memory
3. **Startup cost**: Auto-tuning adds latency during first execution of each operation
4. **Configuration drift**: Different configurations across environments can cause inconsistent behavior

### Mitigations

- Auto-tuning is opt-in (disabled by default)
- Performance cache persists across runs to amortize benchmark cost
- Configuration can be serialized/logged for reproducibility
- Clear logging in verbose mode helps debug configuration issues

## Related Decisions

- **ADR-OpTimingTracker**: Operation timing integrates with kernel dispatch to track which helpers are used
- **ADR-0055-Kernel_Selection_And_Dynamic_Loading**: Foundation for platform helper registration
- **MultiPlatformDispatcher**: Underlying system for managing multiple implementations per operation

## Files Changed/Created

| File | Purpose |
|------|---------|
| `libnd4j/include/ops/declarable/KernelManager.h` | Kernel registry header |
| `libnd4j/include/ops/declarable/impl/KernelManager.cpp` | Registry implementation with C-style API |
| `libnd4j/include/helpers/KernelSelectionEnvironment.h` | Environment configuration header |
| `libnd4j/include/helpers/impl/KernelSelectionEnvironment.cpp` | Environment handling and KernelDispatchHelper |
| `libnd4j/include/helpers/KernelAutoTuner.h` | Auto-tuning header |
| `libnd4j/include/helpers/impl/KernelAutoTuner.cpp` | Runtime benchmarking |
| `libnd4j/include/helpers/KernelPerformanceRegistry.h` | Performance cache header |
| `libnd4j/include/helpers/impl/KernelPerformanceRegistry.cpp` | Cache implementation |
| `libnd4j/include/ops/declarable/impl/DeclarableOp.cpp` | Integration with execute() |
| `nd4j/.../KernelManager.java` | Java wrapper for native API |
| `nd4j/.../KernelConfiguration.java` | Fluent configuration builder |
| `nd4j/.../SameDiff.java` | Added kernelConfiguration() method |

## Date

2026-01-02

## Authors

- Implementation: Claude Code (AI Assistant)
- Architecture: deeplearning4j team
