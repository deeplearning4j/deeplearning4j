# ADR 0055: Kernel Selection and Dynamic Plugin Loading System

## Status
Accepted

## Related ADRs
- [ADR 0058](0058%20-%20Multi-Backend%20Kernel%20Selection%20and%20Management.md) — extends this ADR with KernelManager and Java fluent API
- [ADR 0059](0059%20-%20Multi-Backend%20Op%20Execution%20System.md) — orthogonal: routes between JVM backends (CPU vs CUDA), while this ADR selects kernels within a backend

## Context

Deep learning frameworks increasingly support multiple compute backends (cuDNN, oneDNN, Metal Performance Shaders, ARM Compute Library, etc.) that can execute the same operations with varying performance characteristics. The optimal backend often depends on:

- Hardware platform (NVIDIA GPU, Intel CPU, Apple Silicon, ARM devices)
- Input tensor shapes and sizes
- Data types (FP32, FP16, BF16, INT8)
- Operation parameters (kernel sizes, strides, etc.)

The existing libnd4j architecture has a **PlatformHelper** system that allows backend-specific implementations to be registered for operations. However, this system had limitations:

1. **Static selection**: The first available helper was always used, regardless of performance
2. **No benchmarking**: No runtime performance measurement to choose the fastest implementation
3. **No dynamic loading**: Custom kernels required recompilation of the entire library
4. **No Java-level control**: Configuration was limited to C++ compile-time options

## Decision

We implement a comprehensive kernel selection and dynamic loading system that:

1. **Auto-tunes at runtime** to select the fastest kernel for each operation/shape combination
2. **Caches performance data** persistently to avoid repeated benchmarking
3. **Supports dynamic loading** of custom kernel plugins from shared libraries
4. **Provides Java-level APIs** for configuration and plugin management
5. **Integrates seamlessly** with the existing PlatformHelper architecture

### Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                         Java Layer                               │
│  ┌─────────────────┐  ┌──────────────────┐  ┌────────────────┐  │
│  │ KernelSelector  │  │KernelPluginManager│  │KernelSelection │  │
│  │   Interface     │  │    Interface      │  │    Config      │  │
│  └────────┬────────┘  └────────┬─────────┘  └───────┬────────┘  │
└───────────┼────────────────────┼────────────────────┼───────────┘
            │                    │                    │
            ▼                    ▼                    ▼
┌─────────────────────────────────────────────────────────────────┐
│                        Native Layer (JNI)                        │
└─────────────────────────────────────────────────────────────────┘
            │                    │                    │
            ▼                    ▼                    ▼
┌─────────────────────────────────────────────────────────────────┐
│                        C++ Core Layer                            │
│  ┌─────────────────────┐  ┌─────────────────────────────────┐   │
│  │ MultiPlatformDispatcher│ │    DynamicKernelLoader        │   │
│  │  - Mode selection      │ │    - Plugin loading/unload    │   │
│  │  - Helper routing      │ │    - Hot-reload support       │   │
│  │  - Execution timing    │ │    - Factory registration     │   │
│  └──────────┬─────────────┘ └──────────────┬────────────────┘   │
│             │                              │                     │
│             ▼                              ▼                     │
│  ┌─────────────────────┐  ┌─────────────────────────────────┐   │
│  │KernelPerformanceRegistry│ │      KernelAutoTuner         │   │
│  │  - Performance cache    │ │    - Warmup runs             │   │
│  │  - Shape bucketing      │ │    - Benchmark execution     │   │
│  │  - File persistence     │ │    - Statistical analysis    │   │
│  └──────────┬──────────────┘ └──────────────┬───────────────┘   │
│             │                               │                    │
│             └───────────────┬───────────────┘                    │
│                             ▼                                    │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │                    OpRegistrator                         │    │
│  │  - Helper registration by (opHash, Engine)               │    │
│  │  - Multi-helper enumeration                              │    │
│  └──────────────────────────┬──────────────────────────────┘    │
│                             ▼                                    │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │                   PlatformHelper                         │    │
│  │  - isUsable() check                                      │    │
│  │  - invokeHelper() execution                              │    │
│  └─────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
```

## Implementation Details

### 1. Kernel Performance Registry

**Files**: `libnd4j/include/helpers/KernelPerformanceRegistry.h`, `impl/KernelPerformanceRegistry.cpp`

The registry caches performance data with a signature based on:
- Operation hash (unique identifier for each op)
- Shape bucket (powers of 2 for dimension ranges)
- Data type

```cpp
struct KernelSignature {
    sd::LongType opHash;
    std::vector<int> shapeBucket;
    DataType dataType;
};

struct KernelPerformanceEntry {
    double meanTimeNanos;
    double varianceNanos;
    int sampleCount;
    int64_t lastUpdated;
};
```

**Shape Bucketing**: To reduce cache entries, shapes are bucketed to powers of 2:
- Shapes 1-16 → bucket 16
- Shapes 17-32 → bucket 32
- Shapes 33-64 → bucket 64
- etc.

This provides shape-specific tuning without unbounded cache growth.

### 2. Kernel Auto-Tuner

**Files**: `libnd4j/include/helpers/KernelAutoTuner.h`, `impl/KernelAutoTuner.cpp`

The auto-tuner benchmarks available helpers:

```cpp
BenchmarkResult benchmark(graph::Context& ctx, KernelExecutor* executor,
                          int warmupRuns = 2, int benchmarkRuns = 5);
```

Features:
- Configurable warmup runs (default: 2)
- Configurable benchmark runs (default: 5)
- Context cloning for safe benchmarking
- RAII `TuningGuard` to prevent recursive tuning
- Welford's algorithm for online variance calculation

### 3. Multi-Platform Dispatcher

**Files**: `libnd4j/include/ops/declarable/MultiPlatformDispatcher.h`, `impl/MultiPlatformDispatcher.cpp`

Dispatch modes:
- **AUTO**: Select fastest based on benchmarks (default)
- **FIXED**: Use specified engine only
- **ROUND_ROBIN**: Distribute across backends
- **BENCHMARK**: Always benchmark, never use cache

```cpp
Status MultiPlatformDispatcher::executeWithBestHelper(
    DeclarableOp* op,
    graph::Context& context,
    const std::string& opName);
```

### 4. Dynamic Kernel Loader

**Files**: `libnd4j/include/helpers/DynamicKernelLoader.h`, `impl/DynamicKernelLoader.cpp`

Plugin interface:
```cpp
class KernelPlugin {
public:
    virtual std::string getName() const = 0;
    virtual Version getVersion() const = 0;
    virtual bool initialize() = 0;
    virtual void shutdown() = 0;
    virtual std::vector<KernelInfo> getProvidedKernels() const = 0;
};
```

Plugin macro for exports:
```cpp
SD_DECLARE_KERNEL_PLUGIN(MyPluginClass)
```

This exports the required C functions:
- `sd_plugin_create()` - Factory function
- `sd_plugin_destroy()` - Cleanup function
- `sd_plugin_api_version()` - API compatibility check

### 5. Java API

**Files**:
- `nd4j/.../KernelSelectionConfig.java` - Configuration
- `nd4j/.../KernelSelector.java` - Selection interface
- `nd4j/.../KernelPluginManager.java` - Plugin management

```java
// Configure kernel selection
KernelSelectionConfig config = KernelSelectionConfig.builder()
    .strategy(Strategy.FASTEST)
    .autoTuneEnabled(true)
    .warmupRuns(2)
    .benchmarkRuns(5)
    .build();

Nd4j.getKernelSelector().configure(config);

// Load a custom plugin
Nd4j.loadKernelPlugin("/path/to/my_kernels.so");

// Get performance info
System.out.println(Nd4j.getKernelSelector().getPerformanceSummary());
```

### 6. Environment Variables

| Variable | Description | Example |
|----------|-------------|---------|
| `SD_KERNEL_STRATEGY` | Selection strategy | `FASTEST`, `FIRST_AVAILABLE` |
| `SD_KERNEL_AUTOTUNE` | Enable auto-tuning | `1` or `true` |
| `SD_KERNEL_WARMUP_RUNS` | Warmup iterations | `2` |
| `SD_KERNEL_BENCHMARK_RUNS` | Benchmark iterations | `5` |
| `SD_KERNEL_CACHE_PATH` | Cache file location | `/tmp/nd4j_kernel_cache` |
| `SD_KERNEL_FORCE_ENGINE` | Force specific engine | `cuda`, `onednn` |
| `SD_KERNEL_DISABLE_ENGINES` | Disable engines | `cuda,onednn` |
| `SD_KERNEL_VERBOSE` | Verbose logging | `1` |
| `SD_KERNEL_PLUGIN_PATH` | Plugin search paths | `/opt/plugins:/usr/lib/nd4j` |
| `SD_KERNEL_PLUGIN_AUTO` | Auto-load plugins | `1` |

### 7. Creating Custom Plugins

Example plugin (`my_kernels.cpp`):
```cpp
#include <helpers/DynamicKernelLoader.h>
#include <ops/declarable/PlatformHelper.h>

class MyOptimizedConv2d : public PlatformHelper {
public:
    MyOptimizedConv2d() : PlatformHelper("conv2d", samediff::ENGINE_CPU) {}

    bool isUsable(graph::Context& ctx) override {
        auto input = ctx.getNDArray(0);
        return input && input->dataType() == FLOAT32 && input->lengthOf() >= 1024;
    }

    Status invokeHelper(graph::Context& ctx) override {
        // Optimized implementation
        return Status::OK;
    }
};

class MyKernelPlugin : public SimpleKernelPlugin {
public:
    MyKernelPlugin() : SimpleKernelPlugin("MyKernels", {1, 0, 0}) {}

    bool initialize() override {
        registerKernel("conv2d", samediff::ENGINE_CPU,
            []() { return new MyOptimizedConv2d(); }, 150);
        return true;
    }
};

SD_DECLARE_KERNEL_PLUGIN(MyKernelPlugin)
```

Compilation:
```bash
# Linux
g++ -shared -fPIC -o libmy_kernels.so my_kernels.cpp \
    -I/path/to/libnd4j/include -L/path/to/libnd4j/lib -lnd4j

# macOS
clang++ -shared -fPIC -o libmy_kernels.dylib my_kernels.cpp \
    -I/path/to/libnd4j/include -L/path/to/libnd4j/lib -lnd4j
```

### 8. OpRegistrator Extensions

New methods added to `OpRegistrator`:
```cpp
bool hasAnyHelper(sd::LongType hash);
std::vector<ops::platforms::PlatformHelper*> getAllHelpersForOp(sd::LongType hash);
std::vector<samediff::Engine> getAvailableEnginesForOp(sd::LongType hash);
```

## Consequences

### Benefits

1. **Automatic Performance Optimization**: Operations automatically use the fastest available implementation without manual configuration.

2. **Hardware Adaptation**: Performance cache adapts to specific hardware, providing optimal kernels per system.

3. **Extensibility**: Custom kernel plugins can be developed and loaded without recompiling ND4J.

4. **Hot-Reload Support**: During development, plugins can be reloaded without restarting the application.

5. **Cross-Platform**: Works on Linux, macOS, and Windows with appropriate shared library formats.

6. **Backward Compatible**: Default behavior is compatible with existing code; auto-tuning is opt-in via configuration.

7. **Persistent Cache**: Benchmark results persist across sessions, eliminating repeated tuning overhead.

### Drawbacks

1. **Initial Overhead**: First execution of each operation/shape combination incurs benchmarking overhead.

2. **Memory Usage**: Performance cache consumes memory, though shape bucketing limits growth.

3. **Complexity**: Additional code paths and configuration options increase system complexity.

4. **Plugin ABI Stability**: Plugins must be compiled against compatible libnd4j versions.

### Migration

- Existing code continues to work unchanged
- To enable auto-tuning: `Nd4j.getKernelSelector().setAutoTuneEnabled(true)`
- To load plugins: `Nd4j.loadKernelPlugin("/path/to/plugin.so")`

## References

- [PlatformHelper Architecture](../libnd4j/include/ops/declarable/PlatformHelper.h)
- [OpRegistrator](../libnd4j/include/ops/declarable/OpRegistrator.h)
- [Kernel Plugin Template](../libnd4j/include/helpers/KernelPluginTemplate.h)
- cuDNN documentation on algorithm selection
- oneDNN primitive caching mechanisms
