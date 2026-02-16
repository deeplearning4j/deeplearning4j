# ADR: TPU Backend

## Status

Proposed (Skeleton Implementation)

Proposed by: Adam Gibson (January 2025)

Discussed with: Development Team

## Context

Google Cloud TPUs (Tensor Processing Units) are custom-designed ASICs optimized for large-scale machine learning workloads. TPU v4 and v5 generations offer significant advantages for training and inference:

- **High memory bandwidth**: TPU v5p provides 96GB HBM per chip (vs. 24GB on RTX 4090)
- **Native bfloat16**: TPUs are architected around bfloat16 precision, achieving higher throughput than FP32 without significant accuracy loss
- **Multi-chip scaling**: TPU pods connect hundreds of chips via high-bandwidth inter-chip interconnect (ICI), enabling efficient data and model parallelism
- **HLO compilation**: XLA's High-Level Operations compiler performs whole-graph optimization, including operator fusion, layout assignment, and memory planning

ND4J currently supports CUDA (NVIDIA GPUs) and native CPU backends. Adding TPU support enables:

1. Running existing SameDiff models on Google Cloud TPU infrastructure
2. Leveraging TPU pods for large-scale training that exceeds single-GPU memory
3. Using bfloat16 natively for efficient inference
4. Accessing TPU-specific optimizations (MXU utilization, ICI communication)

The integration uses Google's PJRT (Portable Runtime) API — a hardware-agnostic runtime that provides a uniform interface for compiling and executing HLO computations on TPUs, GPUs, and CPUs.

## Decision

We implement a TPU backend for ND4J using the PJRT API, following the established backend pattern (Nd4jBackend, Environment, NDArrayFactory, OpExecutioner).

### Architecture

```
┌────────────────────────────────────────────────────────────────────┐
│                    ND4J API Layer                                    │
│  Nd4j.create(), INDArray.add(), SameDiff.output()                   │
└──────────────────────────────┬─────────────────────────────────────┘
                               │
                               ▼
┌────────────────────────────────────────────────────────────────────┐
│                    JTpuBackend                                      │
│  extends Nd4jBackend                                                │
│                                                                     │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │ TpuEnvironment                                                │  │
│  │  - TPU version detection (v4, v5e, v5p)                       │  │
│  │  - HBM capacity reporting                                     │  │
│  │  - bfloat16 preference                                        │  │
│  │  - Multi-chip topology                                        │  │
│  └──────────────────────────────────────────────────────────────┘  │
│                                                                     │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │ TpuExecutioner                                                │  │
│  │  extends DefaultOpExecutioner                                 │  │
│  │                                                               │  │
│  │  Op → HLO compilation → PJRT execution                       │  │
│  │  Compiled executable caching                                  │  │
│  │  Shape calculation on CPU (host)                              │  │
│  └──────────────────────────────────────────────────────────────┘  │
│                                                                     │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │ JTpuNDArray / JTpuNDArrayFactory                              │  │
│  │  - Device array management (toDevice/toHost)                  │  │
│  │  - HBM buffer allocation via PJRT                             │  │
│  │  - Lazy host↔device transfer                                 │  │
│  └──────────────────────────────────────────────────────────────┘  │
└──────────────────────────────┬─────────────────────────────────────┘
                               │
                               ▼
┌────────────────────────────────────────────────────────────────────┐
│                    PJRT (Portable Runtime)                           │
│                                                                     │
│  ┌──────────────────┐  ┌──────────────┐  ┌─────────────────────┐  │
│  │ PJRT Client      │  │ PJRT Device  │  │ PJRT Loaded         │  │
│  │ - Device enum    │  │ - TPU chip   │  │   Executable        │  │
│  │ - Plugin load    │  │ - HBM alloc  │  │ - Compiled HLO      │  │
│  │ - Compile        │  │ - Execute    │  │ - Cached per shape   │  │
│  └──────────────────┘  └──────────────┘  └─────────────────────┘  │
└──────────────────────────────┬─────────────────────────────────────┘
                               │
                               ▼
┌────────────────────────────────────────────────────────────────────┐
│                    TPU Hardware                                      │
│                                                                     │
│  TPU v4: 275 TFLOPS BF16, 32GB HBM, 8 cores/chip                  │
│  TPU v5e: 197 TFLOPS BF16, 16GB HBM, 8 cores/chip (cost-opt)     │
│  TPU v5p: 459 TFLOPS BF16, 96GB HBM, 8 cores/chip (perf-opt)     │
└────────────────────────────────────────────────────────────────────┘
```

### Module Structure

```
nd4j/nd4j-backends/nd4j-backend-impls/
├── nd4j-tpu/
│   └── src/main/java/org/nd4j/linalg/jtpu/
│       ├── JTpuBackend.java          # Backend discovery via SPI
│       ├── TpuEnvironment.java       # TPU device info and config
│       ├── JTpuNDArray.java          # TPU-specific array impl
│       ├── JTpuNDArrayFactory.java   # Array creation on TPU
│       └── ops/
│           ├── TpuExecutioner.java   # Op execution via PJRT
│           └── TpuOpContext.java     # Op context for TPU
│
├── nd4j-tpu-preset/
│   └── src/main/java/org/nd4j/
│       ├── Nd4jTpuPresets.java       # JavaCPP PJRT bindings
│       └── Nd4jTpuHelper.java        # Utility methods
```

### Backend Discovery

`JTpuBackend` is discovered via Java SPI (`ServiceLoader<Nd4jBackend>`):

```java
public class JTpuBackend extends Nd4jBackend {
    @Override
    public int getPriority() {
        return BACKEND_PRIORITY_GPU + 10; // Higher than CPU, lower than CUDA
    }

    @Override
    public boolean canRun() {
        // 1. Check PJRT library is loadable
        // 2. Check TPU devices are accessible
        // 3. Check sufficient permissions
        return pjrtClientAvailable && tpuDeviceCount > 0;
    }

    @Override
    public Environment getEnvironment() {
        return TpuEnvironment.getInstance();
    }
}
```

Priority is set higher than CPU (0) but lower than native CUDA (100), ensuring TPU is preferred over CPU when available but doesn't interfere with CUDA on hybrid systems.

### PJRT Execution Model

Operations are compiled to HLO (High-Level Operations) and executed through PJRT:

```java
public class TpuExecutioner extends DefaultOpExecutioner {
    // Compilation cache: opSignature → compiled executable
    private Map<String, PjrtLoadedExecutable> compiledCache = new HashMap<>();

    @Override
    public void exec(CustomOp op, OpContext context) {
        String signature = computeOpSignature(op, context);
        PjrtLoadedExecutable executable = compiledCache.computeIfAbsent(
            signature, k -> compileToHlo(op, context));
        executable.execute(context.getInputArrays(), context.getOutputArrays());
    }
}
```

**Compilation Caching**: HLO compilation is expensive (100ms-10s depending on graph complexity). Compiled executables are cached by op signature (op name + input shapes + dtypes), so repeated executions with the same shapes reuse the compiled code.

**Shape Calculation**: Shape inference runs on the host CPU, not on TPU. This avoids device round-trips for shape computation and reuses existing C++ shape functions.

### TPU Environment

`TpuEnvironment` provides TPU-specific device information:

```java
public class TpuEnvironment implements Environment {
    public String getTpuVersion()     // "v4", "v5e", "v5p"
    public int getTpuCoreCount()      // Cores per chip (typically 8)
    public long getHbmCapacity()      // HBM bytes per chip
    public boolean preferBfloat16()   // Always true for TPUs
}
```

HBM capacity varies by generation:
- TPU v4: 32GB per chip
- TPU v5e: 16GB per chip (cost-optimized)
- TPU v5p: 96GB per chip (performance-optimized)

### Native Binding Strategy

JavaCPP presets (`Nd4jTpuPresets`) map the PJRT C API to Java:

```java
@Properties(target = "jnind4jtpu", link = "nd4jtpu")
public class Nd4jTpuPresets implements InfoMapper {
    // Maps PJRT C API functions to Java via JavaCPP
    // Targets: linux-x86_64, linux-arm64
}
```

The binding uses the PJRT C API (not the C++ API) for maximum ABI stability. PJRT plugins (`libtpu.so`) are loaded at runtime via `PJRT_LoadedClientCreate`.

### Data Transfer Model

Arrays are lazily transferred between host and TPU:

```java
public class JTpuNDArray extends BaseNDArray {
    private boolean deviceDirty = false;
    private boolean hostDirty = false;

    public void toDevice() {
        if (hostDirty) {
            pjrtClient.transferToDevice(hostBuffer, tpuBuffer);
            hostDirty = false;
        }
    }

    public void toHost() {
        if (deviceDirty) {
            pjrtClient.transferToHost(tpuBuffer, hostBuffer);
            deviceDirty = false;
        }
    }
}
```

Operations mark arrays as device-dirty after execution. Host reads trigger a device→host transfer only when needed. This minimizes PCIe/network bandwidth usage for pure-device computation chains.

## Implementation Status

The current implementation is a **skeleton** with the backend infrastructure in place but native bindings pending:

**Implemented**:
- Backend discovery and priority (JTpuBackend)
- Environment configuration (TpuEnvironment)
- Executioner framework with compilation caching pattern (TpuExecutioner)
- NDArray and NDArrayFactory stubs (JTpuNDArray, JTpuNDArrayFactory)
- Maven module structure with JavaCPP preset scaffold

**Pending**:
- JavaCPP PJRT native bindings generation
- Native PJRT client initialization
- HLO compilation from op graphs
- Device buffer allocation and transfer
- Multi-chip execution (data/model parallelism)
- Integration testing on TPU hardware

## Consequences

### Advantages

**TPU Access**: Enables ND4J/SameDiff workloads on Google Cloud TPU infrastructure — the only ASIC designed specifically for machine learning.

**Pod Scaling**: PJRT supports multi-chip TPU pods, enabling model parallelism across hundreds of chips for large language models.

**Native bfloat16**: TPUs achieve peak throughput with bfloat16, and the environment defaults to this precision. No manual dtype management needed.

**Standard Backend Pattern**: Follows the established Nd4jBackend/Environment/Executioner pattern, ensuring full API compatibility with existing ND4J code.

**HLO Optimization**: XLA's compiler performs whole-graph optimization (fusion, layout, memory planning) that can outperform hand-tuned kernels for certain workloads.

### Disadvantages

**Cloud Dependency**: TPUs are only available on Google Cloud (and limited Colab access). No on-premise option exists.

**PJRT Maturity**: The PJRT C API is relatively new and may have stability issues. Error handling and debugging are less mature than CUDA.

**Compilation Overhead**: HLO compilation for large graphs can take seconds. This is amortized for training loops but impacts interactive/latency-sensitive applications.

**Limited Hardware Availability**: TPU access requires Google Cloud account setup, quotas, and potentially waitlists for newer generations.

**Incomplete Implementation**: The current skeleton requires significant engineering to reach production readiness, particularly the native binding generation and multi-chip support.

## References

- nd4j/nd4j-backends/nd4j-backend-impls/nd4j-tpu/ (backend module)
- nd4j/nd4j-backends/nd4j-backend-impls/nd4j-tpu-preset/ (JavaCPP bindings)
- Google PJRT API: https://github.com/openxla/xla/tree/main/xla/pjrt/c
- Google Cloud TPU: https://cloud.google.com/tpu
- ADR 0058 - Multi-Backend Kernel Selection and Management
- ADR 0059 - Multi-Backend Op Execution System
