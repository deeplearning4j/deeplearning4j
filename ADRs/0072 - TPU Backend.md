# ADR: TPU Backend

## Status

Accepted

Proposed by: Adam Gibson (January 2025)

Updated: March 2026 — Added C++ graph replay architecture (PJRT compilation caching, HLO IR builder, TpuGraphBackend)

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
│  │  - Delegates the complete Environment contract to native C++  │  │
│  │  - Device count/name/memory comes from the PJRT client         │  │
│  └──────────────────────────────────────────────────────────────┘  │
│                                                                     │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │ TpuExecutioner                                                │  │
│  │  extends NativeOpExecutioner                                  │  │
│  │                                                               │  │
│  │  Custom eager ops + DSP share trait/KernelSpec lowering        │  │
│  │  KernelExpr → StableHLO → PJRT replay                         │  │
│  └──────────────────────────────────────────────────────────────┘  │
│                                                                     │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │ Host-native NDArray / CpuNDArrayFactory                        │  │
│  │  - One Java buffer/deallocation/serialization contract         │  │
│  │  - PJRT buffers are transient native replay resources          │  │
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
│       ├── TpuEnvironment.java       # Native Environment facade
│       ├── TpuStatisticsProvider.java
│       └── ops/
│           ├── TpuExecutioner.java   # Host-native + DSP TPU identity
│           └── TpuOpContext.java     # Shared opaque native context
│
├── nd4j-tpu-preset/
│   └── src/main/java/org/nd4j/
│       ├── Nd4jTpuPresets.java       # JavaCPP NativeOps binding
│       └── Nd4jTpuHelper.java        # NativeOps base class
```

### Backend Discovery

`JTpuBackend` is discovered via Java SPI (`ServiceLoader<Nd4jBackend>`):

```java
public class JTpuBackend extends Nd4jBackend {
    @Override
    public int getPriority() {
        return midpoint(BACKEND_PRIORITY_CPU, BACKEND_PRIORITY_GPU);
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

SameDiff DSP segments are lowered to StableHLO MLIR and executed through PJRT:

```java
public class TpuExecutioner extends NativeOpExecutioner {
    // Canonical CustomOps enter the same native StableHLO/PJRT lowerer as DSP.
    // Legacy numbered ops fail until given canonical KernelSpecs.
}
```

**Compilation Caching**: StableHLO compilation is expensive (100ms-10s
depending on graph complexity). The DSP plan cache is keyed by graph boundary
shapes and each TPU replay handle owns the loaded executable for that shape.
Shape drift replaces the handle rather than mutating a READY executable.

**Shape Calculation**: Shape inference runs on the host CPU, not on TPU. This avoids device round-trips for shape computation and reuses existing C++ shape functions.

### TPU Environment

`TpuEnvironment` delegates the complete public Environment contract to the
generated native `sd::Environment` binding. Java-only tracking flags remain in
the facade, while device identity and memory telemetry come from NativeOps/PJRT:

```java
Environment env = TpuEnvironment.getInstance();
int devices = Nd4j.getNativeOps().getAvailableDevices();
long hbm = Nd4j.getNativeOps().getDeviceTotalMemory(deviceId);
```

No capacity or core count is guessed from environment-variable TPU version
strings; the active PJRT client is the device authority.

### Native Binding Strategy

JavaCPP presets (`Nd4jTpuPresets`) expose the same backend-neutral `NativeOps`
ABI used by the native and CUDA backends:

```java
@Properties(target = "org.nd4j.linalg.jtpu.bindings.Nd4jTpu",
            link = "nd4jtpu")
public class Nd4jTpuPresets implements InfoMapper {
    // Maps NativeOps/NativeOpsDsp. Raw PJRT objects remain native-owned.
}
```

`PjrtClientManager` uses the typed PJRT C API for ABI stability, but resolves
`GetPjrtApi` from the selected plugin at runtime. `libnd4jtpu` never links or
preloads `libtpu.so`; the manager validates the API version, initializes the
plugin, creates the client, verifies that the platform is TPU, and enumerates
addressable devices.

### Data Transfer Model

Java NDArrays remain host-native and continue to use the mature ND4J buffer,
workspace, serialization, and deallocation contracts; this does not authorize
host numerical fallback. A compiled DSP or eager descriptor
segment records deterministic boundary input/output indices. At replay time,
`TpuReplayHandle` creates typed `PJRT_Buffer` inputs (dtype, dimensions and byte
strides), executes the loaded StableHLO program, waits for completion, copies
the exact boundary outputs into dense C-order NDArrays, and destroys all
transient events and buffers. Raw PJRT handles never cross the JNI boundary.

Persistent device-resident boundary buffers may be added later behind the same
native replay-handle ownership contract; they must be keyed by ND4J buffer
generation and device identity rather than represented by a second Java array
class.

## Implementation Status

The implementation includes both Java backend infrastructure and C++ graph replay architecture:

**Implemented**:
- Runtime backend discovery backed by the generated `Nd4jTpu implements NativeOps`
- Native Environment facade and host-native NDArray/factory/workspace control plane
- Maven JavaCPP generation, JNI/native classifier packaging, SPI and JPMS registration
- Typed PJRT lifecycle: API negotiation, plugin/client/device ownership, events,
  buffers, compilation, execution, output download, diagnostics and teardown
- Strict TPU platform identity (CPU/ROCm/CUDA PJRT plugins cannot select JTpuBackend)
- C++ graph replay: one owned loaded executable per shape-keyed DSP segment
- C++ graph backend: inclusive range admission, deterministic boundary binding,
  device selection, complete-lowering audit and fail-closed forced TPU mode
- Op-local 64-bit `NativeSlot` traits as family/safety authority
- Canonical descriptor-hash resolution into shared `KernelSpec`/`KernelExpr`
- StableHLO target sink for shared expression semantics, broadcasting, rank-2
  matmul, reshape, multiple segment results, and eager CustomOps
- GraphExecutionMode.TPU (native code 13) enum integration
- GraphReplayFactory dispatch for SD_TPU builds
- Runtime-only PJRT CMake configuration with one runtime; the duplicate per-op
  PJRT implementation and backend-local HLO op whitelist were removed

**Pending**:
- Broader shared KernelSpec and exact structural recipe coverage
- Legacy numbered eager op migration to canonical KernelSpecs (currently fails,
  never falls back to CPU numerics)
- Multi-chip/SPMD compile options and execution (current execution targets one
  addressable device per segment)
- Performance benchmarking vs CPU baseline

## Consequences

### Advantages

**TPU Access**: Enables ND4J/SameDiff workloads on Google Cloud TPU infrastructure — the only ASIC designed specifically for machine learning.

**Pod Scaling**: PJRT supports multi-chip TPU pods, enabling model parallelism across hundreds of chips for large language models.

**Native bfloat16**: TPUs execute bfloat16 natively. FLOAT remains the public
default so loading this backend does not silently change application numerics;
models can request or optimize weights to bfloat16 explicitly.

**Standard Backend Pattern**: Follows the established Nd4jBackend/Environment/Executioner pattern, ensuring full API compatibility with existing ND4J code.

**HLO Optimization**: XLA's compiler performs whole-graph optimization (fusion, layout, memory planning) that can outperform hand-tuned kernels for certain workloads.

### Disadvantages

**Cloud Dependency**: TPUs are only available on Google Cloud (and limited Colab access). No on-premise option exists.

**PJRT Maturity**: The PJRT C API is relatively new and may have stability issues. Error handling and debugging are less mature than CUDA.

**Compilation Overhead**: HLO compilation for large graphs can take seconds. This is amortized for training loops but impacts interactive/latency-sensitive applications.

**Limited Hardware Availability**: TPU access requires Google Cloud account setup, quotas, and potentially waitlists for newer generations.

**Conservative Lowering**: Forced TPU execution rejects unsupported operation
forms instead of silently running them on the host. This makes missing coverage
visible but requires expanding the StableHLO catalog for larger models.

## References

- nd4j/nd4j-backends/nd4j-backend-impls/nd4j-tpu/ (backend module)
- nd4j/nd4j-backends/nd4j-backend-impls/nd4j-tpu-preset/ (JavaCPP bindings)
- Google PJRT API: https://github.com/openxla/xla/tree/main/xla/pjrt/c
- Google Cloud TPU: https://cloud.google.com/tpu
- ADR 0058 - Multi-Backend Kernel Selection and Management
- ADR 0059 - Multi-Backend Op Execution System
