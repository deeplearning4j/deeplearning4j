# ADR: Hexagon-MLIR NPU Backend

## Status

Accepted

Proposed by: Development Team (March 2026)

## Context

Qualcomm Hexagon NPUs are present in most modern mobile SoCs (Snapdragon 8 Gen 3+) and edge AI platforms. Hexagon DSPs include HVX (Hexagon Vector eXtension) units optimized for INT8/INT16 quantized inference. In December 2025, Qualcomm open-sourced hexagon-mlir (BSD-3 license), an MLIR-based compiler that targets Hexagon HVX and HTP.

ND4J/SameDiff currently supports CUDA (NVIDIA GPUs), CPU, and TPU backends. Adding Hexagon support enables:

1. Running quantized SameDiff models on mobile and edge devices with Hexagon NPUs
2. Leveraging HVX vector instructions (128B or 64B vectors) for INT8 inference
3. Using TCM (Tightly Coupled Memory, 256KB-1MB) for low-latency on-chip data staging
4. Compiling operation graphs to optimized NPU kernels via hexagon-mlir

## Decision

We implement a Hexagon NPU backend using hexagon-mlir, following the DSP (DynamicShapePlan) graph backend pattern established by the CUDA/Triton backends.

### Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    ND4J API Layer                            │
│  SameDiff graph → DynamicShapePlan → Segment compilation    │
└─────────────────────────────┬───────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│              HexagonGraphBackend (C++)                       │
│  canFuseSegment() → compileSegment() → executeSegment()     │
│                                                             │
│  ┌───────────────────┐  ┌──────────────────────────────┐   │
│  │ HexagonIRBuilder   │  │ HexagonRuntimeManager        │   │
│  │ NativeSlot→MLIR    │  │ dlopen(hexagon_mlir_runtime)  │   │
│  │ HVX op mapping     │  │ NPU device management        │   │
│  │ TCM estimation     │  │ Kernel compile/dispatch       │   │
│  └───────────────────┘  └──────────────────────────────┘   │
│                                                             │
│  ┌───────────────────────────────────────────────────────┐  │
│  │ HexagonReplayHandle                                    │  │
│  │ Command list recording (capture → replay pattern)      │  │
│  │ DMA staging: DDR ↔ TCM                                │  │
│  └───────────────────────────────────────────────────────┘  │
└─────────────────────────────┬───────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    Hexagon NPU Hardware                      │
│  HVX Vector Units (128B SIMD)                               │
│  TCM (256KB-1MB on-chip SRAM)                               │
│  HTP (Hexagon Tensor Processor)                             │
└─────────────────────────────────────────────────────────────┘
```

### Memory Model: TCM Staging

Hexagon NPUs use Tightly Coupled Memory (TCM) — small, fast on-chip SRAM (256KB-1MB):

- Inputs are DMA'd from DDR to TCM before kernel execution
- HVX kernels operate on TCM-resident data at full bandwidth
- Outputs are DMA'd from TCM back to DDR after execution
- `canFuseSegment()` rejects segments whose working set exceeds TCM capacity

### HVX Vectorization

hexagon-mlir compiles MLIR operations to HVX vector instructions:

- **Elementwise ops** (add, mul, relu, sigmoid): HVX 128-byte vector operations
- **Matmul**: HVX VMPY + VACC pipeline (or Hexagon Kernel Library)
- **Reductions**: HVX horizontal reduction patterns
- **Memory ops**: DMA load/store for TCM staging

### Command List Replay

The `HexagonReplayHandle` uses command list recording (similar to CUDA graph capture):

1. `beginCapture()`: Create NPU command list, begin recording DMA + dispatch commands
2. `endCapture()`: Seal command list
3. `finalize()`: Validate and mark READY
4. `replay()`: Submit sealed command list to NPU, wait for completion

### Integration with DSP

The Hexagon backend integrates via `GraphExecutionMode.HEXAGON` (native code 14):

- `GEM_HEXAGON`: Force Hexagon compilation for all segments
- `GEM_AUTO`: Auto-detect Hexagon NPU and use when available
- `GraphReplayFactory::create()` dispatches to `HexagonReplayHandle` when `HAVE_HEXAGON_MLIR` is defined

### Module Structure

```
libnd4j/include/graph/hexagon/
├── HexagonGraphBackend.h/.cpp     # GraphBackend implementation
├── HexagonReplayHandle.h/.cpp     # GraphReplayHandle implementation
├── HexagonIRBuilder.h/.cpp        # NativeSlot → MLIR compilation
└── HexagonRuntimeManager.h/.cpp   # Runtime loading and NPU management

nd4j/nd4j-backends/nd4j-backend-impls/
├── nd4j-hexagon/                  # Java backend module
│   └── HexagonBackend, HexagonEnvironment, HexagonExecutioner
└── nd4j-hexagon-preset/           # JavaCPP bindings preset
```

### Runtime Loading

hexagon-mlir is loaded at runtime via `dlopen("libhexagon_mlir_runtime.so")`. When the library is not available (non-Hexagon systems), all operations fall back to CPU execution. This ensures the backend compiles and loads on any platform.

## Consequences

### Advantages

- **Edge AI**: Enables ND4J models on Snapdragon-powered mobile and edge devices
- **INT8 Performance**: HVX units achieve peak throughput with quantized INT8 workloads
- **Low Latency**: TCM staging eliminates DDR round-trips for small tensor operations
- **Open Source Compiler**: hexagon-mlir (BSD-3) provides a stable, community-backed compilation path
- **DSP Integration**: Reuses the existing DynamicShapePlan infrastructure for segment fusion

### Disadvantages

- **Limited Precision**: HVX is optimized for INT8/INT16; FP32 operations fall back to scalar
- **TCM Constraints**: Working set limited to 256KB-1MB, requiring careful segment sizing
- **Platform Dependency**: Only available on Qualcomm SoCs with Hexagon DSP/NPU
- **Runtime Dependency**: Requires hexagon-mlir runtime library on the target device

## References

- libnd4j/include/graph/hexagon/ (C++ backend)
- nd4j/nd4j-backends/nd4j-backend-impls/nd4j-hexagon/ (Java module)
- hexagon-mlir: https://github.com/qualcomm/hexagon-mlir
- ADR 0072 - TPU Backend
- ADR 0061 - DynamicShapePlan Execution
