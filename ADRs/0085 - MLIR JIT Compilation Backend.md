# ADR 0057: MLIR JIT Compilation Backend

## Status
ACCEPTED

## Context

Deep learning operations can benefit from compiler-level optimizations that are difficult to achieve with hand-written kernels. LLVM's MLIR (Multi-Level Intermediate Representation) provides a flexible compiler infrastructure that enables:

- Graph-level optimizations (operator fusion, dead code elimination)
- Target-specific code generation (CPU vectorization, GPU parallelization)
- JIT compilation with runtime specialization
- Cross-platform support through LLVM backends

The existing libnd4j architecture relies on pre-compiled kernel implementations via PlatformHelpers for backends like oneDNN and cuDNN. While these provide excellent performance for common operations, they have limitations:

1. **No graph-level optimization**: Operations are executed individually without cross-operation optimization
2. **Limited fusion**: Only hand-coded fusion patterns are available
3. **Static implementations**: Kernels cannot adapt to runtime tensor shapes
4. **Backend proliferation**: Each new hardware target requires new hand-written kernels

MLIR addresses these challenges by providing a unified compiler framework that can:
- Lower high-level operations through optimized dialects (Linalg, Vector, GPU)
- Generate target-specific code at runtime
- Apply domain-specific optimizations automatically

## Decision

We implement MLIR as an optional JIT compilation backend for libnd4j that:

1. **Integrates with PlatformHelper**: MLIR implementations register as platform helpers alongside oneDNN/cuDNN
2. **Provides graph compilation**: Full SameDiff graphs can be compiled to optimized native code
3. **Supports CPU and GPU**: LLVM backend for CPU, NVVM for NVIDIA GPUs
4. **Uses runtime thresholds**: Small tensors use native implementations, large tensors use MLIR JIT

### Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                      Maven Build System                              │
│  ┌─────────────────────┐  ┌─────────────────────────────────────┐   │
│  │   libnd4j/pom.xml   │  │     buildnativeoperations.sh        │   │
│  │  - libnd4j.mlir     │──│  --mlir ON/OFF                      │   │
│  │  - libnd4j.mlir.ver │  │  --mlir-version 18                  │   │
│  │  - libnd4j.mlir.gpu │  │  --mlir-gpu ON/OFF                  │   │
│  └─────────────────────┘  └──────────────┬──────────────────────┘   │
└──────────────────────────────────────────┼──────────────────────────┘
                                           │
                                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│                        CMake Configuration                           │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │ Options.cmake                                                │    │
│  │  - HELPERS_mlir=ON/OFF                                       │    │
│  │  - MLIR_VERSION=18                                           │    │
│  │  - MLIR_ENABLE_GPU=ON/OFF                                    │    │
│  └─────────────────────────────────────────────────────────────┘    │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │ FindMLIR.cmake                                               │    │
│  │  - Locates LLVM 18+ installation                             │    │
│  │  - Creates MLIR::* imported targets                          │    │
│  │  - Validates dialect availability                            │    │
│  └─────────────────────────────────────────────────────────────┘    │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │ Dependencies.cmake                                           │    │
│  │  - setup_mlir() function                                     │    │
│  │  - Adds mlir_interface library                               │    │
│  │  - Sets HAVE_MLIR compile definition                         │    │
│  └─────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────┘
                                           │
                                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│                         C++ Core Layer                               │
│                                                                      │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │                    SD Dialect (SDOps.td)                      │   │
│  │  - Custom MLIR operations for ND4J ops                        │   │
│  │  - Type system for NDArray shapes/dtypes                      │   │
│  │  - Interfaces for shape inference                             │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                              │                                       │
│         ┌────────────────────┼────────────────────┐                 │
│         ▼                    ▼                    ▼                  │
│  ┌─────────────┐     ┌─────────────┐     ┌─────────────┐           │
│  │ SDToLinalg  │     │  SDToGPU    │     │  SDToLLVM   │           │
│  │ (CPU path)  │     │ (GPU path)  │     │ (direct)    │           │
│  └──────┬──────┘     └──────┬──────┘     └──────┬──────┘           │
│         │                   │                   │                   │
│         ▼                   ▼                   ▼                   │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                      MLIREngine                              │   │
│  │  - JIT compilation via LLVM ORC                              │   │
│  │  - Kernel caching by operation signature                     │   │
│  │  - Pass pipeline management                                  │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                              │                                       │
│                              ▼                                       │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │               PlatformHelper Implementations                  │   │
│  │  platform/mlir/                                               │   │
│  │  ├── blas/          matmul, gemm, xw_plus_b                  │   │
│  │  ├── nn/            conv2d, pooling, batch_norm              │   │
│  │  ├── transforms/    activations (relu, sigmoid, etc.)        │   │
│  │  ├── reduce/        sum, mean, max, min, variance            │   │
│  │  ├── attention/     dot_product_attention, multi_head        │   │
│  │  ├── shape/         reshape, transpose, concat, slice        │   │
│  │  ├── elementwise/   add, mul, binary_ops, unary_ops          │   │
│  │  ├── rnn/           lstm, gru, simple_rnn                    │   │
│  │  ├── loss/          cross_entropy, huber, ctc                │   │
│  │  ├── embedding/     embedding_lookup, one_hot, segment_*     │   │
│  │  └── image/         resize, crop_and_resize, color_convert   │   │
│  └─────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
```

## Implementation Details

### 1. Build System Integration

#### Maven Configuration (libnd4j/pom.xml)

Properties for MLIR configuration:
```xml
<properties>
    <!-- MLIR Configuration -->
    <libnd4j.mlir>OFF</libnd4j.mlir>
    <libnd4j.mlir.version>18</libnd4j.mlir.version>
    <libnd4j.mlir.gpu>OFF</libnd4j.mlir.gpu>
</properties>
```

Maven profiles for easy activation:
```xml
<!-- MLIR JIT Compilation Profile -->
<profile>
    <id>mlir</id>
    <activation>
        <property>
            <name>libnd4j.mlir</name>
            <value>ON</value>
        </property>
    </activation>
    <properties>
        <libnd4j.mlir>ON</libnd4j.mlir>
        <libnd4j.mlir.version>18</libnd4j.mlir.version>
    </properties>
</profile>

<!-- MLIR with GPU Support Profile -->
<profile>
    <id>mlir-gpu</id>
    <activation>
        <property>
            <name>libnd4j.mlir.gpu</name>
            <value>ON</value>
        </property>
    </activation>
    <properties>
        <libnd4j.mlir>ON</libnd4j.mlir>
        <libnd4j.mlir.gpu>ON</libnd4j.mlir.gpu>
    </properties>
</profile>
```

Build profile arguments passed to buildnativeoperations.sh:
```xml
<argument>--mlir</argument>
<argument>${libnd4j.mlir}</argument>
<argument>--mlir-version</argument>
<argument>${libnd4j.mlir.version}</argument>
<argument>--mlir-gpu</argument>
<argument>${libnd4j.mlir.gpu}</argument>
```

#### Build Script (buildnativeoperations.sh)

Variable defaults:
```bash
# MLIR JIT compilation variables
MLIR="${MLIR:-OFF}"
MLIR_VERSION="${MLIR_VERSION:-18}"
MLIR_GPU="${MLIR_GPU:-OFF}"
```

Argument parsing:
```bash
--mlir)
    MLIR="$value"
    if [[ "$value" == "ON" ]]; then
        print_colored "green" "✓ MLIR JIT compilation enabled"
    fi
    shift
    ;;
--mlir-version)
    MLIR_VERSION="$value"
    print_colored "blue" "✓ MLIR/LLVM version: $value"
    shift
    ;;
--mlir-gpu)
    MLIR_GPU="$value"
    if [[ "$value" == "ON" ]]; then
        print_colored "green" "✓ MLIR GPU backend enabled"
    fi
    shift
    ;;
```

CMake argument generation:
```bash
MLIR_ARG=""
if [ "$MLIR" == "ON" ]; then
    MLIR_ARG="-DHELPERS_mlir=ON -DMLIR_VERSION=${MLIR_VERSION}"
    if [ "$MLIR_GPU" == "ON" ]; then
        MLIR_ARG="${MLIR_ARG} -DMLIR_ENABLE_GPU=ON"
    fi
fi
```

#### CMake Configuration

**Options.cmake**:
```cmake
option(HELPERS_mlir "Enable MLIR/LLVM JIT compilation helper" OFF)
set(MLIR_VERSION "18" CACHE STRING "MLIR/LLVM minimum version (18+)")
option(MLIR_ENABLE_GPU "Enable MLIR GPU dialect and NVVM backend" OFF)
```

**FindMLIR.cmake** searches for LLVM/MLIR in standard locations:
- `/usr/lib/llvm-18`
- `/usr/local/opt/llvm@18` (Homebrew on macOS)
- Custom paths via `LLVM_ROOT`

**Dependencies.cmake** provides `setup_mlir()` function:
```cmake
function(setup_mlir)
    if(NOT HELPERS_mlir STREQUAL "ON")
        message(STATUS "MLIR helper is disabled")
        return()
    endif()

    find_package(LLVM ${MLIR_VERSION} REQUIRED CONFIG)
    find_package(MLIR REQUIRED CONFIG)

    # Create interface library
    add_library(mlir_interface INTERFACE)
    target_link_libraries(mlir_interface INTERFACE
        MLIR::MLIRLinalgDialect
        MLIR::MLIRVectorDialect
        MLIR::MLIRExecutionEngine
        # ... additional dialects
    )

    set(HAVE_MLIR TRUE PARENT_SCOPE)
    add_compile_definitions(HAVE_MLIR=1)
endfunction()
```

### 2. SD Dialect Definition

**Files**: `libnd4j/include/mlir/dialect/SD/`

The SD (SameDiff) dialect defines MLIR operations corresponding to ND4J ops:

```tablegen
// SDOps.td
def SD_Dialect : Dialect {
    let name = "sd";
    let cppNamespace = "::mlir::sd";
    let summary = "SameDiff operations for ND4J";
}

def SD_MatMulOp : SD_Op<"matmul", [NoMemoryEffect]> {
    let summary = "Matrix multiplication";
    let arguments = (ins
        AnyTensor:$a,
        AnyTensor:$b,
        DefaultValuedAttr<BoolAttr, "false">:$transposeA,
        DefaultValuedAttr<BoolAttr, "false">:$transposeB
    );
    let results = (outs AnyTensor:$result);
}

def SD_Conv2DOp : SD_Op<"conv2d", [NoMemoryEffect]> {
    let summary = "2D Convolution";
    let arguments = (ins
        AnyTensor:$input,
        AnyTensor:$weights,
        Optional<AnyTensor>:$bias,
        I64ArrayAttr:$strides,
        I64ArrayAttr:$padding,
        I64ArrayAttr:$dilation,
        StrAttr:$dataFormat
    );
    let results = (outs AnyTensor:$output);
}
```

### 3. Lowering Pipeline

**CPU Path**:
```
SD Dialect
    ↓ (canonicalization)
SD Dialect (optimized)
    ↓ (fusion patterns)
SD Dialect (fused ops)
    ↓ (SDToLinalg)
Linalg Dialect
    ↓ (LinalgToLoops)
SCF Dialect
    ↓ (vectorization)
Vector Dialect
    ↓ (LLVMDialect lowering)
LLVM Dialect
    ↓ (LLVM backend)
Native Code
```

**GPU Path**:
```
SD Dialect
    ↓ (canonicalization + fusion)
SD Dialect (optimized)
    ↓ (SDToGPU)
GPU Dialect
    ↓ (GPU to NVVM)
NVVM Dialect
    ↓ (LLVM NVPTX backend)
PTX Code
```

### 4. MLIREngine Runtime

**Files**: `libnd4j/include/mlir/runtime/MLIREngine.h`

```cpp
class MLIREngine {
public:
    static MLIREngine& getInstance();

    // Compile operation to native kernel
    CompiledKernel compile(const std::string& opName,
                           const std::vector<TensorDesc>& inputs,
                           const std::vector<TensorDesc>& outputs);

    // Get or compile kernel with caching
    CompiledKernel getOrCompile(const std::string& opName,
                                 graph::Context& block);

    // Execute compiled kernel
    Status execute(CompiledKernel& kernel,
                   const std::vector<NDArray*>& inputs,
                   const std::vector<NDArray*>& outputs);

private:
    void buildCPUPipeline(mlir::PassManager& pm);
    void buildGPUPipeline(mlir::PassManager& pm);

    std::unique_ptr<mlir::MLIRContext> context_;
    std::unique_ptr<mlir::ExecutionEngine> engine_;
    KernelCache cache_;
};
```

### 5. PlatformHelper Integration

**Files**: `libnd4j/include/ops/declarable/platform/mlir/`

Each operation follows the standard PlatformHelper pattern:

```cpp
// platform/mlir/blas/matmul.cpp

#include <ops/declarable/platform/mlir/mlirUtils.h>

#if defined(HAVE_MLIR)

namespace sd::ops::platforms {

DECLARE_PLATFORM(matmul, ENGINE_CPU)

PLATFORM_IMPL(matmul, ENGINE_CPU) {
    auto* a = INPUT_VARIABLE(0);
    auto* b = INPUT_VARIABLE(1);
    auto* c = OUTPUT_VARIABLE(0);

    bool transposeA = block.numB() > 0 ? B_ARG(0) : false;
    bool transposeB = block.numB() > 1 ? B_ARG(1) : false;

    std::vector<NDArray*> inputs = {a, b};
    std::vector<NDArray*> outputs = {c};

    auto status = executeMlir("matmul", block, inputs, outputs);

    if (status != Status::OK()) {
        return Status::CODE(ND4J_STATUS_BAD_ARGUMENTS, "MLIR matmul failed");
    }

    return Status::OK();
}

PLATFORM_CHECK(matmul, ENGINE_CPU) {
    auto* a = INPUT_VARIABLE(0);

    Requirements req("MLIR MATMUL");

    req.expectTrue(mlirEnabled(), "MLIR enabled") &&
    req.expectIn(a->dataType(), {FLOAT32, DOUBLE, HALF, BFLOAT16}, "Supported dtype") &&
    req.expectTrue(a->lengthOf() >= MLIR_MIN_TENSOR_SIZE, "Size threshold") &&
    req.expectTrue(a->ews() == 1 || a->ews() == 0, "Contiguous memory");

    return req;
}

} // namespace sd::ops::platforms

#endif // HAVE_MLIR
```

### 6. Utility Functions

**mlirUtils.h**:
```cpp
namespace sd::ops::platforms {

// Check if MLIR is available and enabled
bool mlirEnabled();

// Minimum tensor size for MLIR acceleration (default: 1024 elements)
constexpr int MLIR_MIN_TENSOR_SIZE = 1024;

// Execute operation via MLIR
Status executeMlir(const std::string& opName,
                   graph::Context& block,
                   const std::vector<NDArray*>& inputs,
                   const std::vector<NDArray*>& outputs);

} // namespace sd::ops::platforms
```

### 7. Supported Operations

| Category | Operations |
|----------|-----------|
| **BLAS** | matmul, gemm, xw_plus_b |
| **Convolution** | conv1d, conv2d, conv3d, depthwise_conv2d, separable_conv2d, deconv2d, deconv3d |
| **Pooling** | maxpool2d, avgpool2d, maxpool3d, avgpool3d, global_avg_pool, global_max_pool |
| **Normalization** | batchnorm, layernorm, batch_norm_bp |
| **Activation** | relu, sigmoid, tanh, gelu, softmax, hardswish, mish, selu, prelu, log_softmax |
| **Reduction** | reduce_sum, reduce_mean, reduce_max, reduce_min, reduce_prod, reduce_variance, reduce_logsumexp |
| **Attention** | dot_product_attention, multi_head_attention, self_attention (+ backprops) |
| **Shape** | reshape, transpose, concat, slice, strided_slice, gather, scatter_update, tile, split, stack |
| **Elementwise** | add, subtract, multiply, divide, maximum, minimum, pow, exp, log, sqrt, abs, neg |
| **Comparison** | equals, not_equals, greater, less, where, logical_and/or/xor/not, isnan, isinf |
| **RNN** | lstmLayer, gruCell, gru, simple_rnn, lstmCell (+ backprops) |
| **Loss** | softmax_cross_entropy, sigmoid_cross_entropy, huber_loss, ctc_loss, cosine_distance |
| **Embedding** | embedding_lookup, one_hot, segment_sum/mean/max/min, unique, top_k, in_top_k |
| **Image** | resize_bilinear, resize_bicubic, crop_and_resize, rgb_to_hsv, adjust_contrast/hue/saturation |

## Build Commands

### CPU with MLIR
```bash
# Maven
mvn clean install -Dlibnd4j.mlir=ON

# Direct build script
./buildnativeoperations.sh --mlir ON --mlir-version 18
```

### GPU with MLIR
```bash
# Maven
mvn clean install -Dlibnd4j.mlir=ON -Dlibnd4j.mlir.gpu=ON -Dsd.cuda=ON

# Direct build script
./buildnativeoperations.sh --mlir ON --mlir-version 18 --mlir-gpu ON -c cuda
```

### Multi-Helper Build (MLIR + oneDNN)
```bash
mvn clean install -Dlibnd4j.mlir=ON --helpers onednn,mlir
```

## Consequences

### Benefits

1. **Graph-Level Optimization**: MLIR can optimize across operation boundaries, fusing operations and eliminating intermediate allocations.

2. **Automatic Vectorization**: The Linalg-to-Vector lowering automatically generates SIMD code for the target architecture.

3. **Hardware Portability**: Same high-level code generates optimized kernels for different CPU architectures (x86, ARM, etc.).

4. **GPU Support**: Single dialect can lower to both CPU (LLVM) and GPU (NVVM) backends.

5. **Extensibility**: New operations can be added by defining TableGen ops and lowering patterns.

6. **JIT Specialization**: Kernels can be specialized for specific tensor shapes at runtime.

7. **Integration with Ecosystem**: MLIR is actively developed by Google, NVIDIA, and others for ML workloads.

### Drawbacks

1. **Build Complexity**: Requires LLVM 18+ with MLIR, which may not be available on all systems.

2. **Compilation Overhead**: JIT compilation adds latency to first execution of each operation variant.

3. **Binary Size**: MLIR libraries add significant size to the native library.

4. **Debugging**: Generated code is harder to debug than hand-written kernels.

5. **LLVM Dependency**: Tight coupling to LLVM version may cause compatibility issues.

### Runtime Behavior

- MLIR helpers register alongside other backends (oneDNN, cuDNN)
- Size threshold (`MLIR_MIN_TENSOR_SIZE = 1024`) prevents MLIR overhead on small tensors
- Compiled kernels are cached by operation signature for reuse
- Falls back to native implementation if MLIR compilation fails

## Dependencies

- **LLVM 18+**: Core LLVM libraries and MLIR infrastructure
- **CMake 3.20+**: For modern target-based configuration
- **TableGen**: For generating dialect code (bundled with LLVM)
- **(Optional) CUDA Toolkit**: For GPU backend via NVPTX

## References

- [MLIR Documentation](https://mlir.llvm.org/)
- [Linalg Dialect](https://mlir.llvm.org/docs/Dialects/Linalg/)
- [GPU Dialect](https://mlir.llvm.org/docs/Dialects/GPU/)
- [LLVM ORC JIT](https://llvm.org/docs/ORCv2.html)
- [PlatformHelper Architecture](../libnd4j/include/ops/declarable/PlatformHelper.h)
- [ADR 0055: Kernel Selection and Dynamic Loading](0055-Kernel_Selection_And_Dynamic_Loading.md)
