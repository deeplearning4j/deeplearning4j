# ADR 0087: ZLUDA Transpiler Support for AMD and Intel GPUs

## Status
Accepted

## Context

Deeplearning4j's GPU acceleration has historically been limited to NVIDIA GPUs via CUDA. This excludes users with AMD GPUs (common in workstations and data centers) and Intel GPUs (integrated and discrete Arc GPUs). Supporting these platforms natively would require:

1. Maintaining separate HIP/ROCm codebases for AMD
2. Maintaining separate SYCL/Level Zero codebases for Intel
3. Significant development and testing overhead for each platform

**ZLUDA** (https://github.com/vosen/ZLUDA) is a drop-in CUDA replacement that translates CUDA API calls at runtime to:
- **HIP** for AMD GPUs (via ROCm)
- **Level Zero** for Intel GPUs (via oneAPI)

This allows existing CUDA code to run on non-NVIDIA hardware without source modifications.

### Problem Statement

Users with AMD or Intel GPUs cannot use ND4J's GPU acceleration without:
1. Purchasing NVIDIA hardware
2. Waiting for native HIP/SYCL ports (significant engineering effort)

### Goals

1. Enable GPU acceleration on AMD GPUs with minimal code changes
2. Enable GPU acceleration on Intel GPUs with minimal code changes
3. Maintain a single CUDA codebase
4. Provide equivalent DNN operations via platform-appropriate libraries

## Decision

We implement ZLUDA transpiler support as a **drop-in replacement** approach that:

1. Compiles existing CUDA code normally
2. Links against ZLUDA runtime instead of NVIDIA CUDA runtime
3. Uses **MIOpen** (AMD's DNN library) as cuDNN replacement for AMD GPUs
4. Uses **oneDNN** (Intel's DNN library) as cuDNN replacement for Intel GPUs
5. Provides automatic ZLUDA download when not found locally

### Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           User Application                               │
│                     (Same code for all GPUs)                            │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                              ND4J Java Layer                             │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────────┐  │
│  │  JCublasBackend │  │  JZludaBackend  │  │     Nd4jBackend SPI     │  │
│  │  (NVIDIA CUDA)  │  │  (AMD/Intel)    │  │    (Auto-detection)     │  │
│  └────────┬────────┘  └────────┬────────┘  └────────────┬────────────┘  │
│           │                    │                        │               │
└───────────┼────────────────────┼────────────────────────┼───────────────┘
            │                    │                        │
            ▼                    ▼                        ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                           libnd4j Native Layer                           │
│                                                                          │
│  Engine Dispatch:  ENGINE_CUDA ──┬── ENGINE_ZLUDA_AMD                   │
│                                  └── ENGINE_ZLUDA_INTEL                  │
│                                                                          │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │                    CUDA Kernels (Shared)                         │    │
│  │           Same .cu files for NVIDIA, AMD, Intel                  │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                              │                                           │
│              ┌───────────────┼───────────────┐                          │
│              ▼               ▼               ▼                          │
│  ┌───────────────┐  ┌───────────────┐  ┌───────────────┐               │
│  │    cuDNN      │  │    MIOpen     │  │    oneDNN     │               │
│  │  (NVIDIA)     │  │    (AMD)      │  │   (Intel)     │               │
│  │  Platform Ops │  │  Platform Ops │  │  Platform Ops │               │
│  └───────────────┘  └───────────────┘  └───────────────┘               │
└─────────────────────────────────────────────────────────────────────────┘
            │                    │               │
            ▼                    ▼               ▼
┌───────────────────┐  ┌───────────────┐  ┌───────────────┐
│   NVIDIA Driver   │  │  ZLUDA + HIP  │  │ZLUDA + L0/SYCL│
│                   │  │    + ROCm     │  │   + oneAPI    │
└───────────────────┘  └───────────────┘  └───────────────┘
            │                    │               │
            ▼                    ▼               ▼
┌───────────────────┐  ┌───────────────┐  ┌───────────────┐
│   NVIDIA GPU      │  │    AMD GPU    │  │   Intel GPU   │
└───────────────────┘  └───────────────┘  └───────────────┘
```

### ZLUDA Translation Flow

```
CUDA API Call (e.g., cudaMalloc, cudaLaunchKernel)
        │
        ▼
┌─────────────────────────────────────────┐
│           ZLUDA Runtime                  │
│  ┌─────────────────────────────────┐    │
│  │  CUDA API → Internal IR          │    │
│  │  PTX/SASS → Platform IR          │    │
│  └─────────────────────────────────┘    │
└─────────────────────────────────────────┘
        │
        ├──────────────────┬──────────────────┐
        ▼                  ▼                  ▼
┌───────────────┐  ┌───────────────┐  ┌───────────────┐
│  HIP Runtime  │  │ Level Zero    │  │  (Future)     │
│  AMD ROCm     │  │ Intel oneAPI  │  │  Other APIs   │
└───────────────┘  └───────────────┘  └───────────────┘
```

## Implementation Details

### 1. Engine Types Extension

**File**: `libnd4j/include/execution/Engine.h`

```cpp
enum Engine {
  ENGINE_CPU = 0,
  ENGINE_CUDA = 1,
  ENGINE_TPU = 2,
  ENGINE_ZLUDA_AMD = 3,    // ZLUDA on AMD GPUs
  ENGINE_ZLUDA_INTEL = 4,  // ZLUDA on Intel GPUs
  // ... other engines
};

// Helper functions
inline bool isZludaEngine(Engine e) {
  return e == ENGINE_ZLUDA_AMD || e == ENGINE_ZLUDA_INTEL;
}

inline Engine getEffectiveCudaEngine(Engine e) {
  if (isZludaEngine(e)) return ENGINE_CUDA;
  return e;
}
```

### 2. CMake Configuration

**File**: `libnd4j/cmake/ZludaConfiguration.cmake`

Key functions:
- `setup_zluda()` - Main entry point, detects/downloads ZLUDA
- `setup_zluda_amd()` - AMD-specific configuration (ROCm, MIOpen)
- `setup_zluda_intel()` - Intel-specific configuration (oneAPI, oneDNN)
- `detect_zluda_target()` - Auto-detect GPU vendor via `rocminfo`/`sycl-ls`
- `configure_zluda_cuda_flags()` - CUDA flags compatible with ZLUDA
- `configure_zluda_linking()` - Link appropriate DNN libraries

**Build Options** (in `Options.cmake`):
```cmake
option(SD_ZLUDA "Enable ZLUDA transpiler support" OFF)
set(SD_ZLUDA_TARGET "AUTO" CACHE STRING "ZLUDA target (AMD, INTEL, AUTO)")
option(HELPERS_miopen "Enable MIOpen helper (AMD GPUs)" OFF)
```

### 3. Automatic ZLUDA Download

**File**: `libnd4j/cmake/Dependencies.cmake`

```cmake
function(setup_zluda_download)
    set(ZLUDA_VERSION "3")
    # Platform detection
    if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
        set(ZLUDA_PLATFORM "linux")
    elseif(CMAKE_SYSTEM_NAME STREQUAL "Windows")
        set(ZLUDA_PLATFORM "windows")
    endif()

    set(ZLUDA_URL "https://github.com/vosen/ZLUDA/releases/download/v${ZLUDA_VERSION}/zluda-${ZLUDA_PLATFORM}.tar.gz")

    ExternalProject_Add(zluda_external
        URL               "${ZLUDA_URL}"
        CONFIGURE_COMMAND ""
        BUILD_COMMAND     ""
        INSTALL_COMMAND   ${CMAKE_COMMAND} -E copy_directory <SOURCE_DIR> ${ZLUDA_INSTALL_DIR}
    )
endfunction()
```

### 4. MIOpen Platform Operations (AMD cuDNN Alternative)

**Directory**: `libnd4j/include/ops/declarable/platform/miopen/`

Files:
- `miopenUtils.h` - RAII wrappers for MIOpen descriptors
- `conv2d.cpp` - Convolution operations
- `activations.cpp` - ReLU, Sigmoid, Tanh, etc.
- `softmax.cpp` - Softmax and Log-Softmax
- `batchnorm.cpp` - Batch normalization

Example platform implementation:
```cpp
PLATFORM_IMPL(conv2d, ENGINE_ZLUDA_AMD) {
    // MIOpen-based convolution
    MIOpenTensor inputDesc(input);
    MIOpenTensor outputDesc(output);
    MIOpenConvolution convDesc(kH, kW, sH, sW, pH, pW, dH, dW);

    miopenConvolutionForward(handle, &alpha,
        inputDesc.get(), input->buffer(),
        filterDesc.get(), weights->buffer(),
        convDesc.get(), algo,
        workspace, workspaceSize,
        &beta, outputDesc.get(), output->buffer());

    return Status::OK;
}

PLATFORM_CHECK(conv2d, ENGINE_ZLUDA_AMD) {
    // Check if MIOpen can handle this configuration
    return HAVE_MIOPEN && input->dataType() == FLOAT32;
}
```

### 5. Intel ZLUDA with oneDNN Integration

For Intel GPUs, ZLUDA uses the existing oneDNN platform operations. The `setup_onednn_for_zluda()` function:

1. Checks if `onednn_interface` target exists (from main build)
2. If not, enables `HELPERS_onednn=ON` to trigger auto-download
3. Links Intel ZLUDA against the project's shared oneDNN

This ensures no duplicate oneDNN configurations and consistent behavior.

### 6. Java Backend

**Module**: `nd4j/nd4j-backends/nd4j-backend-impls/nd4j-zluda/`

**JZludaBackend.java**:
```java
public class JZludaBackend extends Nd4jBackend {

    public enum ZludaTarget { AMD, INTEL, UNKNOWN }

    @Override
    public boolean canRun() {
        String zludaPath = System.getenv("ZLUDA_PATH");
        if (zludaPath == null || zludaPath.isEmpty()) {
            return false;
        }
        return detectGpu() != ZludaTarget.UNKNOWN;
    }

    @Override
    public int getPriority() {
        // Lower than native CUDA (100), higher than CPU (0)
        return BACKEND_PRIORITY_GPU - 10;  // 90
    }

    private ZludaTarget detectGpu() {
        // Try rocminfo for AMD
        if (executeCommand("rocminfo").contains("gfx")) {
            return ZludaTarget.AMD;
        }
        // Try sycl-ls for Intel
        if (executeCommand("sycl-ls").toLowerCase().contains("intel")) {
            return ZludaTarget.INTEL;
        }
        return ZludaTarget.UNKNOWN;
    }
}
```

**Service Registration** (`META-INF/services/org.nd4j.linalg.factory.Nd4jBackend`):
```
org.nd4j.linalg.jzluda.JZludaBackend
```

### 7. Runtime Detection Flow

```
Application Start
       │
       ▼
┌─────────────────────────────────┐
│  ServiceLoader<Nd4jBackend>     │
│  loads all backend providers    │
└─────────────────────────────────┘
       │
       ├── JCublasBackend.canRun()?
       │         │
       │         ├── ZLUDA_PATH set? ──Yes──► Check ZLUDA compatibility
       │         │         │
       │         │         └──No──► Check native CUDA devices
       │         │
       │         └── Return based on detection
       │
       ├── JZludaBackend.canRun()?
       │         │
       │         ├── ZLUDA_PATH set?
       │         │         │
       │         │         ├──Yes──► Detect AMD/Intel GPU
       │         │         │              │
       │         │         │              └── Return true if found
       │         │         │
       │         │         └──No──► Return false
       │
       └── Select highest priority available backend
```

## Build Usage

### CMake (Native Build)

```bash
# AMD GPUs with MIOpen
cmake .. -DSD_CUDA=ON -DSD_ZLUDA=ON -DSD_ZLUDA_TARGET=AMD -DHELPERS_miopen=ON

# Intel GPUs with oneDNN
cmake .. -DSD_CUDA=ON -DSD_ZLUDA=ON -DSD_ZLUDA_TARGET=INTEL -DHELPERS_onednn=ON

# Auto-detect GPU vendor
cmake .. -DSD_CUDA=ON -DSD_ZLUDA=ON -DSD_ZLUDA_TARGET=AUTO
```

### Maven (Java Build)

```bash
# Build with ZLUDA support for AMD
mvn install -Pzluda-amd

# Build with ZLUDA support for Intel
mvn install -Pzluda-intel

# Generic ZLUDA build (auto-detect at runtime)
mvn install -Pzluda
```

### Runtime Environment

```bash
# Required: Point to ZLUDA installation
export ZLUDA_PATH=/opt/zluda

# AMD: Ensure ROCm is installed
export ROCM_PATH=/opt/rocm

# Intel: Ensure oneAPI is available
source /opt/intel/oneapi/setvars.sh

# Run application
java -jar myapp.jar
```

### Version-qualified Windows classifiers

The Windows ROCm 10 ZLUDA classifier is qualified by the AMD runtime
compatibility line it targets even though the Linux ROCm SDK is not embedded
on Windows. The Windows build consumes the checksum-pinned ZLUDA release asset
and must attest the complete application-local DLL closure (`nvcuda.dll`,
`nvcudart_hybrid64.dll`, `zluda_redirect.dll`, the generated ND4J/JNI DLLs,
`shared-runtime-manifest.txt`, and every additional runtime named by that
manifest) before publication. Linux-only ROCm package, HSAKMT, and kernel-pack
contracts must not be applied to this Windows classifier.

ROCm 10 therefore has distinct release contracts: Linux embeds and attests the
signed Core SDK/kpack closure, while Windows embeds and attests the pinned
ZLUDA Windows DLL bundle under the `windows-x86_64-zluda-rocm-10.0.0`
classifier. Both use the same CUDA ABI build version and record ROCm 10.0.0 in
the classifier metadata.

## Consequences

### Benefits

1. **Single Codebase**: No need to maintain separate HIP or SYCL codebases
2. **Broad GPU Support**: AMD and Intel GPUs can run existing CUDA code
3. **Minimal Changes**: Drop-in replacement approach requires minimal modifications
4. **Automatic Download**: ZLUDA auto-downloads if not found locally
5. **DNN Parity**: MIOpen and oneDNN provide equivalent DNN operations
6. **Backward Compatible**: Existing NVIDIA CUDA support unchanged

### Drawbacks

1. **Performance Overhead**: ZLUDA translation adds some runtime overhead vs native
2. **Feature Gaps**: Some advanced CUDA features may not be fully supported
3. **Dependency on ZLUDA**: Project depends on external ZLUDA development
4. **Limited Testing**: Less testing coverage than native CUDA path
5. **Debug Complexity**: Debugging issues through translation layer is harder

### Limitations

1. **ZLUDA Maturity**: ZLUDA is still evolving; some edge cases may not work
2. **MIOpen vs cuDNN**: MIOpen API differs from cuDNN; not all algorithms match
3. **Dynamic Parallelism**: CUDA dynamic parallelism may not be supported
4. **Unified Memory**: CUDA unified memory behavior may differ

### Performance Expectations

Based on ZLUDA benchmarks:
- **AMD GPUs**: 80-95% of native HIP performance for most operations
- **Intel GPUs**: 70-90% of native SYCL performance (varies by operation)
- **Memory Operations**: Near-native performance
- **Kernel Execution**: Some overhead for PTX translation

## Testing

### Unit Tests

```java
@Test
@EnabledIf("isZludaAvailable")
public void testZludaMatmul() {
    INDArray a = Nd4j.rand(100, 100);
    INDArray b = Nd4j.rand(100, 100);
    INDArray c = a.mmul(b);
    // Verify correctness against CPU reference
}
```

### Integration Tests

```bash
# Run with ZLUDA backend
ZLUDA_PATH=/opt/zluda mvn test -Pzluda-amd -Dtest=ZludaIntegrationTest
```

## References

- [ZLUDA Project](https://github.com/vosen/ZLUDA)
- [MIOpen Documentation](https://rocm.docs.amd.com/projects/MIOpen/en/latest/)
- [oneDNN Documentation](https://oneapi-src.github.io/oneDNN/)
- [ROCm Documentation](https://rocm.docs.amd.com/)
- [Intel oneAPI Documentation](https://www.intel.com/content/www/us/en/developer/tools/oneapi/overview.html)
- [Engine.h](../libnd4j/include/execution/Engine.h)
- [ZludaConfiguration.cmake](../libnd4j/cmake/ZludaConfiguration.cmake)
- [JZludaBackend.java](../nd4j/nd4j-backends/nd4j-backend-impls/nd4j-zluda/src/main/java/org/nd4j/linalg/jzluda/JZludaBackend.java)
