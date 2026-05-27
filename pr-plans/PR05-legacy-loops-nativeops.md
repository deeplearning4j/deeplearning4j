# PR05: Legacy/Loops/NativeOps

**Estimated files:** ~176
**Merge layer:** 2
**Complexity:** Medium
**Reviewers:** Core C++ team

## Description

NativeOps JNI bridge (the main C++ entry point from Java), NativeOpExecutioner,
kernel loop infrastructure (broadcasting, reduce, scalar, pairwise, type conversions),
and the native C++ test suite. These files form the execution dispatch layer
between Java and C++ op implementations.

## File Categories

### NativeOps headers (2)
- `libnd4j/include/legacy/NativeOps.h`
- `libnd4j/include/legacy/NativeOpExecutioner.h`

### NativeOps CPU implementations (~8)
- `libnd4j/include/legacy/cpu/NativeOps.cpp`
- `libnd4j/include/legacy/cpu/NativeOps_dsp.cpp`
- `libnd4j/include/legacy/cpu/NativeOps_DataBufferLifecycle.cpp`
- `libnd4j/include/legacy/cpu/NativeOps_NDArrayLifecycle.cpp`
- `libnd4j/include/legacy/cpu/NativeOps_OpContextLifecycle.cpp`
- `libnd4j/include/legacy/cpu/NativeOpsHelpers_Arrays_delete.cpp`
- `libnd4j/include/legacy/cpu/NativeOpsHelpers_DataBuffers_close.cpp`
- `libnd4j/include/legacy/cpu/NativeOpsHelpers_DataBuffers_sync.cpp`

### NativeOps CUDA implementations (~30+)
- `libnd4j/include/legacy/cuda/NativeOps*.cu` — CUDA variants of above
- `libnd4j/include/legacy/cuda/NativeOps*.h` — CUDA-specific headers

### NativeOps shared implementations (~18)
- `libnd4j/include/legacy/impl/NativeOps_dsp_shared.cpp`
- `libnd4j/include/legacy/impl/DspRuntimeC.cpp`
- `libnd4j/include/legacy/impl/NativeOpsHelpers.cpp`
- `libnd4j/include/legacy/impl/NativeOpsHelpers_Arrays.cpp`
- `libnd4j/include/legacy/impl/NativeOpsHelpers_Context.cpp`
- `libnd4j/include/legacy/impl/NativeOpsHelpers_DataBuffers.cpp`
- `libnd4j/include/legacy/impl/NativeOpsHelpers_DataBuffers_metrics.cpp`
- `libnd4j/include/legacy/impl/NativeOpsHelpers_DataBuffers_npz.cpp`
- `libnd4j/include/legacy/impl/NativeOpsHelpers_DataBuffers_tad.cpp`
- `libnd4j/include/legacy/impl/NativeOpsHelpers_LifecycleTracking.cpp`
- `libnd4j/include/legacy/impl/NativeOpsHelpers_LifecycleTracking_Cache.cpp`
- `libnd4j/include/legacy/impl/NativeOpsHelpers_LifecycleTracking_Enable.cpp`
- `libnd4j/include/legacy/impl/NativeOpsHelpers_LifecycleTracking_Snapshot.cpp`
- `libnd4j/include/legacy/impl/NativeOpsHelpers_LifecycleTracking_Stats.cpp`
- `libnd4j/include/legacy/impl/NativeOpsHelpers_NumpyInterop.cpp`
- `libnd4j/include/legacy/impl/NativeOpsHelpers_OpTiming.cpp`
- `libnd4j/include/legacy/impl/NativeOpsHelpers_TypeConversions.cpp`

### Environment C++ implementations (~6)
- `libnd4j/include/legacy/impl/Environment.cpp`
- `libnd4j/include/legacy/impl/Environment_CudaConfig.cpp`
- `libnd4j/include/legacy/cuda/Environment_cuda.cu`
- `libnd4j/include/legacy/cuda/Environment_cuda.h`
- `libnd4j/include/legacy/cuda/Environment_CudaConfig_cuda.cu`
- `libnd4j/include/legacy/cuda/Environment_CudaConfig_cuda.h`

### CPU loop kernels (~22)
- `libnd4j/include/loops/cpu/broadcasting*.cpp`
- `libnd4j/include/loops/cpu/pairwise*.cpp`
- `libnd4j/include/loops/cpu/reduce*.cpp`
- `libnd4j/include/loops/cpu/scalar*.cpp`
- `libnd4j/include/loops/cpu/transform*.cpp`
- `libnd4j/include/loops/cpu/type_conversions.cpp`

### CUDA loop kernels (~60+)
- `libnd4j/include/loops/cuda/broadcasting*.cu`
- `libnd4j/include/loops/cuda/pairwise*.cu`
- `libnd4j/include/loops/cuda/reduce*.cu`
- `libnd4j/include/loops/cuda/scalar*.cu`
- `libnd4j/include/loops/cuda/specials*.cu`
- `libnd4j/include/loops/cuda/transform*.cu`
- `libnd4j/include/loops/cuda/type_conversions.cu`

### Loop headers (~6)
- `libnd4j/include/loops/legacy_ops.h`
- `libnd4j/include/loops/pairwise_instantiations.h`
- `libnd4j/include/loops/reduce3.h`
- `libnd4j/include/loops/summarystatsreduce.h`
- `libnd4j/include/loops/type_conversions.h`

### Native C++ tests (~7)
- `libnd4j/tests_cpu/layers_tests/ArrayOptionsTests.cpp`
- `libnd4j/tests_cpu/layers_tests/ContextTests.cpp`
- `libnd4j/tests_cpu/layers_tests/EmptyTests.cpp`
- `libnd4j/tests_cpu/layers_tests/MultiDeviceTests.cpp`
- `libnd4j/tests_cpu/layers_tests/NativeOpsTests.cpp`
- `libnd4j/tests_cpu/layers_tests/StringTests.cpp`
- `libnd4j/tests_cpu/layers_tests/CMakeLists.txt`

## Review Focus

- NativeOps.h changes affect JNI interface — must match Java presets
- DSP dispatch bridge files (NativeOps_dsp*) connect DSP to NativeOps
- Loop kernel changes affect core numerical correctness
