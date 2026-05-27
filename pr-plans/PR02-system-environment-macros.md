# PR02: System/Environment & Platform Macros

**Estimated files:** ~32
**Merge layer:** 0 (no dependencies)
**Complexity:** Medium — header changes will invalidate ccache broadly
**Reviewers:** Core C++ team

## Description

Low-level platform abstraction: Environment subsystem config refactor,
system macros (common.h, type_boilerplate.h, openmp_pragmas.h),
math utilities, type definitions (float8, omp_reductions).
These are foundational headers used by everything else.

**WARNING:** Merging this PR will trigger widespread ccache invalidation
for any downstream builds because these headers are included everywhere.
Should be merged first and given time to rebuild.

## Files (32)

### System config subsystem (12)
- `libnd4j/include/system/config/CoreConfig.h`
- `libnd4j/include/system/config/CudaDeviceConfig.h`
- `libnd4j/include/system/config/DspConfig.h`
- `libnd4j/include/system/config/EnvHelper.h`
- `libnd4j/include/system/config/LifecycleConfig.h`
- `libnd4j/include/system/config/PrintConfig.h`
- `libnd4j/include/system/config/TritonConfig.h`
- `libnd4j/include/system/config/impl/CoreConfig.cpp`
- `libnd4j/include/system/config/impl/CudaDeviceConfig.cpp`
- `libnd4j/include/system/config/impl/DspConfig.cpp`
- `libnd4j/include/system/config/impl/LifecycleConfig.cpp`
- `libnd4j/include/system/config/impl/PrintConfig.cpp`
- `libnd4j/include/system/config/impl/TritonConfig.cpp`

### System headers (9)
- `libnd4j/include/system/BackendNamespace.h`
- `libnd4j/include/system/buffer.h`
- `libnd4j/include/system/common.h`
- `libnd4j/include/system/env_functions.h`
- `libnd4j/include/system/Environment.h`
- `libnd4j/include/system/op_boilerplate.h`
- `libnd4j/include/system/openmp_pragmas.h`
- `libnd4j/include/system/RequirementsHelper.h`
- `libnd4j/include/system/sd_export.h`
- `libnd4j/include/system/type_boiler_plate_expansions.h`
- `libnd4j/include/system/type_boilerplate.h`

### Math utilities (3)
- `libnd4j/include/math/cuda_fast_math.h`
- `libnd4j/include/math/platformmath.h`
- `libnd4j/include/math/templatemath.h`

### Type definitions (3)
- `libnd4j/include/types/float8.h`
- `libnd4j/include/types/omp_reductions.h`
- `libnd4j/include/types/types.h`

### Const messages (2)
- `libnd4j/include/ConstMessages.cpp`
- `libnd4j/include/ConstMessages.h`

### ADR (1)
- `ADRs/0046 - CUDA Macro Standardization.md` — Replace mixed `__CUDABLAS__`/`__CUDACC__` with consistent `SD_`-prefixed hierarchy
