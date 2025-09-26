# ADR-0046: CUDA Macro Standardization

## Status

Proposed

Proposed by: Adam Gibson (27-09-2025)

## Context

LibND4J uses various preprocessor macros to detect CUDA compilation.
Different parts of codebase use different macro names:
- `__CUDACC__` - Standard NVCC compiler macro
- `__CUDABLAS__` - Custom macro for CUDA BLAS operations
- Mixed usage creates confusion and maintenance issues

Need consistent macro naming across entire codebase.
Standardize on project-specific prefix for clarity.

## Decision

Replace all `__CUDABLAS__` with `SD_CUDA`.
Use `SD_` prefix for all project-specific macros.
Keep `__CUDACC__` for direct compiler detection only.

### Macro Definitions

In common.h:
```cpp
#ifdef __CUDACC__
#define SD_CUDA
#define SD_HOST __host__
#define SD_DEVICE __device__
#define SD_KERNEL __global__
#define SD_HOST_DEVICE __host__ __device__
#else
#define SD_HOST
#define SD_DEVICE
#define SD_KERNEL
#define SD_HOST_DEVICE
#endif
```

### Usage Pattern

```cpp
// Before
#ifdef __CUDABLAS__
    // CUDA-specific code
#endif

// After
#ifdef SD_CUDA
    // CUDA-specific code
#endif
```

## Implementation

Search and replace across codebase:
- `__CUDABLAS__` → `SD_CUDA`
- Update all conditional compilation blocks
- Verify no mixed usage remains

## Consequences

### Advantages
- Consistent macro naming
- Clear project ownership with SD_ prefix
- Easier to search and maintain
- Reduces confusion between compiler and project macros

### Disadvantages
- Touches many files
- Must ensure complete replacement
- Risk of missing occurrences

### Migration
- Automated search/replace
- Manual verification of edge cases
- Update documentation

## References
- CUDA programming guide
- C++ preprocessor best practices