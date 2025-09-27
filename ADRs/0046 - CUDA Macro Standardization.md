# ADR: CUDA Macro Standardization

## Status

Proposed

Proposed by: Adam Gibson (September 2025)

Discussed with: Development Team

## Context

LibND4J's codebase has accumulated inconsistent preprocessor macro usage for CUDA compilation detection over the years. We currently use a mix of different macros throughout the codebase:

- `__CUDACC__` - The standard NVCC compiler macro
- `__CUDABLAS__` - A custom macro we introduced for CUDA BLAS operations
- Mixed usage of both in different files

This inconsistency creates several problems. Developers aren't sure which macro to use when adding new CUDA code. Code reviews frequently catch inconsistent usage. Searching for CUDA-specific code requires checking multiple macro names. Most importantly, the distinction between these macros has become meaningless - they're effectively used interchangeably.

The `__CUDABLAS__` name is particularly confusing because it suggests a relationship to CUBLAS specifically, when in reality it's used for general CUDA compilation detection. This historical artifact no longer serves any purpose and actively hinders code clarity.

## Decision

We will standardize on project-specific macros with the `SD_` prefix (Samediff/Deeplearning4j) for all CUDA-related conditional compilation. Specifically:

1. Replace all instances of `__CUDABLAS__` with `SD_CUDA`
2. Use `__CUDACC__` only for direct compiler detection in common headers
3. Define a consistent set of CUDA-related macros in `common.h`

### Macro Architecture

In `common.h`, we'll establish a clear hierarchy:

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

This approach:
- Clearly indicates project-specific macros with the `SD_` prefix
- Provides semantic macros for CUDA attributes
- Allows CUDA-agnostic code when CUDA isn't available
- Creates a single source of truth for CUDA detection

### Usage Patterns

The migration is straightforward:

```cpp
// Before
#ifdef __CUDABLAS__
    // CUDA-specific implementation
#endif

// After
#ifdef SD_CUDA
    // CUDA-specific implementation
#endif
```

Function decorators become more readable:

```cpp
// Before
#ifdef __CUDACC__
__host__ __device__
#endif
void myFunction() { ... }

// After
SD_HOST_DEVICE void myFunction() { ... }
```

## Implementation Plan

### Phase 1: Preparation
1. Add new macro definitions to `common.h`
2. Verify build system passes correct defines
3. Test on both CUDA and CPU builds

### Phase 2: Migration
1. Automated search and replace: `__CUDABLAS__` → `SD_CUDA`
2. Manual review of `__CUDACC__` usage
3. Update function decorators to use semantic macros

### Phase 3: Validation
1. Full build matrix testing
2. Verify no behavioral changes
3. Update coding guidelines

## Consequences

### Advantages

**Consistency**: A single, clear convention for CUDA detection removes ambiguity. Developers know exactly which macro to use, and code reviews become simpler.

**Searchability**: Finding all CUDA-specific code becomes trivial with a single macro name. This helps with maintenance and understanding code organization.

**Semantic Clarity**: The `SD_` prefix clearly indicates project-specific macros, distinguishing them from compiler-provided ones. The name `SD_CUDA` is self-explanatory, unlike `__CUDABLAS__`.

**Portability**: By wrapping CUDA attributes in macros, we make the code more portable. Non-CUDA builds can compile the same source files without modification.

**Future Flexibility**: Establishing a clear macro architecture makes it easier to add new backend support (ROCm, Intel GPU, etc.) using similar patterns.

### Disadvantages

**Large Diff**: This change touches many files across the codebase, creating a large commit that may complicate git history and blame tracking.

**Merge Conflicts**: Outstanding pull requests may need rebasing after this change lands.

**Learning Curve**: Developers familiar with the old macros need to adjust to the new convention, though this is minimal given the straightforward mapping.

### Risk Mitigation

The change is largely mechanical and low-risk:
- Simple string replacement for most cases
- No functional changes to code logic
- Compiler will catch any missed replacements
- Can be done incrementally if needed

## Conclusion

Standardizing on `SD_CUDA` and related macros is a simple change that provides lasting benefits. It makes the codebase more consistent, searchable, and maintainable while setting a clear pattern for future backend support. The temporary inconvenience of a large diff is outweighed by the long-term improvements to code clarity and developer experience.

## References

- CUDA C++ Programming Guide - Preprocessor Definitions
- C++ Core Guidelines - Macro Usage
- Internal coding standards documentation