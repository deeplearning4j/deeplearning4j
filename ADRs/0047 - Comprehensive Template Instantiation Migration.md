# ADR: Comprehensive Template Instantiation Migration for Type Alias Coverage

## Status

Implemented

Proposed by: Adam Gibson (September 2025)

Discussed with: Development Team

## Context

Despite the successful implementation of the Selective Rendering Type System (ADR-0039), libnd4j continued to experience persistent undefined reference errors during linking. These errors revealed a fundamental challenge in C++ template instantiation that our previous approach hadn't fully addressed: type aliasing.

The core issue is that C++ allows multiple type names to refer to the same underlying type, and these relationships vary across platforms:

**Platform-Specific Type Aliasing**: Consider how 64-bit integers are defined:
- On 64-bit Linux: `long` and `int64_t` are often the same type
- On 64-bit Windows: `long` remains 32-bit while `int64_t` is 64-bit  
- On 32-bit systems: Different relationships emerge entirely

**Multiple Names, One Type**: Our codebase uses various aliases:
- `LongType` → platform-specific 64-bit integer
- `sd::LongType` → namespaced version
- `SignedChar`/`UnsignedChar` → explicit signedness
- Standard variations: `long`, `long long`, `int64_t`

**The Template Instantiation Problem**: When code uses `long` but we only instantiated templates for `int64_t`, the linker can't find the required symbols - even though they might be the same type on that platform. This led to frustrating platform-specific build failures that were difficult to reproduce and debug.

The previous macro system (`BUILD_SINGLE_TEMPLATE`, etc.) would only instantiate templates for the exact types specified in our lists. If a developer wrote `PairWiseTransform<long, long, long>`, but our instantiation list only contained `int64_t`, the build would fail with undefined reference errors - even on platforms where these types are identical.

## Decision

Implement a comprehensive template instantiation system that automatically generates all type alias variants for each semantic type, fully integrated with the selective rendering system from ADR-0039.

### Architecture Overview

The solution involves several interconnected components:

**1. Enhanced TemplateProcessing.cmake**

We extend the build system to understand type equivalence classes:

```cmake
function(get_all_type_variants type all_variants_var)
    # Platform-aware type detection
    if(CMAKE_SIZEOF_VOID_P EQUAL 8)
        # 64-bit platforms
        set(INT64_CLASS "int64_t;long long;long;sd::LongType;LongType")
        set(UINT64_CLASS "uint64_t;unsigned long long;unsigned long;sd::UnsignedLong")
    else()
        # 32-bit platforms
        set(INT64_CLASS "int64_t;long long;sd::LongType;LongType")
        set(UINT64_CLASS "uint64_t;unsigned long long;sd::UnsignedLong")
    endif()
    
    # Map input type to all its variants
    # ... implementation details ...
endfunction()
```

**2. Type Normalization System**

To prevent duplicate instantiations, we normalize types to canonical forms:

```cmake
function(normalize_to_canonical_type cpp_type canonical_var)
    # Map all aliases to a single canonical form
    # long, long long, int64_t, LongType → LongType
    # This ensures deduplication while maintaining coverage
endfunction()
```

**3. Integration with Selective Rendering**

The system works seamlessly with our existing semantic filtering:

```cmake
function(handle_pairwise t1 t2 t3 content_var is_cuda)
    # Get all variants for each type
    get_all_type_variants(${t1} t1_variants)
    get_all_type_variants(${t2} t2_variants)
    get_all_type_variants(${t3} t3_variants)
    
    # For each combination of variants
    foreach(v1 ${t1_variants})
        foreach(v2 ${t2_variants})
            foreach(v3 ${t3_variants})
                # Normalize to check if already processed
                normalize_to_canonical_type(${v1} c1)
                normalize_to_canonical_type(${v2} c2)
                normalize_to_canonical_type(${v3} c3)
                
                # Apply semantic filtering rules
                if(is_valid_combination(${c1} ${c2} ${c3}))
                    # Generate instantiation
                endif()
            endforeach()
        endforeach()
    endforeach()
endfunction()
```

**4. Sophisticated Macro System**

The header implementation provides the template expansion machinery:

```cpp
// Platform detection
#define SD_INT64_IS_LONG (std::is_same<int64_t, long>::value)
#define SD_INT64_IS_LONG_LONG (std::is_same<int64_t, long long>::value)

// Type expansion macros
#define EXPAND_INT64_VARIANTS(MACRO, ...) \
    MACRO(int64_t, __VA_ARGS__) \
    MACRO(long long, __VA_ARGS__) \
    MACRO(long, __VA_ARGS__) \
    MACRO(LongType, __VA_ARGS__)

// Conditional instantiation
#define _RANDOMSINGLE(TEMPLATE_NAME, SIGNATURE, ENUM, TYPE) \
    EVAL(SD_IF_SINGLE_ALIAS_COMPILED_DECL( \
        ENUM, \
        CONCAT(_EXPAND_RANDOMSINGLE_, TYPE)(TEMPLATE_NAME, SIGNATURE) \
    ))
```

### How It Works in Practice

When building operations, the system:

1. **Discovers Types**: Parses types.h to find all defined types
2. **Maps Equivalences**: Groups types into equivalence classes
3. **Generates Combinations**: Creates all valid type combinations per selective rendering rules
4. **Expands Aliases**: For each combination, generates all alias variants
5. **Deduplicates**: Ensures we don't instantiate the same thing twice
6. **Chunks Output**: Splits instantiations across files for manageable compilation

Example transformation:
```cpp
// Developer writes:
template void PairWiseTransform<long, long, long>::exec(...);

// System generates (on 64-bit Linux where long == int64_t):
template void PairWiseTransform<long, long, long>::exec(...);
template void PairWiseTransform<int64_t, int64_t, int64_t>::exec(...);
template void PairWiseTransform<long long, long long, long long>::exec(...);
template void PairWiseTransform<LongType, LongType, LongType>::exec(...);
template void PairWiseTransform<sd::LongType, sd::LongType, sd::LongType>::exec(...);
```

## Implementation Details

### Memory Management During Builds

Large template instantiation sets require careful memory management:

```cmake
# Adaptive chunking based on available memory
cmake_host_system_information(RESULT AVAILABLE_MEMORY QUERY AVAILABLE_PHYSICAL_MEMORY)
if(AVAILABLE_MEMORY LESS 4000)
    set(CHUNK_TARGET_INSTANTIATIONS 3)
    message(STATUS "Low memory detected: Using conservative chunking")
else()
    set(CHUNK_TARGET_INSTANTIATIONS 5)
endif()
```

### Diagnostic Infrastructure

Comprehensive diagnostics help debug type issues:
```
build/type_combinations/
├── active_types.txt              # All types being compiled
├── type_equivalences.txt         # Platform-specific type mappings
├── combinations_2_expanded.txt   # All 2-type combinations with aliases
├── deduplication_stats.txt      # Statistics on eliminated duplicates
└── type_platform_report.txt     # Platform-specific type size information
```

## Consequences

### Advantages

**Eliminated Link Errors**: The most significant benefit - undefined reference errors are now virtually eliminated. Code that uses platform-specific types "just works" across all supported platforms.

**True Platform Portability**: Developers can write natural C++ code using standard types without worrying about our internal type system. Whether they use `long`, `int64_t`, or `LongType`, the templates are there.

**Automated Maintenance**: New type aliases are automatically discovered and included. No manual tracking of platform variations required.

**Preserved Optimization Benefits**: We retain all the binary size and compilation time benefits from selective rendering while solving the alias problem.

**Better Developer Experience**: Fewer cryptic linker errors mean less time debugging build issues and more time developing features.

### Disadvantages

**Increased Build Resources**: The comprehensive type coverage comes at a cost:
- Build times increased 2-3x compared to minimal type coverage
- Memory usage during compilation can exceed 16GB for parallel builds
- CI/CD pipelines require beefier build machines

**Larger Intermediate Artifacts**: More template instantiations mean:
- Larger object files during compilation
- Increased disk space requirements
- Slower incremental builds in some cases

**System Complexity**: The template processing system is now significantly more complex:
- Multiple levels of indirection in the build system
- Sophisticated macro machinery that can be hard to debug
- Requires deep understanding to modify

**Potential Over-Instantiation**: We generate templates for some type combinations that may never be used in practice:
- No feedback mechanism to identify unused instantiations
- Binary size is larger than theoretically necessary
- Link-time optimization (LTO) can only partially mitigate this

### Trade-off Analysis

The decision to implement comprehensive type alias coverage represents a deliberate trade-off:

**What We Gained**:
- Robust, platform-independent builds
- Eliminated a major source of developer frustration
- True write-once, compile-anywhere for template code

**What We Paid**:
- Longer build times
- Higher resource requirements
- Increased system complexity

In practice, this trade-off has proven worthwhile. The time saved debugging link errors far exceeds the additional build time, and modern build servers can handle the resource requirements.

## Conclusion

The comprehensive template instantiation migration successfully solves a fundamental challenge in libnd4j's template-heavy architecture. By understanding and embracing C++'s type alias complexity rather than fighting it, we've created a robust system that "just works" across diverse platforms and use cases.

While the solution increases build complexity and resource requirements, it provides a solid foundation for libnd4j's continued evolution. The elimination of mysterious linker errors alone justifies the investment, and the system's automatic handling of new type aliases ensures it remains maintainable as the codebase grows.

This work, combined with the selective rendering system, represents a mature approach to managing template instantiation in large C++ projects - balancing theoretical purity with practical developer needs.

## References

- C++ Standard: Type Aliases and typedef
- Platform ABI Documentation (Linux, Windows, macOS)
- CMake Cross-Platform Build Guide
- Internal Build Performance Metrics (2025)