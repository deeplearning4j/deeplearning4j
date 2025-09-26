# ADR-0047: Comprehensive Template Instantiation Migration for Type Alias Coverage

## Status

Implemented

Proposed by: Adam Gibson (September 2025)

## Context

The libnd4j project has experienced persistent undefined reference errors during linking due to incomplete template instantiation coverage. While ADR-0039 (Selective Rendering Type System) successfully reduced binary size through semantic filtering of type combinations, it did not fully address the complex issue of C++ type aliasing where multiple type names can refer to the same underlying type.

The problem manifests in several ways:

1. **Platform-Dependent Type Aliases**: On different platforms, types like `long`, `long long`, and `int64_t` may or may not be distinct types. For example:
   - On 64-bit Linux: `long` and `int64_t` are often the same type
   - On 64-bit Windows: `long` is 32-bit while `int64_t` is 64-bit
   - On 32-bit systems: `long` and `int` may be the same type

2. **Library-Specific Type Aliases**: The codebase uses various type aliases:
   - `LongType`, `sd::LongType` aliasing to platform-specific 64-bit integers
   - `SignedChar`, `UnsignedChar` for explicit signed/unsigned character types
   - `Int32Type` and other convenience aliases

3. **Incomplete Template Coverage**: The previous macro system (BUILD_SINGLE_TEMPLATE, etc.) would only instantiate templates for the exact types specified, missing critical aliases. This led to:
   - Undefined references when code used `long` but only `int64_t` was instantiated
   - Link failures when mixing libraries compiled with different type assumptions
   - Platform-specific build failures that were difficult to reproduce

4. **Maintenance Burden**: Manually tracking all type aliases and their platform-specific variations was error-prone and led to frequent build failures as new code was added.

## Decision

Implement a comprehensive template instantiation system that automatically generates all type alias variants for each semantic type, integrated with the selective rendering system from ADR-0039.

### Key Components

#### 1. Enhanced TemplateProcessing.cmake

The new system extends the template processing to handle type equivalence classes:

```cmake
# Get ALL type variants for a given type
function(get_all_type_variants type all_variants_var)
    # Platform-specific type detection
    if(CMAKE_SIZEOF_VOID_P EQUAL 8)
        set(INT64_CLASS "int64_t;long long;long;sd::LongType;LongType")
        set(UINT64_CLASS "uint64_t;unsigned long long;unsigned long;sd::UnsignedLong;UnsignedLong")
    else()
        set(INT64_CLASS "int64_t;long long;sd::LongType;LongType")
        set(UINT64_CLASS "uint64_t;unsigned long long;sd::UnsignedLong;UnsignedLong")
    endif()
    
    # Check which equivalence class the type belongs to
    # Return ALL variants of that type
endfunction()
```

#### 2. Type Normalization System

A canonical type system ensures consistent handling across the codebase:

```cmake
function(normalize_to_canonical_type cpp_type canonical_var)
    # Map all type aliases to a single canonical form
    # e.g., long, long long, int64_t, LongType → LongType
    # This ensures deduplication while maintaining coverage
endfunction()
```

#### 3. Template Handler Functions

Each template type has a specific handler that generates all necessary instantiations:

```cpp
// Example: handle_pairwise generates all alias combinations
function(handle_pairwise t1 t2 t3 content_var is_cuda)
    # Normalize types to canonical forms
    # Check semantic validity using selective rendering rules
    # Generate instantiations for the exact types requested
    # The macro system handles alias expansion
endfunction()
```

#### 4. Integration with ITERATE_COMBINATIONS Macros

The system leverages the existing macro infrastructure from ADR-0031 but extends it to handle type aliases:

```cpp
// The dispatch_to_handler function processes type lists
function(dispatch_to_handler template_name t1 t2 t3 parts_count content_var is_cuda)
    # Parse semicolon-separated type lists (all aliases)
    # For each combination of aliases:
    #   - Normalize to canonical form
    #   - Check if already processed (deduplication)
    #   - Apply semantic filtering rules
    #   - Generate instantiation if valid
endfunction()
```

#### 5. Comprehensive Type Coverage

The system ensures all type variants are covered:

- **Integer Types**: All platform variations of int8_t, int16_t, int32_t, int64_t and their unsigned counterparts
- **Floating Point**: float16, bfloat16, float, double (no aliases needed)
- **Special Types**: bool, string types (std::string, std::u16string, std::u32string)
- **Library Aliases**: LongType, UnsignedLong, SignedChar, etc.

### Implementation Strategy

1. **Type Discovery Phase**: 
   - Parse types.h to find all defined types
   - Map types to their equivalence classes
   - Generate comprehensive type lists with all aliases

2. **Combination Generation Phase**:
   - Use selective rendering rules to filter valid combinations
   - For each valid combination, expand to all alias variants
   - Apply deduplication to avoid redundant instantiations

3. **Code Generation Phase**:
   - Generate chunked files to manage compilation memory
   - Each file contains a balanced set of instantiations
   - Platform-specific handling ensures correct behavior

4. **Build Integration**:
   - Transparent integration with existing CMake infrastructure
   - Cached results for faster incremental builds
   - Diagnostic output for debugging type issues

## Consequences

### Advantages

1. **Eliminated Undefined References**: 
   - Complete coverage of all type aliases prevents link errors
   - Platform-independent builds work reliably
   - No more "undefined reference to PairWiseTransform<long, long, long>" errors

2. **Automated Maintenance**:
   - No manual tracking of type aliases required
   - New aliases automatically included in builds
   - Platform differences handled transparently

3. **Preserved Selective Rendering Benefits**:
   - Still filters out invalid type combinations
   - Binary size controlled through semantic rules
   - Type profiles (training, inference, etc.) still work

4. **Better Platform Portability**:
   - Code using platform-specific types (long, size_t) works correctly
   - No need for platform-specific template instantiation lists
   - Consistent behavior across Linux, Windows, macOS

5. **Improved Developer Experience**:
   - Fewer build failures during development
   - Clear diagnostic messages for type issues
   - Type alias usage is transparent

### Disadvantages

1. **Increased Build Times**:
   - More template instantiations to compile
   - 2-3x longer builds compared to minimal type coverage
   - Memory usage during compilation increases
   - CI/CD pipelines take longer

2. **Larger Object Files**:
   - Each semantic type generates multiple instantiations
   - Intermediate build artifacts are larger
   - More disk space required during builds

3. **Complexity**:
   - Template processing system is more complex
   - Debugging build issues requires understanding type equivalence
   - Multiple levels of indirection in the build system

4. **Potential Over-Instantiation**:
   - Some type aliases may never be used in practice
   - Generates code for theoretical rather than actual usage
   - No feedback mechanism to prune unused instantiations

5. **Binary Size Impact**:
   - While filtered by selective rendering, more instantiations still mean larger binaries
   - The size reduction from ADR-0039 is partially offset
   - Mobile/embedded deployments may need custom type profiles

### Technical Details

#### Memory Management During Builds

The system implements chunking to manage memory usage:

```cmake
set(CHUNK_TARGET_INSTANTIATIONS "5" CACHE STRING "Target instantiations per chunk")
set(MULTI_PASS_CHUNK_SIZE "20" CACHE STRING "Direct instantiation file chunk size")

# Auto-detection based on available memory
cmake_host_system_information(RESULT AVAILABLE_MEMORY QUERY AVAILABLE_PHYSICAL_MEMORY)
if(AVAILABLE_MEMORY LESS 4000)
    set(CHUNK_TARGET_INSTANTIATIONS 3)
    message(STATUS "Low memory: Conservative chunking")
endif()
```

#### Diagnostic Output

Comprehensive diagnostics help debug type issues:

```
build/
├── type_combinations/
│   ├── active_types.txt          # Types being compiled
│   ├── combinations_2.txt        # Valid 2-type combinations
│   ├── combinations_3.txt        # Valid 3-type combinations
│   ├── statistics.txt           # Reduction percentages
│   └── rejected_combinations.log # Invalid combinations with reasons
└── cpu_instantiations/
    ├── pairwise_direct_0.cpp    # Chunked instantiation files
    ├── pairwise_direct_1.cpp
    └── ...
```

#### Example Type Expansion

For a single logical type like `int64_t`, the system generates:

```cpp
// Input: handle_pairwise("int64_t", "int64_t", "int64_t", ...)

// Generated instantiations:
template void PairWiseTransform<int64_t, int64_t, int64_t>::exec(...);
template void PairWiseTransform<long long, long long, long long>::exec(...);
template void PairWiseTransform<long, long, long>::exec(...);  // On 64-bit Linux
template void PairWiseTransform<LongType, LongType, LongType>::exec(...);
template void PairWiseTransform<sd::LongType, sd::LongType, sd::LongType>::exec(...);
```


## Conclusion

The comprehensive template instantiation migration successfully addresses the persistent undefined reference errors in libnd4j by ensuring all type aliases are properly instantiated. While this approach increases build times and complexity, it provides a robust solution that eliminates a major source of build failures and improves platform portability.

The integration with the selective rendering system from ADR-0039 ensures that we still benefit from semantic filtering while achieving complete type coverage. The trade-off between build time and reliability has proven worthwhile, as developers spend less time debugging link errors and more time on feature development.

Future optimizations through LTO and usage-based profiling could mitigate the current disadvantages while maintaining the robustness benefits. The system provides a solid foundation for libnd4j's template-heavy architecture across diverse platforms and use cases.