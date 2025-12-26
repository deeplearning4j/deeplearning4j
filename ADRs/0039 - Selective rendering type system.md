# ADR: Selective Rendering Type System for libnd4j

## Status

Implemented

Proposed by: Adam Gibson (August 2025)

Discussed with: Development Team

## Context

The libnd4j project faces significant challenges with binary size and compilation times due to its template-heavy architecture.
The library supports numerous data types (bool, int8, uint8, int16, uint16, int32, uint32, int64, uint64, float16, bfloat16,
float32, double, and string types), and many operations require instantiation for combinations of 2 or 3 types.
This results in:

- **Combinatorial explosion**: With ~20 types, 3-type operations generate 8,000 potential template instantiations
- **Binary bloat**: Compiled libraries exceed 1GB, causing deployment challenges
- **Compilation times**: Full builds take hours, slowing development iteration
- **Unnecessary instantiations**: Many type combinations (e.g., `string × float × double`) are semantically invalid
- **Memory pressure**: Build servers require 32GB+ RAM for parallel compilation
- **Deployment issues**: Mobile and edge devices struggle with large binary sizes

Traditional approaches like explicit template instantiation lists are unmaintainable at this scale. The selective rendering system provides an automated, semantically-aware solution to these challenges.

## Proposal

This ADR documents the implementation of a selective rendering type system that:

1. **Automatically detects valid type combinations** based on semantic rules
2. **Generates optimized template instantiation lists** at CMake configure time
3. **Provides compile-time macros** to conditionally compile only valid combinations
4. **Supports type profiles** for different ML workloads (quantization, training, inference)
5. **Integrates with the build system** transparently

### Core Components

#### 1. SelectiveRenderingCore.cmake
The main orchestration system that:
- Discovers active types from `types.h`
- Applies semantic filtering rules
- Generates valid type combinations
- Creates compile-time configuration headers

#### 2. Semantic Filtering Engine
Implements domain-specific rules:

```cmake
# Example semantic rules
function(_internal_srcore_is_valid_triple type1 type2 type3 output_var)
    # Same type operations are always valid
    if(type1 STREQUAL type2 AND type2 STREQUAL type3)
        set(${output_var} TRUE PARENT_SCOPE)
        return()
    endif()
    
    # Quantization patterns: INT8 accumulation
    if(type1 STREQUAL "INT8" AND type2 STREQUAL "INT8" AND type3 STREQUAL "INT32")
        set(${output_var} TRUE PARENT_SCOPE)
        return()
    endif()
    
    # Mixed precision training: FP16 accumulation to FP32
    if((type1 STREQUAL "HALF" OR type1 STREQUAL "BFLOAT16") AND 
       type2 STREQUAL type1 AND type3 STREQUAL "FLOAT32")
        set(${output_var} TRUE PARENT_SCOPE)
        return()
    endif()
    
    # Block invalid patterns
    # Never output DOUBLE from 8/16-bit inputs
    if(type3 STREQUAL "DOUBLE" AND 
       (type1 MATCHES "INT8|UINT8|INT16|UINT16|HALF|BFLOAT16" OR
        type2 MATCHES "INT8|UINT8|INT16|UINT16|HALF|BFLOAT16"))
        set(${output_var} FALSE PARENT_SCOPE)
        return()
    endif()
    
    # ... additional rules
endfunction()
```

#### 3. Type Profiles (TypeProfiles.cmake)
Pre-configured type sets for common workloads:

- **Quantization**: `int8_t, uint8_t, float, int32_t`
- **Training**: `float16, bfloat16, float, double, int32_t, int64_t`
- **Inference**: `int8_t, uint8_t, float16, float, int32_t`
- **NLP**: `std::string, float, int32_t, int64_t`
- **Standard All**: Optimized from 16+ to 10 essential types

#### 4. Generated Headers
Creates `selective_rendering.h` with compile-time macros:

```cpp
// Single type compilation flags
#define SD_SINGLE_TYPE_5_COMPILED 1      // float32
#define SD_SINGLE_TYPE_7_COMPILED 1      // int8
#define SD_SINGLE_TYPE_50_COMPILED 0     // utf8 (not compiled)

// Pair type compilation flags  
#define SD_PAIR_TYPE_7_5_COMPILED 1      // int8,float32
#define SD_PAIR_TYPE_50_5_COMPILED 0     // utf8,float32 (invalid)

// Triple type compilation flags
#define SD_TRIPLE_TYPE_7_7_9_COMPILED 1  // int8,int8,int32 (accumulation)
#define SD_TRIPLE_TYPE_6_3_3_COMPILED 0  // double,half,half (invalid)

// Conditional compilation macros
#define SD_IF_SD_TRIPLE_TYPE_7_7_9_COMPILED(code) code
#define SD_IF_SD_TRIPLE_TYPE_6_3_3_COMPILED(code) do {} while(0)
```

### Validation Rules

The system implements sophisticated validation based on ML operation semantics:

#### Valid Patterns
1. **Same-type operations**: Always valid (e.g., `float,float,float`)
2. **Quantization**:
   - INT8 accumulation: `int8,int8,int32`
   - Dequantization: `int8,float32,float32`
   - Quantization: `float32,float32,int8`
3. **Mixed Precision**:
   - FP16 accumulation: `half,half,float32`
   - BF16 accumulation: `bfloat16,bfloat16,float32`
4. **Comparisons**: Any matching types to bool (e.g., `float32,float32,bool`)
5. **Indexing**: Integer indices with any data type
6. **Conditionals**: `bool,T,T` for any type T

#### Invalid Patterns
1. **Precision downgrades**: `double,double,float32` (loses precision)
2. **Type size jumps**: `int8,int8,int64` (unnecessarily large)
3. **Mixed string/numeric**: `string,float,double` (nonsensical)
4. **Bool to precision types**: `bool,bool,double` (unnecessary)

### Integration with Build System

```cmake
# In CMakeLists.txt
set(SD_TYPE_PROFILE "training" CACHE STRING "Type profile for selective rendering")
include(cmake/SelectiveRendering.cmake)

# Automatically sets up:
# - UNIFIED_COMBINATIONS_2: Valid 2-type combinations
# - UNIFIED_COMBINATIONS_3: Valid 3-type combinations
# - Generated headers in ${CMAKE_BINARY_DIR}/include/system/

# In source code
#include <system/selective_rendering.h>

template<typename T, typename U, typename V>
class BroadcastOp {
    static void exec(...) {
        SD_IF_SD_TRIPLE_TYPE_COMPILED(T, U, V, {
            // Implementation only compiled for valid type combinations
            // ...
        });
    }
};
```

### Diagnostic Reporting

The system generates detailed diagnostics in `${CMAKE_BINARY_DIR}/type_combinations/`:

- `active_types.txt`: List of types being compiled
- `combinations_2.txt`: All valid 2-type combinations
- `combinations_3.txt`: All valid 3-type combinations
- `statistics.txt`: Reduction percentages and counts
- `rejected_combinations.log`: Invalid combinations with reasons

## Implementation Details from Type Alias Expansion System

### Overview of the Macro System

The header file reveals the sophisticated macro system that enables the selective rendering. This system handles the complex type aliasing relationships in C++ where multiple type names can refer to the same underlying type (e.g., `long`, `long long`, and `int64_t` might all be the same on certain platforms).

### Key Implementation Components

#### 1. Platform Type Detection

The system first detects actual type relationships on the target platform:

```cpp
// Platform detection for type sizes
#define SD_INT32_IS_INT (std::is_same<int32_t, int>::value)
#define SD_INT32_IS_LONG (std::is_same<int32_t, long>::value)

#define SD_INT64_IS_LONG (std::is_same<int64_t, long>::value)
#define SD_INT64_IS_LONG_LONG (std::is_same<int64_t, long long>::value)

#define SD_UINT32_IS_UINT (std::is_same<uint32_t, unsigned int>::value)
#define SD_UINT32_IS_ULONG (std::is_same<uint32_t, unsigned long>::value)
```

This ensures the system generates correct instantiations regardless of platform-specific type definitions.

#### 2. Type Equivalence Expansion Macros

For each type family, the system defines expansion macros that instantiate templates for all equivalent types:

```cpp
// Core expansion macros by equivalence class
#define EXPAND_INT64_VARIANTS(MACRO, ...) \
    MACRO(LongType, __VA_ARGS__)

#define EXPAND_UINT64_VARIANTS(MACRO, ...) \
    MACRO(uint64_t, __VA_ARGS__) \
    MACRO(unsigned long long, __VA_ARGS__) \
    MACRO(unsigned long, __VA_ARGS__)

#define EXPAND_INT32_VARIANTS(MACRO, ...) \
    MACRO(int32_t, __VA_ARGS__) \
    MACRO(int, __VA_ARGS__)

#define EXPAND_INT8_VARIANTS(MACRO, ...) \
    MACRO(int8_t, __VA_ARGS__)

#define EXPAND_UINT8_VARIANTS(MACRO, ...) \
    MACRO(uint8_t, __VA_ARGS__) \
    MACRO(unsigned char, __VA_ARGS__)
```

#### 3. Template Instantiation Macros

The system provides different macro families for different instantiation patterns:

##### Single Type Instantiation (_RANDOMSINGLE)
```cpp
#define _RANDOMSINGLE(TEMPLATE_NAME, SIGNATURE, ENUM, TYPE) \
    EVAL(SD_IF_SINGLE_ALIAS_COMPILED_DECL( \
        ENUM, \
        CONCAT(_EXPAND_RANDOMSINGLE_, TYPE)(TEMPLATE_NAME, SIGNATURE) \
    ))
```

This macro:
- Checks if the single type is enabled via `SD_IF_SINGLE_ALIAS_COMPILED_DECL`
- Expands to all type aliases if enabled
- Generates nothing if the type is disabled

##### Double Type Instantiation (_RANDOMDOUBLE2)
```cpp
#define _RANDOMDOUBLE2(TEMPLATE_NAME, SIGNATURE, ENUM_A, TYPE_A, ENUM_B, TYPE_B) \
    EVAL(SD_IF_PAIR_ALIAS_COMPILED_DECL( \
        ENUM_A, \
        ENUM_B, \
        CONCAT(_EXPAND_RANDOMDOUBLE2_, TYPE_B)(TEMPLATE_NAME, SIGNATURE, TYPE_A) \
    ))
```

Handles all combinations of two types, expanding each to their respective aliases.

##### Triple Type Instantiation (_RANDOMTRIPLE3)
```cpp
#define _RANDOMTRIPLE3(TEMPLATE_NAME, SIGNATURE, ENUM_Z, TYPE_Z, ENUM_Y, TYPE_Y, ENUM_X, TYPE_X) \
    EVAL(SD_IF_TRIPLE_ALIAS_COMPILED_DECL( \
        ENUM_X, \
        ENUM_Y, \
        ENUM_Z, \
        CONCAT(_EXPAND_RANDOMTRIPLE3_X_, TYPE_X)(TEMPLATE_NAME, SIGNATURE, TYPE_Y, TYPE_Z) \
    ))
```

The most complex case, handling three-type combinations with full alias expansion.

#### 4. Type-Specific Expansion Helpers

For each type, specific expansion helpers handle platform variations:

```cpp
// Example for int8_t family
#define _EXPAND_RANDOMSINGLE_int8_t(TEMPLATE_NAME, SIGNATURE) \
    EXPAND_INT8_VARIANTS(INSTANTIATE_TEMPLATE_1, TEMPLATE_NAME, SIGNATURE)

#define _EXPAND_RANDOMSINGLE_SignedChar(TEMPLATE_NAME, SIGNATURE) \
    EXPAND_INT8_VARIANTS(INSTANTIATE_TEMPLATE_1, TEMPLATE_NAME, SIGNATURE)

// Example for floating point (no aliases needed)
#define _EXPAND_RANDOMSINGLE_float(TEMPLATE_NAME, SIGNATURE) \
    template TEMPLATE_NAME<float> SIGNATURE;

// Example for complex aliased types like unsigned long
#define _EXPAND_RANDOMSINGLE_ulong(TEMPLATE_NAME, SIGNATURE) \
    template TEMPLATE_NAME<unsigned long> SIGNATURE;
```

#### 5. Special Cases

##### String Type Support
```cpp
// String types
#define _EXPAND_RANDOMSINGLE_UTF8(TEMPLATE_NAME, SIGNATURE) template TEMPLATE_NAME<utf8string> SIGNATURE;
#define _EXPAND_RANDOMSINGLE_UTF16(TEMPLATE_NAME, SIGNATURE) template TEMPLATE_NAME<utf16string> SIGNATURE;
#define _EXPAND_RANDOMSINGLE_UTF32(TEMPLATE_NAME, SIGNATURE) template TEMPLATE_NAME<utf32string> SIGNATURE;
#define _EXPAND_RANDOMSINGLE_string(TEMPLATE_NAME, SIGNATURE) template TEMPLATE_NAME<stdstring> SIGNATURE;
```

##### Platform-Specific Types
```cpp
// LongType handling for 64-bit integer variations
#define _EXPAND_RANDOMSINGLE_LongType(TEMPLATE_NAME, SIGNATURE) \
    EXPAND_INT64_VARIANTS(INSTANTIATE_TEMPLATE_1, TEMPLATE_NAME, SIGNATURE)

// Special handling for sd::LongType
#define _EXPAND_RANDOMSINGLE_sd_LongType(TEMPLATE_NAME, SIGNATURE) \
    template TEMPLATE_NAME<sd::LongType> SIGNATURE; \
    template TEMPLATE_NAME<int64_t> SIGNATURE; \
    template TEMPLATE_NAME<long long> SIGNATURE; \
    template TEMPLATE_NAME<long> SIGNATURE;
```

### Usage in Practice

The macro system is used throughout libnd4j to conditionally instantiate templates:

```cpp
// In an operation implementation file
#include <system/selective_rendering.h>

// Single type instantiation
BUILD_SINGLE_TEMPLATE(template class ND4J_EXPORT MyOp, , LIBND4J_TYPES);

// This expands to:
// _RANDOMSINGLE(MyOp, , FLOAT32, float)
// _RANDOMSINGLE(MyOp, , INT8, int8_t)
// ... etc for each type in LIBND4J_TYPES

// Double type instantiation
BUILD_DOUBLE_TEMPLATE(template class ND4J_EXPORT BinaryOp, , LIBND4J_TYPES, LIBND4J_TYPES);

// Triple type instantiation
BUILD_TRIPLE_TEMPLATE(template class ND4J_EXPORT TernaryOp, , LIBND4J_TYPES, LIBND4J_TYPES, LIBND4J_TYPES);
```

### Function Call Variants

The system also provides call variants for invoking templated functions rather than instantiating them:

```cpp
// Call variants (no 'template' keyword)
#define _CALL_SINGLE(FUNC_NAME, ARGS, ENUM, TYPE) \
    EVAL(SD_IF_SINGLE_ALIAS_COMPILED_DECL( \
        ENUM, \
        CONCAT(_EXPAND_CALL_SINGLE_, TYPE)(FUNC_NAME, ARGS) \
    ))

#define _CALL_DOUBLE2(FUNC_NAME, ARGS, ENUM_A, TYPE_A, ENUM_B, TYPE_B) \
    EVAL(SD_IF_PAIR_ALIAS_COMPILED_DECL( \
        ENUM_A, \
        ENUM_B, \
        CONCAT(_EXPAND_CALL_DOUBLE2_, TYPE_B)(FUNC_NAME, ARGS, TYPE_A) \
    ))
```

These are used for conditional runtime dispatch based on which type combinations were compiled.

### Integration with Selective Rendering

The macro system integrates with the selective rendering flags generated by CMake:

1. **CMake generates** `selective_rendering.h` with flags like:
   ```cpp
   #define SD_SINGLE_TYPE_7_COMPILED 1      // int8
   #define SD_PAIR_TYPE_7_5_COMPILED 1      // int8,float32
   #define SD_TRIPLE_TYPE_7_7_9_COMPILED 1  // int8,int8,int32
   ```

2. **The macro system checks** these flags via `SD_IF_*_COMPILED_DECL` macros

3. **Templates are instantiated** only for enabled combinations

4. **Type aliases are handled** automatically, ensuring all equivalent types work correctly

## Consequences

### Advantages

1. **Dramatic Binary Size Reduction**: 60-90% smaller binaries depending on profile
   - Quantization profile: ~85% reduction (300 vs 8000 combinations)
   - Training profile: ~70% reduction
   - Inference profile: ~80% reduction

2. **Faster Compilation**: 
   - 4-10x faster builds due to fewer instantiations
   - Reduced memory usage during compilation
   - Better parallelization opportunities

3. **Semantic Correctness**: Invalid type combinations are never compiled
   - Prevents runtime errors from nonsensical operations
   - Enforces type safety at compile time
   - Documents valid operation patterns

4. **Maintainable Type Management**:
   - Centralized semantic rules in CMake
   - Easy to add new types or modify rules
   - Clear diagnostics for debugging

5. **Profile-Based Optimization**:
   - Different builds for different workloads
   - Mobile/edge deployments get minimal binaries
   - Training servers get full precision support

6. **Transparent Integration**:
   - No changes required to existing op implementations
   - Works with existing template code
   - Backward compatible with explicit instantiations

7. **Platform Independence**:
   - Handles type aliasing differences across platforms
   - Ensures consistent behavior regardless of underlying type definitions
   - Automatic detection of platform-specific type relationships

### Disadvantages

1. **Build Complexity**: 
   - Additional CMake configuration step
   - More complex build scripts
   - Requires understanding of the selective rendering system

2. **Type Discovery Overhead**:
   - CMake must parse `types.h` to discover types
   - Additional processing time during configuration
   - Platform-specific parsing considerations

3. **Learning Curve**:
   - Developers must understand type profiles
   - Semantic rules need documentation
   - Debugging template errors requires checking validity

4. **Profile Management**:
   - Must choose appropriate profile for use case
   - Profile changes require full rebuild
   - Risk of missing needed type combinations

5. **Maintenance of Semantic Rules**:
   - Rules must be updated for new operation types
   - Complex validation logic in CMake
   - Testing burden for rule correctness

6. **Macro Complexity**:
   - Deep macro nesting can be hard to debug
   - Platform-specific variations add complexity
   - IDE support for macro expansion may be limited

7. **Increased Build Times for Comprehensive Coverage**:
   - When more type combinations are needed, build times can increase significantly
   - The "all" profile with comprehensive coverage can negate compilation speed benefits
   - Trade-off between coverage and build time must be carefully managed
   - Testing all type combinations requires multiple builds with different profiles

8. **Arbitrary Nature of Type Selection**:
   - Manually defined type profiles are somewhat arbitrary
   - No automatic detection of which types are actually used in production
   - Profile definitions based on assumptions rather than real usage data
   - Risk of including unnecessary types or excluding important ones
   - Lack of feedback mechanism to refine type profiles based on runtime usage

9. **Manual Type Definition Requirements**:
   - All type combinations must be manually specified in profiles
   - No dynamic discovery of required type combinations from code analysis
   - Developers must predict type usage patterns upfront
   - Updates require manual intervention and domain expertise
   - Potential for human error in type specification

### Technical Details

#### Type Discovery Process

```cmake
# 1. Parse types.h to find type definitions
file(READ "${CMAKE_CURRENT_SOURCE_DIR}/include/types/types.h" types_content)
string(REGEX MATCHALL "#define[ \t]+TTYPE_([A-Z0-9_]+)[ \t]*,[ \t]*\\(([^)]+)\\)" 
       type_matches "${types_content}")

# 2. Extract enum values and C++ types
foreach(type_match ${type_matches})
    # Parse TTYPE_INT8 , (DataType::INT8, int8_t)
    # Extracts: INT8, DataType::INT8, int8_t
endforeach()

# 3. Map to normalized names for consistency
```

#### Combination Generation Algorithm

```cmake
# Generate all possible combinations
foreach(i RANGE ${max_index})
    foreach(j RANGE ${max_index})
        foreach(k RANGE ${max_index})
            # Apply semantic filtering
            _internal_srcore_is_valid_triple(
                "${type_i}" "${type_j}" "${type_k}" is_valid)
            
            if(is_valid)
                list(APPEND combinations_3 "${i},${j},${k}")
            else()
                # Log rejection reason for diagnostics
            endif()
        endforeach()
    endforeach()
endforeach()
```

#### Header Generation

The system generates platform-independent headers with:
- Preprocessor flags for each type combination
- Mapping macros from type names to numeric values
- Conditional compilation helpers
- Runtime dispatch infrastructure (optional)

## Discussion

The selective rendering system represents a fundamental shift in how libnd4j manages template instantiation complexity. 
Key design decisions:

1. **CMake-Time vs Compile-Time**: Performing type discovery and filtering at CMake configuration time allows for better 
diagnostics and simpler generated code.

2. **Semantic Rules vs Whitelist**: Rather than maintaining explicit lists of valid combinations, 
semantic rules scale better and are more maintainable.

3. **Profile-Based Approach**: Different ML workloads have different type requirements. 
Profiles allow optimal builds for each use case.

4. **Numeric Type Mapping**: Using numeric type IDs (0-255) allows efficient compile-time lookups and smaller generated code.

5. **Diagnostic Output**: Comprehensive reporting helps developers understand what types are being compiled and why.

6. **Type Aliasing Handling**: The sophisticated macro system ensures correct behavior across platforms with different type definitions.

Future enhancements being considered:

- **Dynamic Type Discovery**: Runtime registration of valid type combinations
- **Profile Composition**: Combining multiple profiles for hybrid workloads
- **Automatic Profile Detection**: Analyzing codebase to suggest optimal profile
- **IDE Integration**: Visual Studio Code extension to show type validity
- **Performance Profiling**: Measuring actual usage of type combinations
- **Macro Simplification**: Potential code generation to reduce macro complexity
- **Usage-Based Type Selection**: Instrumenting production systems to determine actual type usage patterns
- **Automated Profile Generation**: Using static analysis or runtime profiling to generate optimal type profiles

## Conclusion

The selective rendering type system successfully addresses the combinatorial explosion problem in libnd4j's template instantiation. By applying semantic filtering based on ML operation patterns and handling platform-specific type aliasing through a sophisticated macro system, the system achieves 60-90% reductions in binary size and compilation time while maintaining type safety and operational correctness. The profile-based approach ensures different deployment scenarios get optimally sized binaries, from edge inference to full training systems. 

However, the system's reliance on manually defined type profiles and the potential for increased build times with comprehensive coverage represent ongoing challenges. The somewhat arbitrary nature of type selection highlights the need for more data-driven approaches to profile definition. Future work should focus on automated type discovery and usage-based profile generation to reduce the manual maintenance burden and ensure optimal type coverage.

The implementation demonstrates how compile-time metaprogramming, build system integration, and domain-specific knowledge can combine to solve complex software engineering challenges in high-performance computing libraries, while also revealing the trade-offs inherent in such sophisticated systems.