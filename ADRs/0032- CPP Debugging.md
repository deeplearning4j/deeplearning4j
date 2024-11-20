# C++ Print Debugging Utilities

## Status

Implemented

Proposed by: Adam Gibson (20-11-2024)
Discussed with: Paul Dubs

## Context

We need a way to provide debugging utilities for C++ code in the nd4j library. These utilities help in identifying issues such as out-of-bounds crashes and unexpected behavior in mathematical operations. The goal is to have a set of tools that can be easily enabled or disabled via configuration flags.

## Decision

We implement three distinct debugging utilities that can be controlled through configuration flags:

1. Print Indices - For tracking loop execution and array access
2. Print Math - For debugging mathematical operations
3. Preprocessor Output - For debugging macro behavior

Each utility serves a specific debugging purpose and can be independently enabled or disabled through both Maven and CMake configuration.

## Configuration

### Maven Configuration (pom.xml)
```xml
<!-- Print Indices Configuration -->
<libnd4j.printindices>OFF</libnd4j.printindices>

<!-- Print Math Configuration -->
<libnd4j.printmath>OFF</libnd4j.printmath>

<!-- Preprocessor Configuration -->
<libnd4j.preprocess>OFF</libnd4j.preprocess>
```

### Build System Integration
The utilities are integrated into the build system through a shell script that configures CMake. The script:

1. Echoes configuration status:
```bash
echo PRINT_INDICES       = "$PRINT_INDICES"
echo PRINT_MATH          = "$PRINT_MATH"
echo PREPROCESS          = "$PREPROCESS"
```

2. Passes configuration to CMake:
```bash
eval "$CMAKE_COMMAND" \
    -DPRINT_MATH="$PRINT_MATH" \
    -DPRINT_INDICES="$PRINT_INDICES" \
    # ... other CMake configurations
```

3. Special handling for preprocessing:
```bash
if [ "$PREPROCESS" == "ON" ]; then
    # Re-run CMake with preprocessing enabled
    eval "$CMAKE_COMMAND" \
        -DPRINT_MATH="$PRINT_MATH" \
        -DPRINT_INDICES="$PRINT_INDICES" \
        -DSD_PREPROCESS="$PREPROCESS" \
        # ... other configurations
    echo "Running preprocessing step..."
    exit 0
fi
```

## Implementation Details

### 1. Print Indices Utility

#### Purpose
Tracks loop iterations and array access patterns to help identify out-of-bounds issues and iteration-related bugs.

#### Build Configuration
1. Set through Maven property `libnd4j.printindices`
2. Passed to CMake as `PRINT_INDICES`
3. Defines `PRINT_INDICES` macro in C++ code

#### Implementation
```cpp
#if defined(PRINT_INDICES)
    printf("i: %lld xEws %lld ReduceBoolFunction<X, Z>::execScalar\n", i, xEws);
#endif
```

#### Example Usage
```cpp
else {
    for (auto i = start; i < stop; i++) {
#if defined(PRINT_INDICES)
        printf("i: %lld xEws %lld ReduceBoolFunction<X, Z>::execScalar\n", i, xEws);
#endif
        intermediate[thread_id] = OpType::update(
            intermediate[thread_id], 
            OpType::op(x[i * xEws], extraParams), 
            extraParams
        );
    }
}
```

### 2. Print Math Utility

#### Purpose
Provides detailed tracking of mathematical operations, including input values, outputs, and execution flow.

#### Build Configuration
1. Set through Maven property `libnd4j.printmath`
2. Passed to CMake as `PRINT_MATH`
3. Defines `SD_GCC_FUNCTRACE` macro in C++ code when enabled

#### Implementation
```cpp
template <>
SD_INLINE SD_HOST void sd_print_math2<uint16_t>(char* func_name, uint16_t input1, uint16_t input2, uint16_t output) {
#if defined(SD_GCC_FUNCTRACE)
    PRINT_IF_NECESSARY(func_name);
#endif
    printf("%s: input1 = %d, input2 = %d, output = %d\n", func_name, input1, input2, output);
    fflush(stdout);
}
```

### 3. Preprocessor Output Utility

#### Purpose
Generates preprocessed source files to help debug macro-related issues and understand macro expansion.

#### Build Configuration
1. Set through Maven property `libnd4j.preprocess`
2. Passed to CMake as `SD_PREPROCESS`
3. Triggers special build flow in shell script when enabled

#### Build Process
When preprocessing is enabled:
1. Maven sets `libnd4j.preprocess=ON`
2. Shell script detects `PREPROCESS=ON`
3. CMake is run with `-DSD_PREPROCESS=ON`
4. Build exits after preprocessing step
5. Preprocessed files are generated in the build directory

#### Implementation
```cmake
if("${SD_PREPROCESS}" STREQUAL "ON")
    message("Preprocessing enabled ${CMAKE_BINARY_DIR}")
    include_directories(${CMAKE_BINARY_DIR}/../../include)
    list(REMOVE_DUPLICATES ALL_SOURCES)
    # Generate preprocessed files
```

## Consequences

### Print Indices
#### Advantages
- Easily identifies loop boundary issues
- Helps debug array access patterns
- Minimal code overhead

#### Drawbacks
- Can generate large amounts of output
- Performance impact when enabled

### Print Math
#### Advantages
- Detailed visibility into mathematical operations
- Type-safe debugging
- Stack trace integration

#### Drawbacks
- Increased binary size
- Runtime overhead when enabled
- Additional complexity in template code

### Preprocessor Output
#### Advantages
- Clear visibility into macro expansion
- Helps debug complex macro interactions
- Useful for template debugging
- Early exit prevents unnecessary compilation steps

#### Drawbacks
- Increases build complexity
- Additional disk space requirements
- Requires separate build run for preprocessing
- Cannot be combined with normal build flow

## References
- platformmath.h
- templatemath.h
- CMakeLists.txt configuration
- pom.xml configuration
- build-dl4j-natives.sh script
- ReduceBoolFunction implementation