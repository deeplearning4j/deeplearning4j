# ADR: Integration of ppstep Preprocessor Debugger with Recording Capabilities for libnd4j

## Status

Implemented

Proposed by: Adam Gibson (January 2025)

Discussed with: Development Team

## Context

The libnd4j project, being a complex C++ library with extensive use of preprocessor macros for performance optimization, SIMD instructions, and cross-platform compatibility, faces significant challenges in debugging macro expansions. Traditional debugging tools are inadequate for understanding the complex preprocessor transformations that occur during compilation, particularly when dealing with:

- Multi-level nested macros for operator definitions
- Platform-specific conditional compilation (#ifdef chains)
- SIMD intrinsics wrapped in macros
- Code generation macros that expand to hundreds of lines
- Template metaprogramming combined with preprocessor directives

The ppstep tool (https://github.com/agibsonccc/ppstep) provides an interactive debugger for the C/C++ preprocessor, allowing developers to step through macro expansions, set breakpoints, and visualize the transformation process. This ADR documents the integration of ppstep into the libnd4j build system with enhanced recording capabilities for offline analysis.

## Proposal

This ADR proposes and documents the integration of ppstep as an optional build tool for libnd4j, with the following enhancements:

1. **Automated Build Integration**: ppstep is built as an external project through CMake when `BUILD_PPSTEP=ON` is specified
2. **Robust Include Path Discovery**: Multiple methods to automatically discover system include paths at CMake configure time
3. **Recording Capabilities**: New commands to record macro expansion traces to files for offline analysis
4. **Wrapper Script Generation**: Automatic generation of a wrapper script with all necessary include paths and defines

### Recording Feature Implementation

The recording feature adds three new commands to ppstep:

```cpp
// New commands added to the ppstep grammar
| lexeme[(lit("record") | lit("rec")) > +space > anything[PPSTEP_ACTION(start_record(attr))]]
| (lit("stoprecord") | lit("sr"))[PPSTEP_ACTION(stop_record())]
| lit("status")[PPSTEP_ACTION(status())]
```

Recording functionality is implemented in the client class:

```cpp
class client {
    // Recording state
    std::ofstream record_file;
    bool recording_active;
    std::string record_filename;
    
    bool start_recording(const std::string& filename);
    void stop_recording();
    bool is_recording() const;
    
    // Integration into event handlers
    void on_expand_function(...) {
        if (recording_active) {
            record_file << "[CALL] " << call_tokens << " // Args: " << arguments << std::endl;
        }
    }
    
    void on_expanded(...) {
        if (recording_active) {
            record_file << "[EXPANDED] " << initial << " => " << result << std::endl;
        }
    }
    
    void on_rescanned(...) {
        if (recording_active) {
            record_file << "[RESCANNED] " << initial << " => " << result 
                       << " // Caused by: " << cause << std::endl;
        }
    }
};
```

### Include Path Discovery Strategy

The CMake integration (`Ppstep.cmake`) implements a comprehensive 7-method approach to discover system include paths:

```cmake
# Method 1: CMAKE_CXX_IMPLICIT_INCLUDE_DIRECTORIES
foreach(dir ${CMAKE_CXX_IMPLICIT_INCLUDE_DIRECTORIES})
    list(APPEND DISCOVERED_INCLUDE_PATHS "${dir}")
endforeach()

# Method 2: Compiler verbose output parsing
execute_process(
    COMMAND ${CMAKE_CXX_COMPILER} -E -x c++ -Wp,-v -
    ERROR_VARIABLE COMPILER_ERROR
)
# Parse output for include paths...

# Method 3: GCC-specific commands
execute_process(
    COMMAND ${CMAKE_CXX_COMPILER} -print-file-name=include
    OUTPUT_VARIABLE GCC_INCLUDE_DIR
)

# Method 4: Target-specific paths
execute_process(
    COMMAND ${CMAKE_CXX_COMPILER} -dumpmachine
    OUTPUT_VARIABLE GCC_TARGET
)
# Construct paths like /usr/lib/gcc/${GCC_TARGET}/${GCC_VERSION}/include

# Method 5: Common system paths
set(COMMON_INCLUDE_PATHS
    /usr/include
    /usr/local/include
    /usr/include/linux
    /usr/include/x86_64-linux-gnu
    # ... more paths
)

# Method 6: Test compilation with -H flag
file(WRITE test_includes.cpp "#include <limits.h>...")
execute_process(
    COMMAND ${CMAKE_CXX_COMPILER} -H -E test_includes.cpp
    ERROR_VARIABLE HEADER_ERROR
)
# Parse -H output for actual include locations

# Method 7: Clang resource directory (if available)
execute_process(
    COMMAND clang++ -print-resource-dir
    OUTPUT_VARIABLE CLANG_RESOURCE_DIR
)
```

### Wrapper Script Generation

The build process generates a wrapper script (`ppstep-nd4j`) that:
- Includes all discovered system paths
- Adds all libnd4j-specific include directories
- Sets appropriate preprocessor defines
- Provides debugging options

```bash
#!/bin/bash
# Auto-generated ppstep wrapper for libnd4j

PPSTEP_BIN="${PPSTEP_BUILD_DIR}/ppstep"

# System includes discovered at CMake configure time
SYSTEM_INCLUDES="-I/usr/include -I/usr/lib/gcc/x86_64-linux-gnu/11/include ..."

# Include paths for libnd4j
INCLUDES="-I${CMAKE_SOURCE_DIR}/include"
INCLUDES="$INCLUDES -I${CMAKE_SOURCE_DIR}/include/array"
INCLUDES="$INCLUDES -I${CMAKE_SOURCE_DIR}/include/ops"
# ... more includes

# Basic defines
DEFINES="-D__CPUBLAS__=1 -DSD_CPU=1"

exec "$PPSTEP_BIN" $SYSTEM_INCLUDES $INCLUDES $DEFINES "$@"
```

## Usage Examples

### Basic Interactive Debugging
```bash
# Build ppstep
cmake -DBUILD_PPSTEP=ON ..
make ppstep_build

# Debug a source file
./ppstep-nd4j include/ops/declarable/generic/parity_ops.cpp

# In ppstep prompt
pp> break call DECLARE_OP
pp> continue
pp (called)> step
```

### Recording Macro Expansions
```bash
# Start ppstep on a complex macro file
./ppstep-nd4j include/ops/declarable/headers/parity.h

# Record all expansions to a file
pp> record macro_trace.txt
Recording to macro_trace.txt

# Step through many expansions
pp> step 1000

# Or set a breakpoint and continue
pp> break call DECLARE_CUSTOM_OP
pp> continue
pp (called)> step 500

# Stop recording
pp> stoprecord
Recording stopped
```

### Analyzing Specific Macros
```bash
# Use with the expand command
pp> record custom_op_expansion.txt
pp> expand DECLARE_CUSTOM_OP(conv2d, 1, 1, false, 0, -2)
pp [DECLARE_CUSTOM_OP(...)]> step 100
pp [DECLARE_CUSTOM_OP(...)]> quit
pp> stoprecord
```

### Sample Recording Output
```
=== PPSTEP TRACE ===
Started: Fri Jan 10 14:30:45 2025
===================

[CALL] DECLARE_CUSTOM_OP(conv2d, 1, 1, false, 0, -2) // Args: conv2d, 1, 1, false, 0, -2
[EXPANDED] DECLARE_CUSTOM_OP(conv2d, 1, 1, false, 0, -2) => class ND4J_EXPORT conv2d : public DeclarableCustomOp<1, 1, false, 0, -2> { ... }
[CALL] DECLARE_SHAPE_FN(conv2d) // Args: conv2d
[EXPANDED] DECLARE_SHAPE_FN(conv2d) => DECLARE_SHAPE_FN_(conv2d)
[CALL] DECLARE_SHAPE_FN_(conv2d) // Args: conv2d
[EXPANDED] DECLARE_SHAPE_FN_(conv2d) => ShapeList* conv2d::calculateOutputShape(ShapeList* inputShape, sd::graph::Context& block)
[RESCANNED] ShapeList* conv2d::calculateOutputShape(...) => ShapeList* conv2d::calculateOutputShape(...) // Caused by: DECLARE_SHAPE_FN_(conv2d)

=== END OF TRACE ===
```

## Consequences

### Advantages

1. **Deep Macro Visibility**: Provides unprecedented visibility into complex macro expansions that are otherwise opaque
2. **Recording for Documentation**: Can generate traces that serve as documentation for how macros work
3. **Debugging Complex Issues**: Helps debug macro-related compilation errors and unexpected behavior
4. **Offline Analysis**: Recording feature allows team members to share and analyze macro expansion traces
5. **Automated Path Discovery**: Robust include path discovery works across different compiler configurations
6. **Non-Invasive**: Optional build flag means it doesn't affect normal builds

### Disadvantages

1. **Build Complexity**: Adds another external dependency and build step
2. **Boost Dependency**: Requires Boost libraries to be installed (Wave, System, Filesystem, etc.)
3. **Include Path Discovery Overhead**: The comprehensive discovery process adds time to CMake configuration
4. **Platform Limitations**: Some discovery methods are GCC/Clang specific
5. **Learning Curve**: Developers need to learn ppstep commands and workflow
6. **Recording File Size**: Large macro expansions can generate substantial trace files

### Trade-offs of Include Path Discovery

The multi-method approach to discovering include paths represents a trade-off between:

**Robustness** (✓):
- Works across different compiler versions
- Handles non-standard installations
- Finds compiler-specific includes (GCC internal headers)
- Validates critical headers are found

**Configuration Time** (✗):
- Each discovery method adds overhead
- Multiple execute_process calls during CMake configuration
- File system searches can be slow on some systems

**Maintenance** (✗):
- Complex CMake code to maintain
- May need updates for new compiler versions
- Platform-specific code paths

The decision to use comprehensive discovery was made because:
1. libnd4j is built on diverse systems (various Linux distros, macOS, different compiler versions)
2. Missing system headers cause cryptic ppstep errors
3. One-time configuration cost is acceptable for debugging capability
4. Generated wrapper script caches the discovered paths

## Technical Implementation Details

### Integration Points

1. **CMake Integration**: `libnd4j/cmake/Ppstep.cmake` handles the entire build and configuration
2. **External Project**: ppstep is built as a CMake ExternalProject from the forked repository
3. **Recording Changes**: Added to `client.hpp` and `view.hpp` in the ppstep source
4. **Grammar Extensions**: Modified Boost.Spirit grammar to parse new commands

### File Modifications

```
ppstep/
├── src/
│   ├── client.hpp      # Added recording state and methods
│   └── view.hpp        # Added command parsing for record/stoprecord/status
libnd4j/
├── cmake/
│   └── Ppstep.cmake    # Complete build and discovery logic
└── CMakeLists.txt      # Added include(cmake/Ppstep.cmake)
```

### Build Flags

- `BUILD_PPSTEP=ON`: Enables ppstep build (default: OFF)
- Early return from CMakeLists.txt prevents full libnd4j build when only building ppstep

## Future Enhancements

1. **JSON Output Format**: Export traces in structured JSON for programmatic analysis
2. **Macro Dependency Graph**: Generate visual graphs of macro dependencies
3. **Integration with IDEs**: VSCode/CLion extensions to visualize traces
4. **Conditional Compilation Analysis**: Track which #ifdef branches are taken
5. **Performance Profiling**: Measure preprocessing time per macro
6. **Automated Testing**: Use recordings to create regression tests for macro behavior

## Conclusion

The integration of ppstep with recording capabilities provides libnd4j developers with powerful tools for understanding and debugging the complex macro systems used throughout the codebase. While the include path discovery adds complexity to the build configuration, it ensures the tool works reliably across diverse development environments. The recording feature transforms ppstep from an interactive debugging tool into a comprehensive macro analysis system that can generate documentation and support offline debugging workflows.