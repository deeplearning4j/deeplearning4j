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

The ppstep tool (https://github.com/agibsonccc/ppstep) provides an interactive debugger for the C/C++ preprocessor, allowing developers to step through macro expansions, set breakpoints, and visualize the transformation process. This ADR documents the integration of ppstep into the libnd4j build system with enhanced recording capabilities and error handling features for offline analysis.

## Proposal

This ADR proposes and documents the integration of ppstep as an optional build tool for libnd4j, with the following enhancements:

1. **Automated Build Integration**: ppstep is built as an external project through CMake when `BUILD_PPSTEP=ON` is specified
2. **Maven Build System Integration**: Seamless integration with the Maven build via `libnd4j.build.ppstep` property
3. **Robust Include Path Discovery**: Multiple methods to automatically discover system include paths at CMake configure time
4. **Recording Capabilities**: New commands to record macro expansion traces to files for offline analysis
5. **Error Handling and Break on Error**: Ability to pause execution when preprocessing errors occur, with full error context capture
6. **Wrapper Script Generation**: Automatic generation of a wrapper script with all necessary include paths and defines

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
            record_file << "[CALL] ";
            output_tokens_normalized(record_file, preserved_call_tokens);
            record_file << std::endl;
        }
    }
    
    void on_expanded(...) {
        if (recording_active) {
            record_file << "[EXPANDED]" << std::endl;
            record_file << "  FROM: ";
            output_tokens_normalized(record_file, preserved_initial);
            record_file << std::endl;
            record_file << "  TO:   ";
            output_tokens_normalized(record_file, preserved_result);
            record_file << std::endl;
        }
    }
    
    void on_rescanned(...) {
        if (recording_active) {
            record_file << "[RESCANNED]" << std::endl;
            record_file << "  FROM:      ";
            output_tokens_normalized(record_file, preserved_initial);
            record_file << std::endl;
            record_file << "  TO:        ";
            output_tokens_normalized(record_file, preserved_result);
            record_file << std::endl;
            record_file << "  CAUSED BY: ";
            output_tokens_normalized(record_file, preserved_cause);
            record_file << std::endl;
        }
    }
};
```

### Error Handling and Break on Error Feature

ppstep includes comprehensive error handling capabilities that integrate with both interactive debugging and recording features:

```cpp
class client {
    // Error handling state
    bool break_on_error;
    bool error_occurred;
    std::string last_error_message;
    std::string last_error_file;
    int last_error_line;
    
    void on_error(const std::string& error_msg, const std::string& file, int line) {
        error_occurred = true;
        last_error_message = error_msg;
        last_error_file = file;
        last_error_line = line;
        
        // Record error if recording is active
        if (recording_active) {
            record_file << "[PPSTEP-ERROR] " << file << ":" << line 
                       << " - " << error_msg << std::endl;
        }
        
        // Print error info
        std::cerr << "\n" << ansi::bold << "❌ ERROR: " << ansi::reset 
                  << error_msg << "\n";
        std::cerr << "  at " << file << ":" << line << "\n";
        
        if (break_on_error) {
            std::cerr << "\n⚠️  Stopped on error (break on error enabled)\n";
            std::cerr << "Commands: 'continue' to proceed, 'quit' to exit\n" << std::endl;
        }
    }

    void set_break_on_error(bool enable);
    bool should_break_on_error() const;
    void clear_error();
};
```

The break on error feature is controlled via breakpoint commands:

```cpp
// Grammar additions for error breakpoints in view.hpp
| lexeme[
    (lit("break") | lit("b")) >> *space > (
          // ... other breakpoint types ...
          | lit("error")[PPSTEP_ACTION(cl.set_break_on_error(true))]
  )]
| lexeme[
    (lit("delete") | lit("d")) >> *space > (
          // ... other breakpoint deletions ...
          | lit("error")[PPSTEP_ACTION(cl.set_break_on_error(false))]
  )]
```

When an error occurs with break on error enabled:
1. Preprocessing pauses immediately
2. Error details are displayed with file location and line number
3. Error is recorded to the trace file if recording is active
4. User can continue or quit from the error state
5. Error context is preserved for analysis

### Maven Build Integration

The ppstep tool is fully integrated with the Maven build system through the `libnd4j/pom.xml` configuration. This replaces the need to manually run the `ppstep` script.

#### Maven Property Configuration

The `libnd4j/pom.xml` defines the following property:

```xml
<properties>
    <!-- Other properties... -->
    <libnd4j.build.ppstep>OFF</libnd4j.build.ppstep>
    <!-- Other properties... -->
</properties>
```

#### Build Profile Integration

Both CPU and CUDA build profiles in `pom.xml` pass the ppstep flag to `buildnativeoperations.sh`:

```xml
<!-- CPU Build Profile -->
<buildCommand>
    <program>${libnd4j.buildprogram}</program>
    <argument>buildnativeoperations.sh</argument>
    <!-- ...other arguments... -->
    <argument>--ppstep</argument>
    <argument>${libnd4j.build.ppstep}</argument>
    <!-- ...other arguments... -->
</buildCommand>
```

The same configuration exists in the CUDA build profile, ensuring consistent ppstep support across all build targets.

#### Shell Script Processing

The `buildnativeoperations.sh` script accepts the ppstep flag:

```bash
--ppstep|--build-ppstep)
    BUILD_PPSTEP="$value"
    shift # past argument
    ;;
```

This value is then propagated to the CMake configuration during the build process, which triggers the ppstep external project build when set to `ON`.

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

## How to Use ppstep with Maven (Recommended)

### Building ppstep via Maven

Instead of manually running CMake and build scripts, you can build ppstep directly through Maven:

```bash
# Navigate to the libnd4j directory
cd deeplearning4j/libnd4j

# Build ppstep along with libnd4j (CPU)
mvn clean install -Dlibnd4j.build.ppstep=ON

# Or for CUDA builds
mvn clean install -Dlibnd4j.build.ppstep=ON -Dlibnd4j.cuda=12.3
```

This command will:
1. Configure CMake with `BUILD_PPSTEP=ON`
2. Build the ppstep external project
3. Generate the `ppstep-nd4j` wrapper script
4. Continue with the normal libnd4j build

### Using the Generated Wrapper

After building with `-Dlibnd4j.build.ppstep=ON`, the wrapper script will be available:

```bash
# Find the wrapper script (location depends on your build configuration)
cd blasbuild/cpu  # or blasbuild/cuda for CUDA builds

# Use the wrapper for interactive debugging
./ppstep-nd4j include/ops/declarable/generic/parity_ops.cpp
```

### Integration with Development Workflow

The Maven integration allows ppstep to be part of your normal development workflow:

```bash
# Initial setup - build ppstep once
mvn clean install -Dlibnd4j.build.ppstep=ON -DskipTests

# The wrapper is now available for use
cd blasbuild/cpu
./ppstep-nd4j <source-file>

# Future builds without ppstep (faster)
mvn clean install -DskipTests
```

### Advantages Over Manual Script Execution

Using Maven integration instead of manually running ppstep scripts provides:

1. **Consistent Environment**: Maven ensures all dependencies and paths are correctly configured
2. **Automated Path Discovery**: System include paths are discovered automatically during CMake configuration
3. **Platform Independence**: Works across Linux, macOS, and Windows without manual configuration
4. **Build System Integration**: ppstep build state is tracked by Maven's build lifecycle
5. **Reproducible Builds**: Team members get identical ppstep configurations

## Usage Examples

### Basic Interactive Debugging
```bash
# Build with Maven integration
cd deeplearning4j/libnd4j
mvn clean install -Dlibnd4j.build.ppstep=ON

# Debug a source file
cd blasbuild/cpu
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

### Debugging with Break on Error
```bash
# Start ppstep
./ppstep-nd4j include/problematic_file.cpp

# Enable break on error
pp> break error
Break on error: enabled

# Continue preprocessing - will stop automatically on any error
pp> continue

# When error occurs:
# ❌ ERROR: undefined macro 'MISSING_MACRO'
#   at include/problematic_file.cpp:45
# 
# ⚠️  Stopped on error (break on error enabled)
# Commands: 'continue' to proceed, 'quit' to exit

pp (error)> what
# Shows context around the error

pp (error)> backtrace
# Shows macro expansion stack at error point

# Can continue or quit
pp (error)> continue
```

### Recording with Error Capture
```bash
# Record a problematic preprocessing session
./ppstep-nd4j problematic_macros.cpp

pp> break error
pp> record error_trace.txt
Recording to error_trace.txt

pp> continue
# Processing continues until error...
# ❌ ERROR: macro redefinition of 'BUFFER_SIZE'
#   at problematic_macros.cpp:120

pp (error)> stoprecord
Recording stopped

# The error_trace.txt file now contains:
# [CALL] BUFFER_SIZE
# [EXPANDED] BUFFER_SIZE => 1024
# [PPSTEP-ERROR] problematic_macros.cpp:120 - macro redefinition of 'BUFFER_SIZE'
```

### Combined Recording and Breakpoints with Error Handling
```bash
# Complex debugging session with multiple features
./ppstep-nd4j include/ops/complex_op.cpp

pp> break call DECLARE_CUSTOM_OP
pp> break error
pp> record full_debug_session.txt
Recording to full_debug_session.txt

pp> continue
# Stops at first DECLARE_CUSTOM_OP call
pp (called)> step 50

pp> continue
# Continues until next breakpoint or error

# If error occurs, it's captured in the recording
pp (error)> backtrace
pp (error)> forwardtrace
pp (error)> continue

# When done
pp> stoprecord
pp> quit
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

[CALL] DECLARE_CUSTOM_OP(conv2d, 1, 1, false, 0, -2)
  ARG[0]: conv2d
  ARG[1]: 1
  ARG[2]: 1
  ARG[3]: false
  ARG[4]: 0
  ARG[5]: -2
[EXPANDED]
  FROM: DECLARE_CUSTOM_OP(conv2d, 1, 1, false, 0, -2)
  TO:   class ND4J_EXPORT conv2d : public DeclarableCustomOp<1, 1, false, 0, -2> { ... }
[CALL] DECLARE_SHAPE_FN(conv2d)
  ARG[0]: conv2d
[EXPANDED]
  FROM: DECLARE_SHAPE_FN(conv2d)
  TO:   DECLARE_SHAPE_FN_(conv2d)
[CALL] DECLARE_SHAPE_FN_(conv2d)
  ARG[0]: conv2d
[EXPANDED]
  FROM: DECLARE_SHAPE_FN_(conv2d)
  TO:   ShapeList* conv2d::calculateOutputShape(ShapeList* inputShape, sd::graph::Context& block)
[RESCANNED]
  FROM:      ShapeList* conv2d::calculateOutputShape(...)
  TO:        ShapeList* conv2d::calculateOutputShape(...)
  CAUSED BY: DECLARE_SHAPE_FN_(conv2d)
[PPSTEP-ERROR] include/ops/declarable/headers/conv2d.h:89 - undefined macro 'DEPRECATED_API'
[CALL] FALLBACK_MACRO()
[EXPANDED]
  FROM: FALLBACK_MACRO()
  TO:   /* fallback implementation */

=== END OF TRACE ===
```

## Command Reference

### Recording Commands

| Command | Shortcut | Description | Example |
|---------|----------|-------------|---------|
| `record <file>` | `rec <file>` | Start recording to file | `record trace.txt` |
| `stoprecord` | `sr` | Stop recording | `stoprecord` |
| `status` | - | Show recording status | `status` |

### Error Handling Commands

| Command | Description | Example |
|---------|-------------|---------|
| `break error` | Enable break on error | `break error` |
| `delete error` | Disable break on error | `delete error` |

### Other Useful Commands

| Command | Shortcut | Description |
|---------|----------|-------------|
| `step [n]` | `s [n]` | Step forward n times (default 1) |
| `continue` | `c` | Continue until breakpoint |
| `break call <macro>` | `bc <macro>` | Break when macro is called |
| `break expand <macro>` | `be <macro>` | Break when macro expands |
| `delete call <macro>` | `dc <macro>` | Remove call breakpoint |
| `delete expand <macro>` | `de <macro>` | Remove expand breakpoint |
| `backtrace` | `bt` | Show expansion stack |
| `forwardtrace` | `ft` | Show rescan queue |
| `what` | `?` | Explain current state |
| `quit` | `q` | Exit ppstep |

## Consequences

### Advantages

1. **Deep Macro Visibility**: Provides unprecedented visibility into complex macro expansions that are otherwise opaque
2. **Recording for Documentation**: Can generate traces that serve as documentation for how macros work
3. **Debugging Complex Issues**: Helps debug macro-related compilation errors and unexpected behavior
4. **Error Context Capture**: Break on error feature provides immediate context when preprocessing errors occur, eliminating guesswork
5. **Offline Analysis**: Recording feature allows team members to share and analyze macro expansion traces
6. **Error Trace Files**: Errors captured in recordings provide reproducible bug reports with full preprocessing context
7. **Automated Path Discovery**: Robust include path discovery works across different compiler configurations
8. **Non-Invasive**: Optional build flag means it doesn't affect normal builds
9. **Maven Integration**: Seamless integration with existing build workflows eliminates manual script execution
10. **Reproducible Environments**: Maven ensures consistent ppstep configurations across development teams
11. **Interactive Error Recovery**: Ability to continue after errors or explore error state before quitting

### Disadvantages

1. **Build Complexity**: Adds another external dependency and build step
2. **Boost Dependency**: Requires Boost libraries to be installed (Wave, System, Filesystem, etc.)
3. **Include Path Discovery Overhead**: The comprehensive discovery process adds time to CMake configuration
4. **Platform Limitations**: Some discovery methods are GCC/Clang specific
5. **Learning Curve**: Developers need to learn ppstep commands and workflow, including error handling features
6. **Recording File Size**: Large macro expansions can generate substantial trace files
7. **Maven Build Time**: Enabling ppstep increases initial Maven build time (one-time cost)
8. **Error State Complexity**: Understanding error context in deeply nested macro expansions may still be challenging

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

### Trade-offs of Error Handling

**Benefits**:
- Immediate feedback when errors occur
- Full context available at error point
- Errors captured in recordings for later analysis
- Can continue past errors to find multiple issues in one session

**Costs**:
- Additional state management in client
- Slightly more complex command grammar
- Need to understand error states in addition to normal preprocessing states

## Technical Implementation Details

### Integration Points

1. **Maven Integration**: `libnd4j/pom.xml` property `libnd4j.build.ppstep` controls ppstep build
2. **Shell Script**: `libnd4j/buildnativeoperations.sh` accepts `--ppstep` flag and propagates to CMake
3. **CMake Integration**: `libnd4j/cmake/Ppstep.cmake` handles the entire build and configuration
4. **External Project**: ppstep is built as a CMake ExternalProject from the forked repository
5. **Recording Changes**: Added to `client.hpp` and `view.hpp` in the ppstep source
6. **Error Handling**: Implemented in `client.hpp` with integration in event handlers and prompt handling
7. **Grammar Extensions**: Modified Boost.Spirit grammar to parse new commands including error breakpoints

### File Modifications and Configuration

```
deeplearning4j/
└── libnd4j/
    ├── pom.xml                          # Maven property: libnd4j.build.ppstep
    ├── buildnativeoperations.sh         # Shell script flag: --ppstep
    ├── cmake/
    │   └── Ppstep.cmake                 # Complete build and discovery logic
    └── CMakeLists.txt                   # Includes cmake/Ppstep.cmake

ppstep/ (external project)
├── src/
│   ├── client.hpp                       # Added recording and error handling state and methods
│   └── view.hpp                         # Added command parsing for record/stoprecord/status/error
```

### Build Flags and Properties

**Maven Properties:**
- `libnd4j.build.ppstep`: Controls ppstep build (default: OFF, set to ON to enable)

**Shell Script Flags:**
- `--ppstep ON/OFF`: Enables/disables ppstep build
- `--build-ppstep ON/OFF`: Alternative flag name (same functionality)

**CMake Variables:**
- `BUILD_PPSTEP`: Set by shell script, triggers ExternalProject build
- Early return from CMakeLists.txt prevents full libnd4j build when only building ppstep

## Future Enhancements

1. **JSON Output Format**: Export traces in structured JSON for programmatic analysis
2. **Macro Dependency Graph**: Generate visual graphs of macro dependencies
3. **Integration with IDEs**: VSCode/CLion extensions to visualize traces
4. **Conditional Compilation Analysis**: Track which #ifdef branches are taken
5. **Performance Profiling**: Measure preprocessing time per macro
6. **Automated Testing**: Use recordings to create regression tests for macro behavior
7. **Maven Plugin**: Create a dedicated Maven plugin for easier ppstep integration
8. **Incremental Builds**: Cache ppstep builds to avoid rebuilding on every Maven invocation
9. **Error Pattern Analysis**: Automatically detect common error patterns in recordings
10. **Reverse Stepping**: Rewind preprocessing to view earlier states (planned in ppstep TODO)
11. **Conditional Compilation Visualization**: Explore #if/#elif/#else branches (planned in ppstep TODO)

## Conclusion

The integration of ppstep with recording capabilities and error handling provides libnd4j developers with powerful tools for understanding and debugging the complex macro systems used throughout the codebase. The Maven build system integration eliminates the need for manual script execution and ensures consistent, reproducible ppstep environments across development teams.

The **break on error** feature is particularly valuable for libnd4j development because:
- Complex macro systems often have cascading errors that are hard to diagnose
- Error context at the point of failure is captured automatically
- Developers can explore the preprocessing state when errors occur
- Error traces in recordings provide reproducible bug reports
- Interactive recovery allows investigation of multiple issues in one session

While the include path discovery adds complexity to the build configuration, it ensures the tool works reliably across diverse development environments. The recording feature transforms ppstep from an interactive debugging tool into a comprehensive macro analysis system that can generate documentation and support offline debugging workflows.

The Maven integration represents a significant improvement over manual ppstep invocation by providing:
- Automated dependency management
- Consistent build environments
- Platform-independent configuration
- Integration with existing development workflows

Developers should enable ppstep with `-Dlibnd4j.build.ppstep=ON` during their initial project setup, after which the generated wrapper script remains available for ongoing macro debugging needs. The error handling features should be used whenever investigating preprocessing failures or unexpected macro behavior.

---
