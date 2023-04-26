# ADR: Implementing -finstrument-functions for Tracing in nd4j and libnd4j Code Base

## Status

Implemented

Proposed by: Adam Gibson (26-04-2023)

Discussed with: Paul Dubs

## Context

The nd4j and libnd4j code base can be difficult to debug due to the lack of tools for tracing program execution. The proposed implementation aims to use the `-finstrument-functions` flag provided by GCC to trace function calls and exits in the code base. This will help developers better understand the flow of the program and identify potential issues.

## Decision

The implementation process involves the following steps:

1. Add the correct compiler flags to the javacpp plugin.
2. Add the correct flags to the cmake build.
3. Add a maven profile to each nd4j backend containing the correct compiler flags to work with `-finstrument-functions`.
4. Add the following compiler flags for the libnd4j code base:

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Bsymbolic -rdynamic -fno-omit-frame-pointer -fno-optimize-sibling-calls -rdynamic -finstrument-functions -g -O0")

sql
Copy code

5. Implement a method in each backend to set the output file for tracing results.

## Implementation Details

### Compiler Flags

- `-Bsymbolic`: Bind references to global symbols at link time, reducing the runtime overhead.
- `-rdynamic`: Export all dynamic symbols to the dynamic symbol table, making them available for backtracing.
- `-fno-omit-frame-pointer`: Do not omit the frame pointer, allowing for accurate backtraces.
- `-fno-optimize-sibling-calls`: Disable sibling call optimization to maintain the correct call stack.
- `-finstrument-functions`: Enable instrumentation of function entry and exit points.
- `-g`: Enable debugging information.
- `-O0`: Set the optimization level to zero for easier debugging.

### Setting the Output File

A method is implemented in each backend to set the output file for tracing results. For example, in the Nd4jCpu class:

```java
Nd4jCpu nd4jCpu = (Nd4jCpu) NativeOpsHolder.getInstance().getDeviceNativeOps();
nd4jCpu.setInstrumentOut("profilerout.txt");
```
This calls a C++ method that sets the appropriate file to use. Each backend will have this call. Note that we don't put this in NativeOps (the parent backend agnostic interface for this) because this normally should not be included in any builds due to overhead.

Consequences
Advantages
Easier debugging of the nd4j and libnd4j code base.
Improved understanding of program execution flow.
Drawbacks
Limited to GCC compiler.
May introduce additional complexity during compilation.
Increased overhead when enabled.
