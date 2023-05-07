

## Overview

The nd4j and libnd4j code base can be difficult to debug due to the lack of tools for tracing program execution. We have implemented function tracing using the `-finstrument-functions` flag provided by GCC to trace function calls and exits in the code base. This feature will help developers better understand the flow of the program and identify potential issues.

## How to Enable Function Tracing

To enable function tracing, follow these steps:

1. Build the code base with `-Dlibnd4j.functrace=ON`.
2. Add the correct compiler flags to the javacpp plugin.
3. Add the correct flags to the cmake build.
4. Add a maven profile to each nd4j backend containing the correct compiler flags to work with `-finstrument-functions`.

## Implementation Details

### Compiler Flags

We have used the following compiler flags to enable function tracing:

- `-Bsymbolic`: Bind references to global symbols at link time, reducing the runtime overhead.
- `-rdynamic`: Export all dynamic symbols to the dynamic symbol table, making them available for backtracing.
- `-fno-omit-frame-pointer`: Do not omit the frame pointer, allowing for accurate backtraces.
- `-fno-optimize-sibling-calls`: Disable sibling call optimization to maintain the correct call stack.
- `-finstrument-functions`: Enable instrumentation of function entry and exit points.
- `-g`: Enable debugging information.
- `-O0`: Set the optimization level to zero for easier debugging.

### Setting the Output File

We have implemented a method in each backend to set the output file for tracing results. For example, in the Nd4jCpu class:

```java
Nd4jCpu nd4jCpu = (Nd4jCpu) NativeOpsHolder.getInstance().getDeviceNativeOps();
nd4jCpu.setInstrumentOut("profilerout.txt");
```

This calls a C++ method that sets the appropriate file to use. Each backend will have this call. Note that we don't put this in NativeOps (the parent backend agnostic interface for this) because this normally should not be included in any builds due to overhead.

### LD_PRELOAD and LD_DEBUG

To ensure the correct implementation of the enter/exit functions is used, `LD_PRELOAD` is utilized to preload the libnd4j binary generated during the build. The built-in libc implementation of these functions is a no-op, so preloading the libnd4j binary is necessary.

`LD_DEBUG` can be used to verify that the correct implementation is being used by showing the symbols being loaded and their origin.

## Sample Log Output

```
g long> > >::end() (/home/agibsonccc/Documents/GitHub/deeplearning4j/libnd4j/blasbuild/cpu/blas/libnd4jcpu.so)
 enter std::operator==(std::_Rb_tree_iterator<std::pair<int const, long long> > const&, std::_Rb_tree_iterator<std::pair<int const, long long> > const&) (/home/agibsonccc/Documents/GitHub/deeplearning4j/libnd4j/blasbuild/cpu/blas/libnd4jcpu.so)
 exit std::operator==(std::_Rb_tree_iterator<std::pair<int const, long long> > const&, std::_Rb_tree_iterator<std::pair<int const, long long> > const&) (/home/agibsonccc/Documents/GitHub/deeplearning4j/libnd4j/blasbuild/cpu/blas/libnd4