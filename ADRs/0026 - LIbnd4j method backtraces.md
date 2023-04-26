Function Tracing in nd4j and libnd4j Code Base
Implemented by: Adam Gibson (26-04-2023)

Discussed with: Paul Dubs

Overview
The nd4j and libnd4j code base can be difficult to debug due to the lack of tools for tracing program execution.
We have implemented function tracing using the -finstrument-functions flag provided by GCC to trace 
function calls and exits in the code base. 
This feature will help developers better understand the flow of the program and identify potential issues.

Function tracing can also work with CUDA, as long as GCC is the underlying compiler.

How to Enable Function Tracing
To enable function tracing, follow these steps:

Build the code base with -Dlibnd4j.functrace=ON.
Add the correct compiler flags to the javacpp plugin.
Add the correct flags to the cmake build.
Add a maven profile to each nd4j backend containing the correct compiler flags to work with -finstrument-functions.
Build with Maven
Here are the example Maven build commands for the CPU and CUDA backends, respectively:

For CPU backend:

```
mvn -Dlibnd4j.functrace=ON -Pcpu clean install -DskipTests
```
For CUDA backend:

```
mvn  -Dlibnd4j.functrace=ON -Pcuda clean install -DskipTests
```
Make sure you have the necessary profiles configured in your pom.xml to enable tracing for the respective backend.

Implementation Details
Compiler Flags
We have used the following compiler flags to enable function tracing:

-Bsymbolic: Bind references to global symbols at link time, reducing the runtime overhead.
-rdynamic: Export all dynamic symbols to the dynamic symbol table, making them available for backtracing.
-fno-omit-frame-pointer: Do not omit the frame pointer, allowing for accurate backtraces.
-fno-optimize-sibling-calls: Disable sibling call optimization to maintain the correct call stack.
-finstrument-functions: Enable instrumentation of function entry and exit points.
-g: Enable debugging information.
-O0: Set the optimization level to zero for easier debugging.
Setting the Output File
We have implemented a method in each backend to set the output file for tracing results.

For the CPU backend, use the following code:

java
```Copy code
Nd4jCpu nd4jCpu = (Nd4jCpu) NativeOpsHolder.getInstance().getDeviceNativeOps();
nd4jCpu.setInstrumentOut("profilerout.txt");
```


For the CUDA backend, use the following code:

java
```
Nd4jCuda nd4jCuda = (Nd4jCuda) NativeOpsHolder.getInstance().getDeviceNativeOps();
nd4jCuda.setInstrumentOut("profilerout.txt");
```
These calls set the appropriate file to use for each backend. Note that we don't put this in NativeOps (the parent backend agnostic interface for this) because this normally should not be included in any builds due to overhead.

LD_PRELOAD and LD_DEBUG
To ensure the correct implementation of the enter/exit functions is used, LD_PRELOAD is utilized to preload the libnd4j binary generated during the build. The built-in libc implementation of these functions is a no-op, so preloading the libnd4j binary is necessary.

LD_DEBUG can be used to verify that the correct implementation is being used by showing the symbols being loaded and their origin.

Sample Log Output
bash
Copy code
g long> > >::end() (/home/agibsonccc/Documents/GitHub/deeplearning4j/libnd4j/blasbuild/cpu/blas/libnd4