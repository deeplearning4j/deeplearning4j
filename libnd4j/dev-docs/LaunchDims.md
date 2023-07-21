# libnd4j CUDA Kernel Launch Configuration

This part of the libnd4j codebase provides a centralized way to handle CUDA launch configurations. It allows customization of launch dimensions using environment variables.

## Conceptual Overview

The code sets up CUDA launch configurations for a wide range of algorithms used in the libnd4j codebase. These configurations are expressed as `dim3` structures, representing grid and block dimensions for launching CUDA kernels. The configurations are stored in a map, improving performance via a caching mechanism.

### Key Feature

A significant feature of this code is the flexibility it offers, allowing users to specify custom launch dimensions through environment variables. This enables modifications of these configurations at both compile-time and runtime.

## Examples of Usage

```cpp
// Fetching Launch Dimensions
dim3 launchDims = getLaunchDims("matrixMultiply");
//invoke your cuda function
yourKernel<<<launchDims.y,launchDims.x,launchDims.z>>>
//This returns the blocks, threads and shared memory used for the kernel.
```


The launch configuration can be found in
```bash
../include/execution/cuda/LaunchDims.h
../include/execution/cuda/LaunchDims.cu
```

Every kernel has environment variables that work at build time and runtime.
Build time is used to modify defaults. Runtime is used to override buildtime and defaults.

