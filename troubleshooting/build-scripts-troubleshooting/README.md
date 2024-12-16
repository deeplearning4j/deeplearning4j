# Debug Build Scripts

This directory contains specialized build scripts for debugging various aspects of libnd4j. Each script is configured with specific debug flags for different debugging scenarios.

## Available Scripts

### CPU Backend Scripts

1. **build-cpu-backend-full-debug.sh**
   - Full debug build with all debug flags enabled
   - Includes calltrace, printmath, printindices, and address sanitizer
   - Best for comprehensive debugging of CPU issues

2. **build-cpu-backend-preprocessor-debug.sh**
   - Enables preprocessor output for macro debugging
   - Useful for investigating template and macro expansion issues

3. **build-cpu-backend-valgrind-debug.sh**
   - Optimized for Valgrind analysis
   - Includes memory leak detection and call tracing
   - Disables optimizations for better debugging

4. **build-cpu-backend-onednn-debug.sh**
   - Debug build with OneDNN support
   - Includes all debug flags
   - Useful for OneDNN-specific issues

### CUDA Backend Scripts

1. **build-cuda-backend-full-debug.sh**
   - Full debug build for CUDA with all debug flags
   - Includes NVCC output preservation
   - Best for comprehensive debugging of CUDA issues

2. **build-cuda-backend-compute-sanitizer.sh**
   - Optimized for NVIDIA Compute Sanitizer
   - Includes debug symbols and reduced optimization
   - Best for CUDA memory and race condition debugging

## Usage Notes

1. **General Debug Builds**
   ```bash
   # For CPU debugging
   ./build-cpu-backend-full-debug.sh
   
   # For CUDA debugging
   ./build-cuda-backend-full-debug.sh
   ```

2. **Macro Debugging**
   ```bash
   ./build-cpu-backend-preprocessor-debug.sh
   # Check preprocessed files in the preprocessed/ directory
   ```

3. **Memory Analysis**
   ```bash
   # For CPU with Valgrind
   ./build-cpu-backend-valgrind-debug.sh
   
   # For CUDA with Compute Sanitizer
   ./build-cuda-backend-compute-sanitizer.sh
   ```

## Flag Combinations

The scripts use various combinations of these debug flags:

- `libnd4j.build=debug`: Enables debug symbols
- `libnd4j.calltrace=ON`: Enables function call tracing
- `libnd4j.printmath=ON`: Prints math operations
- `libnd4j.printindices=ON`: Prints array indices
- `libnd4j.sanitize=ON`: Enables sanitizer support
- `libnd4j.preprocess=ON`: Enables preprocessor output
- `libnd4j.keepnvcc=ON`: Preserves NVCC output (CUDA)

## Important Notes

1. Debug builds are significantly slower than release builds
2. Some flag combinations may be incompatible (e.g., different sanitizers)
3. Ensure sufficient disk space for debug symbols and preprocessor output
4. CUDA debugging requires appropriate NVIDIA tools installation