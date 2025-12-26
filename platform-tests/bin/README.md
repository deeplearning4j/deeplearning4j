# Java Configuration for Surefire and Memory Debugging

## Overview

The "java" file here is actually a shell script we use to allow us to customize surefire test execution via the `<jvm>` parameter in surefire.

Surefire "detects" java by checking for a parent bin directory and a java executable. There is no configurable way to pass a wrapper script. Thus we do this.

## Memory Debugging Tools

This wrapper script supports three memory debugging tools that can be enabled via the `TEST_RUNNER_PREFIX` environment variable:

### 1. Valgrind

Traditional memory debugging tool for Linux. Slower but very thorough.

#### Usage
```bash
TEST_RUNNER_PREFIX="valgrind" mvn test
```

#### Features
- Automatically generates JVM suppressions for libjvm.so
- Enables memory leak detection
- Tracks memory origins
- Keeps allocation and free stack traces
- Generates suppression file automatically

#### Options
The script automatically adds:
- `--error-limit=no` - No limit on error reporting
- `--suppressions=<file>` - JVM-specific suppressions
- `--track-origins=yes` - Track uninitialized value origins
- `--keep-stacktraces=alloc-and-free` - Keep both allocation and deallocation traces
- `-Djava.compiler=none` - Disables JIT for better stack traces

### 2. AddressSanitizer (ASAN)

Fast memory error detector with leak detection. Recommended for most cases.

#### Usage
```bash
TEST_RUNNER_PREFIX="asan" mvn test
```

#### Features
- 2-3x slowdown (vs Valgrind's 20-50x)
- Detects memory leaks with full stack traces
- Automatic detection of libasan.so
- Reports written to `/tmp/asan.log.<pid>`
- Works with JNI and native code

#### Configuration
The script automatically sets:
```bash
LD_PRELOAD=/usr/lib*/libasan.so.*
ASAN_OPTIONS="alloc_dealloc_mismatch=0:detect_leaks=1:new_delete_type_mismatch=0:halt_on_error=0:exitcode=0:report_objects=1:use_stacks=1:use_registers=1:leak_check_at_exit=1:fast_unwind_on_malloc=0:log_path=/tmp/asan.log"
```

#### Installation
```bash
# Fedora/RHEL
sudo dnf install libasan

# Ubuntu/Debian
sudo apt-get install libasan6
```

#### Viewing Reports
```bash
# After test run
cat /tmp/asan.log.*
```

#### Note on Exit Behavior
ASAN may cause the JVM to hang during shutdown after writing leak reports. This is expected. The leak reports are written **before** the hang, so you can safely kill the process:
```bash
kill -9 <pid>
```

### 3. CUDA Compute Sanitizer

Memory debugging for CUDA applications.

#### Usage
```bash
TEST_RUNNER_PREFIX="compute-sanitizer" mvn test
```

#### Features
- CUDA-specific memory error detection
- GPU memory leak detection
- Race condition detection

## Script Behavior

### JVM Detection
The script automatically locates `libjvm.so`:
1. Checks `$libjvm_so` environment variable
2. Falls back to finding java binary and locating libjvm.so in `$JAVA_HOME`
3. Exits with error if libjvm.so cannot be found

### Suppression File Generation (Valgrind)
For Valgrind, the script generates suppressions for these error types:
- addr1, addr2, addr4, addr8 (address errors)
- value1, value2, value4, value8 (uninitialized value errors)
- jump, cond (conditional jump errors)

The suppression file is automatically cleaned up after test execution.

### JIT Compilation
When using any memory debugger, the script automatically adds `-Djava.compiler=none` to disable JIT compilation for better stack traces and more predictable behavior.

## Environment Variables

### Required
- None - script auto-detects configuration

### Optional
- `TEST_RUNNER_PREFIX` - Set to "valgrind", "asan", or "compute-sanitizer"
- `libjvm_so` - Override libjvm.so location
- `CUDA_VISIBLE_DEVICES` - Set to 1 by default

## Examples

### Run single test with ASAN
```bash
TEST_RUNNER_PREFIX="asan" mvn test -Dtest=MyTest
```

### Run all tests with Valgrind
```bash
TEST_RUNNER_PREFIX="valgrind" mvn test
```

### Run with custom ASAN options
```bash
export LD_PRELOAD=/usr/lib64/libasan.so.8
export ASAN_OPTIONS="detect_leaks=1:log_path=/tmp/my_asan.log"
mvn test
```

### Debug CUDA tests
```bash
TEST_RUNNER_PREFIX="compute-sanitizer" mvn test -Dtest=CudaTest
```

## Troubleshooting

### ASAN library not found
```
ERROR: libasan.so not found. Install with:
  Fedora/RHEL: sudo dnf install libasan
  Ubuntu/Debian: sudo apt-get install libasan6
```

### Process hangs with ASAN
This is expected behavior when the JVM has complex native cleanup. The leak reports are written before the hang. Just kill the process:
```bash
ps aux | grep java
kill -9 <pid>
cat /tmp/asan.log.*
```

### Valgrind too slow
Use ASAN instead - it's 10x faster:
```bash
TEST_RUNNER_PREFIX="asan" mvn test
```

## Performance Comparison

| Tool | Slowdown | Memory Overhead | Setup | Best For |
|------|----------|-----------------|-------|----------|
| ASAN | 2-3x | 2x | Easy | General debugging |
| Valgrind | 20-50x | 10x+ | Easy | Thorough analysis |
| Compute Sanitizer | 2-10x | Varies | Medium | CUDA debugging |

## References

- ADR 0049: AddressSanitizer Memory Leak Detection
- ADR 0026: Libnd4j Method Backtraces
- ADR 0032: C++ Debugging
- Valgrind Manual: https://valgrind.org/docs/manual/
- AddressSanitizer: https://github.com/google/sanitizers/wiki/AddressSanitizer
- CUDA Compute Sanitizer: https://docs.nvidia.com/compute-sanitizer/
