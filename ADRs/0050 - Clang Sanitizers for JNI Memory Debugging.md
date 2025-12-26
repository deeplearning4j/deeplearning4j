# Clang Sanitizers for JNI Memory Debugging

## Status

Implemented

Proposed by: Adam Gibson (04-11-2025)
Discussed with: N/A

## Context

Memory safety is critical for the deeplearning4j native libraries, which interact with the JVM through JNI. Traditional debugging tools like Valgrind are slow and impractical for large-scale testing. Clang provides a suite of sanitizers (AddressSanitizer, MemorySanitizer, LeakSanitizer) that offer fast, precise detection of memory errors, undefined behavior, and memory leaks.

However, using Clang sanitizers with JNI presents unique challenges:

1. **Runtime Library Loading Order**: The sanitizer runtime must be loaded before the JVM to properly intercept memory allocation functions
2. **Shared Library Dependencies**: Sanitizer runtimes depend on C++ standard library (`libc++.so`)
3. **CMake Configuration**: Build system must correctly configure sanitizer compile and link flags
4. **RPATH Configuration**: Shared libraries must embed runtime paths to find sanitizer libraries
5. **JNI-Specific Issues**: JVM shutdown sequences can conflict with sanitizer exit handlers
6. **Different Sanitizers**: Each sanitizer (ASAN, MSAN, LSAN) has different requirements and configurations

This ADR documents the complete configuration and usage patterns for Clang sanitizers with deeplearning4j's JNI bindings.

## Decision

We implement a comprehensive sanitizer integration system with the following components:

### 1. Build System Configuration

#### CMake Sanitizer Support

**File**: `libnd4j/cmake/CompilerFlags.cmake`

The build system supports three sanitizer modes via the `SD_SANITIZERS` CMake variable:

- `leak` - LeakSanitizer (memory leak detection)
- `memory` - MemorySanitizer (uninitialized memory reads)
- `address` - AddressSanitizer (buffer overflows, use-after-free)

**Configuration**:

```cmake
# Enable via Maven property
-Dlibnd4j.sanitizers=leak    # or memory, or address
```

#### Sanitizer Compilation Flags

```cmake
set(SANITIZE_FLAGS " -fPIC -fsanitize=${SD_SANITIZERS} -fno-sanitize-recover=all -fuse-ld=gold -gline-tables-only")

# MemorySanitizer-specific: Use ignorelist to skip instrumentation of external libraries
if(SD_SANITIZERS MATCHES "memory")
    set(SANITIZE_FLAGS "${SANITIZE_FLAGS} -fsanitize-ignorelist=${CMAKE_CURRENT_SOURCE_DIR}/msan_ignorelist.txt")
    set(SANITIZE_FLAGS "${SANITIZE_FLAGS} -ftls-model=initial-exec")
    message(STATUS "Applied MemorySanitizer ignorelist and TLS model for external libraries")
endif()
```

#### Sanitizer Linking Flags with RPATH

**Critical Fix**: The sanitizer runtime library path must be embedded in the shared library RPATH:

```cmake
# Determine OS-specific lib subdirectory
if(APPLE)
    set(SANITIZER_LIB_SUBDIR "darwin")
elseif(WIN32)
    set(SANITIZER_LIB_SUBDIR "windows")
else()
    set(SANITIZER_LIB_SUBDIR "linux")
endif()

# Determine architecture for sanitizer library names
if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
    set(SANITIZER_ARCH "x86_64")
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|ARM64")
    set(SANITIZER_ARCH "aarch64")
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "ppc64le")
    set(SANITIZER_ARCH "powerpc64le")
else()
    set(SANITIZER_ARCH ${CMAKE_SYSTEM_PROCESSOR})
endif()

set(SANITIZER_LIB_PATH "${CLANG_RESOURCE_DIR}/lib/${SANITIZER_LIB_SUBDIR}")

# Add library search path and RPATH for sanitizer runtime
if(APPLE)
    set(SANITIZE_LINK_FLAGS "${SANITIZE_LINK_FLAGS} -L${SANITIZER_LIB_PATH} -Wl,-rpath,${SANITIZER_LIB_PATH}")
elseif(WIN32)
    set(SANITIZE_LINK_FLAGS "${SANITIZE_LINK_FLAGS} -L${SANITIZER_LIB_PATH}")
else()
    set(SANITIZE_LINK_FLAGS "${SANITIZE_LINK_FLAGS} -L${SANITIZER_LIB_PATH} -Wl,-rpath,${SANITIZER_LIB_PATH}")
endif()
```

#### Explicit Runtime Library Linking

```cmake
# MemorySanitizer explicit runtime linking
if(SD_SANITIZERS MATCHES "memory")
    set(SANITIZE_LINK_FLAGS "${SANITIZE_LINK_FLAGS} -lclang_rt.msan-${SANITIZER_ARCH} -lclang_rt.msan_cxx-${SANITIZER_ARCH}")
    message(STATUS "Added explicit MSan runtime libraries for gold linker (${SANITIZER_ARCH})")
endif()

# LeakSanitizer/AddressSanitizer explicit runtime linking
if(SD_SANITIZERS MATCHES "leak" OR SD_SANITIZERS MATCHES "address")
    set(SANITIZE_LINK_FLAGS "${SANITIZE_LINK_FLAGS} -lclang_rt.asan-${SANITIZER_ARCH}")
    message(STATUS "Added explicit ASAN shared runtime library for leak/address sanitizer (${SANITIZER_ARCH}, JNI compatibility)")
endif()
```

**Why This Is Required**:
- When using gold linker with clang sanitizers, `-fsanitize=` flag does NOT automatically add RPATH
- Only adds `-L` (library search path) which works at link time but fails at runtime
- Must explicitly add `-Wl,-rpath,${SANITIZER_LIB_PATH}` for runtime library loading

### 2. Maven Integration

#### Sanitizer Profile

**File**: `nd4j/nd4j-backends/nd4j-backend-impls/nd4j-native/pom.xml`

```xml
<profile>
    <id>sanitizer</id>
    <activation>
        <activeByDefault>false</activeByDefault>
        <property>
            <name>libnd4j.sanitizers</name>
        </property>
    </activation>
    <properties>
        <platform.sanitizer.flag>-fsanitize=${libnd4j.sanitizers}</platform.sanitizer.flag>
        <javacpp.platform.compiler>${platform.compiler.actual}</javacpp.platform.compiler>
    </properties>
    <build>
        <plugins>
            <plugin>
                <groupId>org.bytedeco</groupId>
                <artifactId>javacpp</artifactId>
                <configuration>
                    <compilerOptions>
                        <compilerOption>${platform.sanitizer.flag}</compilerOption>
                    </compilerOptions>
                    <propertyKeysAndValues>
                        <property>
                            <name>platform.compiler.output</name>
                            <value>-Wl,-rpath,$ORIGIN/ -Wl,-z,noexecstack -Wl,-Bsymbolic -Wall -fPIC -pthread ${platform.sanitizer.flag} -shared -o </value>
                        </property>
                    </propertyKeysAndValues>
                </configuration>
            </plugin>
        </plugins>
    </build>
</profile>
```

### 3. Runtime Configuration

#### JNI Library Loading with LD_PRELOAD

The sanitizer runtime must be preloaded before the JVM to intercept memory allocation functions:

```bash
LD_PRELOAD="<libc++.so> <sanitizer_runtime.so>" java ...
```

**Order Matters**: C++ standard library first, then sanitizer runtime.

#### LeakSanitizer (LSAN)

**Build Command**:
```bash
mvn -Pcpu \
    -Dlibnd4j.buildthreads=14 \
    -Dlibnd4j.sanitize=ON \
    -Dlibnd4j.sanitizers=leak \
    -Dlibnd4j.calltrace=ON \
    -Dlibnd4j.build=debug \
    -Dlibnd4j.compiler=clang \
    -pl libnd4j,:nd4j-cpu-backend-common,:nd4j-native \
    clean install -DskipTests
```

**Runtime Command**:
```bash
export LSAN_OPTIONS="print_summary=1:log_path=lsan.log:symbolize=1:external_symbolizer_path=/home/linuxbrew/.linuxbrew/bin/llvm-symbolizer:leak_check_at_exit=1:suppressions=/path/to/lsan_suppressions.txt:verbosity=1:log_threads=1:report_objects=1:max_leaks=100:detect_leaks=1"

LD_PRELOAD="/home/linuxbrew/.linuxbrew/Cellar/llvm/20.1.6/lib/libc++.so.1 /home/linuxbrew/.linuxbrew/Cellar/llvm/20.1.6/lib/clang/20/lib/linux/libclang_rt.asan-x86_64.so" \
java -cp your-app.jar YourMainClass
```

**Valid LSAN Options**:
- `print_summary` - Print summary of leaks (default: true)
- `log_path` - Path to log file (default: stderr)
- `symbolize` - Enable symbolization (default: true)
- `external_symbolizer_path` - Path to llvm-symbolizer
- `leak_check_at_exit` - Check for leaks at program exit (default: true)
- `suppressions` - Path to suppressions file
- `verbosity` - Log verbosity level (0-2)
- `log_threads` - Log thread information
- `report_objects` - Report leaked object addresses
- `max_leaks` - Maximum number of leaks to report (0 = unlimited)
- `detect_leaks` - Enable leak detection (default: true)

**Invalid Options** (these cause warnings and are ignored):
- ❌ `halt_on_error` - ASAN-specific, not valid for LSAN
- ❌ `print_stats` - ASAN-specific, not valid for LSAN
- ❌ `atexit` - Use `leak_check_at_exit` instead
- ❌ `exitcode` - ASAN-specific, not valid for LSAN

#### MemorySanitizer (MSAN)

**Build Command**:
```bash
mvn -Pcpu \
    -Dlibnd4j.buildthreads=14 \
    -Dlibnd4j.sanitize=ON \
    -Dlibnd4j.sanitizers=memory \
    -Dlibnd4j.calltrace=ON \
    -Dlibnd4j.build=debug \
    -Dlibnd4j.compiler=clang \
    -pl libnd4j,:nd4j-cpu-backend-common,:nd4j-native \
    clean install -DskipTests
```

**Runtime Command**:
```bash
export MSAN_OPTIONS="halt_on_error=0:exitcode=0:print_stats=1:log_path=msan.log:symbolize=1:external_symbolizer_path=/home/linuxbrew/.linuxbrew/bin/llvm-symbolizer"

LD_PRELOAD="/home/linuxbrew/.linuxbrew/Cellar/llvm/20.1.6/lib/libc++.so.1 /home/linuxbrew/.linuxbrew/Cellar/llvm/20.1.6/lib/clang/20/lib/linux/libclang_rt.msan-x86_64.so" \
java -cp your-app.jar YourMainClass
```

**MSAN-Specific Configuration**:

1. **Ignorelist File**: `libnd4j/msan_ignorelist.txt`
   - Excludes external libraries from instrumentation
   - Prevents false positives from OpenBLAS, system libraries

2. **TLS Model**: `-ftls-model=initial-exec`
   - Required for MemorySanitizer with dynamic loading
   - NOT used for LeakSanitizer or AddressSanitizer

**Valid MSAN Options**:
- `halt_on_error` - Stop on first error (default: true)
- `exitcode` - Exit code when errors detected (default: 77)
- `print_stats` - Print statistics at exit
- `log_path` - Path to log file
- `symbolize` - Enable symbolization
- `external_symbolizer_path` - Path to llvm-symbolizer

#### AddressSanitizer (ASAN)

**Build Command**:
```bash
mvn -Pcpu \
    -Dlibnd4j.buildthreads=14 \
    -Dlibnd4j.sanitize=ON \
    -Dlibnd4j.sanitizers=address \
    -Dlibnd4j.calltrace=ON \
    -Dlibnd4j.build=debug \
    -Dlibnd4j.compiler=clang \
    -pl libnd4j,:nd4j-cpu-backend-common,:nd4j-native \
    clean install -DskipTests
```

**Runtime Command**:
```bash
export ASAN_OPTIONS="alloc_dealloc_mismatch=0:new_delete_type_mismatch=0:halt_on_error=0:exitcode=0:report_objects=1:use_stacks=1:use_registers=1:fast_unwind_on_malloc=0:log_path=asan.log:detect_leaks=1:leak_check_at_exit=1"

LD_PRELOAD="/home/linuxbrew/.linuxbrew/Cellar/llvm/20.1.6/lib/libc++.so.1 /home/linuxbrew/.linuxbrew/Cellar/llvm/20.1.6/lib/clang/20/lib/linux/libclang_rt.asan-x86_64.so" \
java -cp your-app.jar YourMainClass
```

**Valid ASAN Options**:
- `alloc_dealloc_mismatch` - Detect malloc/delete mismatches
- `new_delete_type_mismatch` - Detect new/delete type mismatches
- `halt_on_error` - Stop on first error
- `exitcode` - Exit code when errors detected
- `report_objects` - Report object addresses
- `use_stacks` - Use stack information
- `use_registers` - Use register information
- `fast_unwind_on_malloc` - Use fast unwinding for malloc
- `log_path` - Path to log file
- `detect_leaks` - Enable leak detection (ASAN includes LSAN)
- `leak_check_at_exit` - Check leaks at exit

### 4. Platform-Specific Paths

The configuration is fully OS and architecture agnostic:

**Linux x86_64**:
```bash
LD_PRELOAD="/home/linuxbrew/.linuxbrew/Cellar/llvm/20.1.6/lib/libc++.so.1 \
/home/linuxbrew/.linuxbrew/Cellar/llvm/20.1.6/lib/clang/20/lib/linux/libclang_rt.asan-x86_64.so"
```

**Linux aarch64**:
```bash
LD_PRELOAD="/home/linuxbrew/.linuxbrew/Cellar/llvm/20.1.6/lib/libc++.so.1 \
/home/linuxbrew/.linuxbrew/Cellar/llvm/20.1.6/lib/clang/20/lib/linux/libclang_rt.asan-aarch64.so"
```

**macOS x86_64**:
```bash
DYLD_INSERT_LIBRARIES="/home/linuxbrew/.linuxbrew/Cellar/llvm/20.1.6/lib/libc++.dylib:\
/home/linuxbrew/.linuxbrew/Cellar/llvm/20.1.6/lib/clang/20/lib/darwin/libclang_rt.asan-x86_64.dylib"
```

**macOS arm64**:
```bash
DYLD_INSERT_LIBRARIES="/home/linuxbrew/.linuxbrew/Cellar/llvm/20.1.6/lib/libc++.dylib:\
/home/linuxbrew/.linuxbrew/Cellar/llvm/20.1.6/lib/clang/20/lib/darwin/libclang_rt.asan-arm64.dylib"
```

### 5. Finding Sanitizer Runtime Paths

To find the correct paths for your system:

```bash
# Find LLVM/Clang installation
which clang++

# Get resource directory
clang++ -print-resource-dir
# Output: /home/linuxbrew/.linuxbrew/Cellar/llvm/20.1.6/lib/clang/20

# List sanitizer libraries
ls $(clang++ -print-resource-dir)/lib/*/libclang_rt.*
```

### 6. Suppressions Files

#### LeakSanitizer Suppressions

**File**: `libnd4j/lsan_suppressions.txt`

The suppressions file categorizes known non-leaks and false positives:

1. **JVM Internal Allocations**: Memory managed by JVM garbage collector
   - `libjvm.so` - JVM internal memory
   - `JavaCalls::*` - Method invocation infrastructure
   - `Reflection::*` - Java reflection API
   - `JVM_StartThread` - Thread management

2. **System Libraries**: OS and runtime allocations
   - `libstdc++.so`, `libc.so`, `libpthread.so` - Standard libraries
   - `ld-linux` - Dynamic linker

3. **Third-Party Libraries**: External dependencies
   - `libopenblas` - BLAS library internal state
   - `flatbuffers::` - Serialization static buffers
   - `libjnind4jcpu.so` - JavaCPP JNI bridge

4. **libnd4j Singletons** (Intentional lifetime allocations):
   - `samediff::ThreadPool::getInstance` - Thread pool singleton
   - `ConstantHelper::getInstance` - Constant caching
   - `Environment::getInstance` - Global configuration

5. **Template Instantiations**: Operation implementations
   - `sd::SpecialMethods<*>` - Type-specific operations
   - `sd::ops::helpers::concat_` - Concat operation variants
   - `sd::ops::DeclarableOp::execute` - Operation dispatch

6. **Thread-Local Storage**: Per-thread allocations
   - `pthread_create`, `__tls_get_addr` - Thread infrastructure
   - `__cxa_thread_atexit` - Thread cleanup

7. **Known False Positives**: RAII-managed with timing issues
   - `sd::NDArray::NDArray` - Temporary arrays
   - `sd::graph::Context` - Operation context

**Full suppressions file** contains ~50 patterns organized by category. See `libnd4j/lsan_suppressions.txt` for complete list.

#### MemorySanitizer Ignorelist

**File**: `libnd4j/msan_ignorelist.txt`

```
# Ignore external libraries not compiled with MSAN
src:*/openblas/*
src:*/flatbuffers/*
src:*/javacpp/*

# System libraries
fun:*@GLIBC*
```

**Why Required**: MemorySanitizer requires ALL code (including dependencies) to be instrumented. The ignorelist excludes external libraries to prevent false positives from uninstrumented code.

## Implementation Details

### Build Flow

1. **CMake Configuration**:
   ```bash
   cmake -DSD_SANITIZE=ON -DSD_SANITIZERS=leak ..
   ```

2. **Compilation**:
   - All source files compiled with `-fsanitize=leak`
   - RPATH embedded in shared libraries

3. **Linking**:
   - Sanitizer runtime explicitly linked: `-lclang_rt.asan-x86_64`
   - RPATH includes sanitizer library path

4. **Maven Build**:
   - JavaCPP bindings compiled with same sanitizer flags
   - JNI library linked with `-fsanitize=leak`

### Runtime Flow

1. **LD_PRELOAD**:
   - Loads `libc++.so` first (sanitizer dependency)
   - Loads sanitizer runtime (`libclang_rt.asan-x86_64.so`)

2. **JVM Startup**:
   - Sanitizer runtime intercepts malloc/free
   - JNI libraries load successfully (RPATH finds sanitizer runtime)

3. **Execution**:
   - Memory operations instrumented by sanitizer
   - Errors/leaks logged to configured path

4. **JVM Shutdown**:
   - Leak check performed at exit (if `leak_check_at_exit=1`)
   - Reports written to log files

### Verification

**Check Binary RPATH**:
```bash
readelf -d /path/to/libnd4jcpu.so | grep RPATH
# Should show: RPATH: /home/linuxbrew/.linuxbrew/Cellar/llvm/20.1.6/lib/clang/20/lib/linux
```

**Check Binary Dependencies**:
```bash
ldd /path/to/libnd4jcpu.so | grep clang_rt
# Should show: libclang_rt.asan-x86_64.so => /path/to/libclang_rt.asan-x86_64.so
```

**Test Sanitizer Loading**:
```bash
LD_PRELOAD="..." java -version
# Should NOT show "symbol lookup error"
```

## Consequences

### Advantages

1. **Fast Detection**: 2-3x slowdown vs Valgrind's 20-50x
2. **Precise Reports**: Exact stack traces for memory errors
3. **Multiple Sanitizers**: ASAN, MSAN, LSAN for different error types
4. **JNI Compatible**: Works correctly with Java/native boundary
5. **Automated**: No code changes required
6. **OS Agnostic**: Works on Linux, macOS, Windows with appropriate configuration
7. **Architecture Agnostic**: Supports x86_64, aarch64, ppc64le

### Drawbacks

1. **LD_PRELOAD Required**: Cannot work without runtime preloading
2. **Build Complexity**: Requires careful CMake and Maven configuration
3. **Symbol Dependencies**: Sanitizer runtime needs libc++
4. **JVM Warning**: "ASan runtime does not come first" warning (harmless)
5. **Memory Overhead**: 2x memory usage during execution
6. **Cannot Mix Sanitizers**: Can only use one sanitizer at a time

### Comparison of Sanitizers

| Feature | LeakSanitizer | MemorySanitizer | AddressSanitizer |
|---------|---------------|-----------------|------------------|
| Detects | Memory leaks | Uninitialized reads | Buffer overflows, UAF |
| Runtime | libclang_rt.asan | libclang_rt.msan | libclang_rt.asan |
| Slowdown | 2x | 3x | 2x |
| Memory overhead | 2x | 3x | 2x |
| Requires rebuild | Yes | Yes (all deps!) | Yes |
| False positives | Low | Medium | Low |
| JNI compatible | Yes | Yes | Yes |
| Ignorelist needed | Optional | Required | Optional |

## Troubleshooting

### "symbol lookup error: undefined symbol: _ZTISt9type_info"

**Cause**: Sanitizer runtime requires C++ standard library

**Fix**: Add `libc++.so` to LD_PRELOAD before sanitizer runtime:
```bash
LD_PRELOAD="/path/to/libc++.so.1 /path/to/libclang_rt.asan-x86_64.so" java ...
```

### "ASan runtime does not come first in initial library list"

**Cause**: JVM loads before sanitizer runtime

**Fix**: This is expected with LD_PRELOAD and JNI. The warning is harmless - sanitizers still function correctly.

### "libclang_rt.asan-x86_64.so: cannot open shared object file"

**Cause**: Library not in LD_LIBRARY_PATH and RPATH not embedded

**Fix**: Verify RPATH is embedded in libnd4jcpu.so:
```bash
readelf -d libnd4jcpu.so | grep RPATH
```

If missing, rebuild with sanitizer configuration from this ADR.

### "WARNING: found 3 unrecognized flag(s): halt_on_error, print_stats, atexit"

**Cause**: Using ASAN-specific flags with LeakSanitizer

**Fix**: Use only valid LSAN flags (see section 3 above)

### MemorySanitizer Reports False Positives from OpenBLAS

**Cause**: OpenBLAS not compiled with MSAN instrumentation

**Fix**: Add to `msan_ignorelist.txt`:
```
src:*/openblas/*
```

## Usage Examples

### Example 1: Detect Memory Leaks in Unit Tests

```bash
# Build with LeakSanitizer
mvn -Pcpu -Dlibnd4j.sanitizers=leak -Dlibnd4j.compiler=clang clean install -DskipTests

# Set environment
export LSAN_OPTIONS="print_summary=1:log_path=lsan.log:verbosity=1:report_objects=1:max_leaks=100"
export LD_PRELOAD="/usr/local/lib/libc++.so.1 /usr/local/lib/clang/20/lib/linux/libclang_rt.asan-x86_64.so"

# Run tests
mvn test -Dtest=YourTestClass

# Check results
cat lsan.log.*
```

### Example 2: Detect Uninitialized Memory Reads

```bash
# Build with MemorySanitizer
mvn -Pcpu -Dlibnd4j.sanitizers=memory -Dlibnd4j.compiler=clang clean install -DskipTests

# Set environment
export MSAN_OPTIONS="print_stats=1:log_path=msan.log:symbolize=1"
export LD_PRELOAD="/usr/local/lib/libc++.so.1 /usr/local/lib/clang/20/lib/linux/libclang_rt.msan-x86_64.so"

# Run application
java -cp app.jar MainClass

# Check results
cat msan.log.*
```

### Example 3: Detect Buffer Overflows

```bash
# Build with AddressSanitizer
mvn -Pcpu -Dlibnd4j.sanitizers=address -Dlibnd4j.compiler=clang clean install -DskipTests

# Set environment
export ASAN_OPTIONS="halt_on_error=1:log_path=asan.log:detect_leaks=1"
export LD_PRELOAD="/usr/local/lib/libc++.so.1 /usr/local/lib/clang/20/lib/linux/libclang_rt.asan-x86_64.so"

# Run application (will halt on first error)
java -cp app.jar MainClass

# Check results
cat asan.log.*
```

## References

- AddressSanitizer: https://clang.llvm.org/docs/AddressSanitizer.html
- MemorySanitizer: https://clang.llvm.org/docs/MemorySanitizer.html
- LeakSanitizer: https://clang.llvm.org/docs/LeakSanitizer.html
- Sanitizer Common Flags: https://github.com/google/sanitizers/wiki/SanitizerCommonFlags
- LLVM Sanitizers Wiki: https://github.com/google/sanitizers/wiki
- `libnd4j/cmake/CompilerFlags.cmake` - Build configuration
- `nd4j-native/pom.xml` - Maven integration
- ADR 0049 - AddressSanitizer Memory Leak Detection (predecessor to this ADR)
