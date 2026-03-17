# CLAUDE.md - Development Guide for Deeplearning4j

## Build Commands

**Ask the user for a build command if one isn't provided.** The user is often working on something specific and the build target varies.

**IMPORTANT:** Only build backend-specific modules. **NEVER** include `platform-tests` in a build `-pl` list -- it is only for running tests, never for building with the project.

### CUDA builds

Require `-Pcuda -Dlibnd4j.chip=cuda`:
```bash
mvn -Pcuda -Dlibnd4j.chip=cuda -Dlibnd4j.buildthreads=12 \
  -Dlibnd4j.log=libnd4j-build.log \
  -pl libnd4j,:nd4j-cuda-12.9 \
  clean install -DskipTests 2>&1 | tee cuda-build-output.log
```

### CPU builds

Require `-Pcpu`:
```bash
mvn -Pcpu -Dlibnd4j.buildthreads=12 \
  -Dlibnd4j.log=libnd4j-build.log \
  -pl libnd4j,:nd4j-cpu-backend-common,:nd4j-native \
  clean install -DskipTests 2>&1 | tee cpu-build-output.log
```

### Java-only module install (no native compile)
```bash
mvn install -DskipTests -pl <module>
```

### General rules

- **Selecting a specific backend:** Use `-Dbackend.artifactId=` (e.g., `-Dbackend.artifactId=nd4j-cuda-12.9` or `-Dbackend.artifactId=nd4j-native`).
- Always `install`, never just `compile` -- downstream modules need the jar in the local repo.
- If building C++, always rebuild CUDA bindings too.
- Always use `-Dlibnd4j.buildthreads=12` for native builds.
- **Always pass `-Dlibnd4j.log=libnd4j-build.log`** for native builds. This captures the C++ build log separately.
- **ALL build commands MUST be piped through `tee` to a known file.** Use `mvn ... 2>&1 | tee build-output.log` (pick a descriptive name). This captures the full Maven + native output for review.
- **NEVER use `tail`** on build output. Always use `tee` with a real file name so the complete output is preserved.
- **NEVER use `make` directly — BANNED.** Running `make` in `libnd4j/` only builds the C++ library without regenerating Java bindings. The bindings (preset module) MUST be rebuilt for any C++ change to be visible from Java. Always use the full `mvn` build command that includes both `libnd4j` and the backend bindings module (e.g., `:nd4j-cuda-12.9` or `:nd4j-native`). CMake takes negligible time — there is ZERO benefit to running make directly. Always use the full mvn build.
- **Build timeouts:** Header changes trigger full recompiles that take 30-45 minutes. Always use a sufficiently long timeout (at least 3600000ms / 60 minutes) for build commands. If a build times out, restart the full `mvn` build (not `make`).
- **NEVER change the CUDA compute capability** (`-Dlibnd4j.compute=...`) in a build. Changing compute capability invalidates the entire ccache and forces multi-hour full rebuilds. Use whatever compute capability is already cached.
- **NEVER clear ccache** (`ccache -C`, `ccache --clear`). Clearing ccache forces a full rebuild of the entire C++ codebase which takes hours. There is NEVER a valid reason to clear ccache.
- **NEVER deviate from the standard build command.** Always build both `libnd4j` AND the bindings module together. Partial builds (just libnd4j, just bindings) cause stale/mismatched artifacts.

## Testing

**ALL tests go in `platform-tests`. ALWAYS run tests from there:**
```bash
cd /home/agibsonccc/Documents/GitHub/deeplearning4j/platform-tests && mvn test -Dtest=<TestClass>#<method>
```

- **NEVER** run `mvn test` from the project root -- it triggers full rebuilds of native code and runs everything.
- **NEVER** use jemalloc (`LD_PRELOAD=libjemalloc.so`) unless the user explicitly asks for it.
- Tests run once. Use surefire logs for debugging: `platform-tests/target/surefire-reports/<TestClass>-output.txt`
- Never pipe test output through `tail` -- always capture full output to a file.

### Test Runner Wrapper (`platform-tests/bin/java`)

`platform-tests` has a custom `bin/java` wrapper script that surefire uses as its JVM. It supports diagnostic tools via `-Dtest.prefix`:

- **Valgrind:** `-Dtest.prefix=valgrind` -- thorough memory debugging with automatic JVM suppression files. Focuses on libnd4j errors only.
- **Compute-sanitizer:** `-Dtest.prefix=/usr/local/cuda/bin/compute-sanitizer` -- CUDA memory error detection, race conditions, GPU leak detection.
- **AddressSanitizer:** `-Dtest.prefix=asan` -- fast (2-3x slowdown) memory error detector with leak detection.
- **Nsight Systems:** `-Dtest.prefix=nsys` -- NVIDIA GPU profiling with CUDA/cuBLAS/cuDNN tracing.
- **nvprof:** `-Dtest.prefix=nvprof` -- legacy NVIDIA profiler.

**ALWAYS** use `-Dtest.prefix` with these tools. The wrapper handles suppressions, output files, and JIT disabling automatically.

### Environment Variables Do NOT Work with Surefire

Surefire forks a new JVM. Shell environment variables (`ENV=value mvn test`) do **NOT** propagate. To pass configuration:

1. Add the property to `platform-tests/pom.xml` in the surefire `<configuration>` section under `<environmentVariables>`.
2. Wire it via a `-D` Maven property so it can be set from the command line.
3. **NEVER** rely on `export VAR=value` before `mvn test` -- the forked JVM won't see it.

### Writing Tests

- **Always write standalone isolation tests** when debugging. Reproduce the bug in a minimal test before fixing it.
- **Test all configuration combinations.** Use parameterized/matrix-style tests that enumerate all valid configurations (backends, data types, execution modes, etc.).
- **Make individual configurations runnable.** Structure parameterized tests so a specific broken configuration can be run directly (e.g., via `@MethodSource` with named parameters or `-Dtest=TestClass#method[configName]`).

## Development Rules

### No Workarounds -- EVER

**NEVER** work around a bug. Fix the root cause directly. A workaround is ANY compromise: a shortcut, a guard in the caller, reordering in test code, a "temporary" hack. If you find an issue while working on something else, dispatch a subagent to fix it. Do not move on with a workaround in place.

### Investigate Before Coding

**Fully investigate** every task before writing code. Builds take too long to guess. Read the relevant code, trace values to their origins, understand the architecture. Use subagents to investigate hypotheses in parallel when dealing with difficult bugs.

### Parallelize Work

When dealing with a difficult bug or complex task, **multi-task aggressively**. Dispatch subagents to:
- Investigate competing hypotheses simultaneously
- Fix discovered bugs while you continue the main task
- Run searches across different parts of the codebase

### C++ Header Discipline

**Avoid modifying headers whenever possible.** Header changes invalidate caches and cause hours of rebuilds. If you can refactor code to keep headers unchanged (move logic to .cpp/.cu files, use forward declarations, etc.), do that. Only modify headers when there is no alternative.

### C++ Kernel and Helper Organization

When adding a new helper or kernel:
1. **Study how existing kernels work first.** Look at the pattern: header in `helpers/`, CPU impl in `helpers/cpu/`, CUDA impl in `helpers/cuda/`.
2. **NEVER** add one-off standalone code. Follow the established kernel pattern.
3. **NEVER** add a kernel for one platform and stub the others. Implement for ALL platforms (CPU and CUDA at minimum).
4. Use templates (`<typename T>`) + `BUILD_SINGLE_TEMPLATE` + `BUILD_SINGLE_SELECTOR` -- never hardcode types.
5. Use `getLaunchDims("op_name")` from `LaunchDims.h` -- never hardcode thread counts.
6. Register new ops in `LaunchDims.h` (macros) and `LaunchDims.cu` (both maps).

### C++ Platform Abstraction Macros -- MANDATORY

**ALWAYS** use libnd4j's platform macros instead of raw compiler/platform keywords. These are defined in `libnd4j/include/system/` and ensure code compiles on CPU (GCC, Clang, MSVC) and GPU (NVCC) without `#ifdef` sprawl.

**CUDA function qualifiers** (`system/common.h`) -- NEVER write `__host__`, `__device__`, `__global__` directly:
- `SD_HOST` -- replaces `__host__`
- `SD_DEVICE` -- replaces `__device__`
- `SD_KERNEL` -- replaces `__global__`
- `SD_HOST_DEVICE` -- replaces `__host__ __device__`
- These expand to empty on CPU builds, so the same code compiles everywhere.

**Inline** (`system/common.h`) -- NEVER write `__forceinline__` or platform-specific inline:
- `SD_INLINE` -- portable forced inline (maps to `__forceinline__` on NVCC, `__forceinline` on MSVC, `inline` on GCC/Clang).

**Library export** (`system/common.h`, `system/sd_export.h`) -- NEVER write `__declspec(dllexport)` or `__attribute__((visibility))`:
- `SD_LIB_EXPORT` -- public API visibility.
- `SD_LIB_HIDDEN` -- hidden visibility.

**OpenMP pragmas** (`system/openmp_pragmas.h`) -- NEVER write raw `#pragma omp` directives:
- `PRAGMA_OMP_PARALLEL_FOR` -- replaces `#pragma omp parallel for`
- `PRAGMA_OMP_PARALLEL_FOR_SIMD` -- replaces `#pragma omp parallel for simd`
- `PRAGMA_OMP_PARALLEL_FOR_THREADS(n)` -- parallel for with thread count
- `PRAGMA_OMP_PARALLEL_FOR_COLLAPSE(n)` -- parallel for with loop collapsing
- `PRAGMA_OMP_PARALLEL_FOR_REDUCTION(...)` -- parallel for with reduction
- `PRAGMA_OMP_SIMD` -- replaces `#pragma omp simd`
- `PRAGMA_OMP_ATOMIC` -- replaces `#pragma omp atomic`
- `PRAGMA_OMP_CRITICAL` -- replaces `#pragma omp critical`
- `OMP_IF(args)` / `OMP_SCHEDULE(args)` -- conditional/schedule wrappers
- These are no-ops on MSVC (which has limited OpenMP support) and fully expand on GCC/Clang.

**SIMD safety** (`system/openmp_pragmas.h`):
- `PRAGMA_OMP_DECLARE_SIMD_SAFE` -- SIMD declaration that suppresses warnings for unsupported types (compiler-specific push/pop).

**Template instantiation** (`system/type_boilerplate.h`) -- NEVER manually instantiate templates for each type:
- `BUILD_SINGLE_TEMPLATE(NAME, SIGNATURE, TYPES)` -- instantiate a template for all types in a type list
- `BUILD_SINGLE_SELECTOR(XTYPE, NAME, SIGNATURE, TYPES)` -- runtime type dispatch (switch on DataType)
- `BUILD_DOUBLE_SELECTOR(XTYPE, YTYPE, ...)` -- two-type dispatch
- `BUILD_TRIPLE_SELECTOR(XTYPE, YTYPE, ZTYPE, ...)` -- three-type dispatch

**Math type promotion** (`math/templatemath.h`):
- `SD_PROMOTE_FUNC(FUNC_NAME, BODY)` -- wraps binary math in automatic type promotion

**Platform-specific ops** (`system/platform_boilerplate.h`):
- `DECLARE_PLATFORM(NAME, ENGINE)` -- declare a platform-specific op implementation
- `PLATFORM_IMPL(NAME, ENGINE)` -- implement the op
- `PLATFORM_CHECK(NAME, ENGINE)` -- check if the platform can run this op

**Memory alignment** (`system/common.h`):
- `SD_ALIGN32` -- 32-byte alignment attribute

### Configuration and Environment

When adding any new configuration option:
1. Add it to `libnd4j/include/system/Environment.h` and `Environment.cpp` (C++ side).
2. Add it to `nd4j/nd4j-backends/nd4j-api-parent/nd4j-api/src/main/java/org/nd4j/linalg/factory/Environment.java` (Java interface).
3. Add the system property constant to `ND4JSystemProperties.java`.
4. Wire it through the preset if it needs JNI exposure.

### Generated Code -- Do NOT Modify

**NEVER** edit generated code directly. Update the **presets** instead:
- `nd4j-native-preset`, `nd4j-cuda-preset`, `nd4j-minimizer-preset`, `nd4j-tpu-preset`
- Headers are automatically parsed by JavaCPP. Most of the time, modifying a header is sufficient -- the presets pick it up.

### ADR (Architecture Decision Records)

New features and significant changes require ADR checks:
1. Read existing ADRs in `ADRs/` to understand the format (Status, Context, Decision sections).
2. **Create a new ADR** for new features or architectural decisions.
3. **Update existing ADRs** when modifying behavior covered by a prior decision.

### Double-Free and Shutdown Crashes

When debugging double-frees, use-after-free, or shutdown crashes:
- **Always check the DeallocatorService** for proper shutdown flags.
- Verify that deallocation ordering respects object lifetimes.
- Check that `setCloseable(false)` / `setConstant(true)` poisoning is properly undone.

### DSP Development Rules

**NEVER fall back to slot-by-slot execution.** If DSP (DynamicShapePlan) has a bug or a kernel fails, fix the root cause. Falling back to slot-by-slot is a workaround -- it hides the real problem and forfeits the performance gains DSP exists to provide.

**NEVER skip kernels.** If a Triton kernel or any DSP kernel crashes, produces wrong results, or fails to compile, fix the kernel. Do NOT skip it, stub it out, or route around it. Every kernel must have a working baseline. ALL Triton kernels need a working baseline -- no exceptions.

**FIX bugs encountered during profiling.** When profiling DSP performance and you encounter a crash, wrong result, or other bug along the way, **fix it immediately** (dispatch a subagent if needed). Do NOT defer, skip, or work around it. Profiling is not an excuse to ignore correctness.

**Maximize configuration optionality.** The goal is to be able to blend different execution configurations (graph replay, slot-based, Triton-compiled, cuBLAS fallback, etc.) for optimal performance. Skipping kernels or falling back to slot-by-slot destroys this optionality. Every execution path must work correctly so configurations can be mixed freely.

### DSP Diagnostics

When debugging DSP (DynamicShapePlan) related issues, **always use DSP diagnostics**. Enable via `-Dnd4j.dsp.diagnostics=<level>` (e.g., `MEMORY`, `EXECUTION`, `ALL`). DSP diagnostics provide structured, reusable tracing for plan compilation, slot execution, memory allocation, and kernel launches. **Do NOT add ad-hoc printf/logging for DSP debugging** -- use the existing diagnostic infrastructure.

### Printing Array Values

**Use `array->printIndexedBuffer()` instead of manual loops** when you need to print NDArray values for debugging. This method handles all data types, formatting, and edge cases correctly. Manual `for` loops over buffer elements are error-prone (wrong strides, wrong types, missing sync) and wasteful.

### Additional Rules

- **No `.arr` or `.shape` in model import code** -- use `sd.shape(..)` and `sd.rank(..)`. Everything must be variable-based for dynamic shape support.
- **No fully qualified class names in code** -- use imports.
- **Trace values to roots** -- always search for the origin of a value before attempting a fix.
- **`MALLOC_CHECK_=3` does NOT work reliably** -- don't rely on it.
- **Make diagnostics reusable.** When adding diagnostic or debug output, add it to the appropriate diagnostic framework (DSP diagnostics, OpTimingTracker, etc.) rather than one-off prints. Diagnostic code should be toggleable via configuration, not commented-out code.

### Optimization and Crash Handling

When optimizing code or searching for optimal configurations, if you encounter a crash or bug, **dispatch a subagent to fix it** rather than working around it or abandoning the optimization.

## CUDA-Specific Notes

- Heap corruption is often from buffer overruns in native ops, not double-frees. The glibc `(!prev)` message means corrupted malloc metadata from a prior write.
- Views from `.get()` / `.getRow()` on CUDA may have stale device buffers. Use `.dup()` after view operations when the result will be used outside the current SameDiff execution scope.
- `Nd4j.argMax()` has issues with views/non-contiguous arrays. Manual iteration may be needed.
- `setPrimaryBuffer` / `setSpecialBuffer` must keep allocation sizes in sync -- mismatched sizes cause overruns during sync.

## ONNX Import

- ONNX Gather with 2D constant indices `[[0]]` produces higher-rank output than expected. Squeeze single-element constant indices.
- ONNX Softmax opset 13+ defaults axis to -1. The libnd4j softmax op normalizes negative dimensions.
- Mixed-type ops (FLOAT + LONG) silently truncate. Cast explicitly.
- Attention masks must be FLOAT, not LONG, to work with FLOAT attention scores.

## Project Structure

- `libnd4j/` -- C++ native library (CPU and CUDA kernels)
- `nd4j/` -- Java ND4J API, backends, SameDiff
- `nd4j/samediff-import/samediff-import-onnx/` -- ONNX model import
- `deeplearning4j/` -- High-level DL4J layers and model import (Keras etc.)
- `platform-tests/` -- **ALL tests go here** (the only place to run tests)
- `codegen/op-codegen/` -- Op code generation (run `./generate.sh all` after changes)
- `ADRs/` -- Architecture Decision Records
