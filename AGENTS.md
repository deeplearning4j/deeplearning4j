# Development Guide for Deeplearning4j

## Banned Actions

These actions are NEVER allowed regardless of context. Violating any of these is a critical error.

### Banned Commands

| Command | Why it's banned |
|---|---|
| `git checkout <file>` | Destroys ALL uncommitted changes including the user's work. No undo. |
| `git stash` | Silently hides uncommitted changes, risks losing work. |
| `git reset --hard` | Destroys uncommitted work irreversibly. |
| `git clean` | Deletes untracked files irreversibly. |
| `make` (direct invocation) | Skips Java binding regeneration. Bindings MUST be rebuilt for any C++ change to be visible from Java. Always use full `mvn` build with both `libnd4j` and the bindings module. |
| `tail` on build/test output | Loses earlier output. Use `tee` to capture the complete log. |
| `LD_PRELOAD=libjemalloc.so` | Crashes on pointers from other allocators. System allocator only. |
| `ccache -C` / `ccache --clear` | Forces a full rebuild of the entire C++ codebase (hours). |
| Changing `-Dlibnd4j.compute=...` | Invalidates the entire ccache, forcing multi-hour full rebuilds. |
| `mvn test` from project root | Triggers full native rebuilds and runs everything. |
| `export VAR=val` before `mvn test` | Surefire forks a new JVM; shell env vars do NOT propagate. Use `-D` Maven properties. |

### Banned Code Patterns

| Pattern | Why | Use Instead |
|---|---|---|
| `ews()` / `elementWiseStride` | Deprecated. Invalid for views, non-contiguous arrays. Silently produces wrong results. | `shape::strideDescendingCAscendingF(shapeInfo)`, `ordering() == 'c'` + stride checks, `strideAt(dim)` |
| `unique_ptr` / `shared_ptr` | Not used in libnd4j codebase | Raw pointers with manual delete |
| Raw `__host__`, `__device__`, `__global__` | Won't compile cross-platform | `SD_HOST`, `SD_DEVICE`, `SD_KERNEL`, `SD_HOST_DEVICE` |
| Raw `__forceinline__` | Platform-specific | `SD_INLINE` |
| Raw `#pragma omp` directives | MSVC compatibility issues | `PRAGMA_OMP_*` macros |
| `.arr` or `.shape` in model import | Must be variable-based for dynamic shapes | `sd.shape(..)`, `sd.rank(..)` |
| Fully qualified class names | Clutters code | Use imports |
| Ad-hoc printf/logging for DSP | Use diagnostic infrastructure | DSP diagnostics framework |
| `MALLOC_CHECK_=3` | Unreliable | Don't rely on it |

### Banned Practices

**No workarounds -- EVER.** Fix root causes directly. A workaround is ANY compromise: a shortcut, a guard in the caller, reordering in test code, a "temporary" hack, forcing a particular approach to sidestep a problem, or disabling a feature because it has a bug.

Specific manifestations that are all workarounds:

- If graph replay crashes, fix graph replay -- do NOT bypass it or fall back to eager execution.
- If multi-device transfer fails, fix the transfer -- do NOT hardcode to a single GPU.
- If a kernel produces wrong results, fix the kernel -- do NOT route around that code path.
- If DSP has a bug, fix it -- do NOT fall back to slot-by-slot execution.
- If a Triton kernel crashes, fix the kernel -- do NOT skip it, stub it out, or route around it.
- If CUDA graph replay fails, fix the replay infrastructure -- do NOT disable it.
- If cross-device transfers fail, fix the transfer code -- do NOT replace D2D with H2D+D2H.
- If device routing picks the wrong device, fix the routing -- do NOT hardcode `device=0`.

**Fix ALL errors -- no exceptions.** Never dismiss a test error as "pre-existing" or "unrelated." The word "pre-existing" is BANNED. If you encounter ANY failure -- whether you caused it or not -- fix it immediately. Dispatch a parallel task if needed.

**If you need to undo YOUR changes to a file**, restore the specific lines you changed by editing them directly. Do NOT use git commands that affect the entire file.

---

## Build Commands

**Ask the user for a build command if one isn't provided.** The user is often working on something specific and the build target varies.

**IMPORTANT:** Only build backend-specific modules. **NEVER** include `platform-tests` in a build `-pl` list -- it is only for running tests, never for building with the project.

### CUDA builds

Require `-Pcuda -Dlibnd4j.chip=cuda`:
```bash
mvn -Pcuda -Dlibnd4j.triton=ON -Dlibnd4j.chip=cuda -Dlibnd4j.buildthreads=12 \
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

### Build Rules

- **Backend selection:** `-Dbackend.artifactId=nd4j-cuda-12.9` or `-Dbackend.artifactId=nd4j-native`.
- Always `install`, never just `compile` -- downstream modules need the jar in the local repo.
- If building C++, always rebuild bindings too (both `libnd4j` AND `:nd4j-cuda-12.9` or `:nd4j-native`).
- Always use `-Dlibnd4j.buildthreads=12` for native builds.
- Always pass `-Dlibnd4j.log=libnd4j-build.log` for native builds (captures C++ log separately).
- **ALL build commands MUST be piped through `tee` to a named file.** Use `mvn ... 2>&1 | tee build-output.log`.
- **Build timeouts:** Header changes trigger full recompiles (30-45 min). Use at least 3600000ms (60 min) timeout. If a build times out, restart the full `mvn` build (not `make`).
- Never deviate from the standard build command. Partial builds (just libnd4j, just bindings) cause stale/mismatched artifacts.

### Build Log Locations

| Log | Location |
|---|---|
| Maven + native output | The `tee` log file |
| C++ build log | `libnd4j/blasbuild/cuda/libnd4j-build.log` (when `-Dlibnd4j.log` is used) |

---

## Testing

**ALL tests go in `platform-tests`. ALWAYS run tests from there.**

### Running Tests

**ALL test commands MUST be piped through `tee`** -- this is the ONLY reliable way to capture ALL output:
```bash
cd /home/agibsonccc/Documents/GitHub/deeplearning4j/platform-tests && \
  mvn test -Dtest=<TestClass>#<method> 2>&1 | tee /tmp/<descriptive-name>.log
```

**Read the `tee` log file for output.** Do NOT read surefire report files (`target/surefire-reports/*`) -- they split output across files, may omit stdout/stderr, and are unreliable for C++ diagnostics. The `tee` file has EVERYTHING in one place.

### Test Runner Wrapper (`platform-tests/bin/java`)

`platform-tests` has a custom `bin/java` wrapper that surefire uses as its JVM. It supports diagnostic tools via `-Dtest.prefix`:

| Prefix | Tool | Purpose |
|---|---|---|
| `valgrind` | Valgrind | Memory debugging with automatic JVM suppressions |
| `/usr/local/cuda/bin/compute-sanitizer` | compute-sanitizer | CUDA memory errors, race conditions, GPU leaks |
| `asan` | AddressSanitizer | Fast (2-3x slowdown) memory error + leak detection |
| `nsys` | Nsight Systems | GPU profiling with CUDA/cuBLAS/cuDNN tracing |
| `nvprof` | nvprof | Legacy NVIDIA profiler |

The wrapper handles suppressions, output files, and JIT disabling automatically.

### Environment Variables and Surefire

Surefire forks a new JVM. Shell environment variables do NOT propagate. To pass configuration:

1. Add the property to `platform-tests/pom.xml` in the surefire `<configuration>` under `<environmentVariables>`.
2. Wire it via a `-D` Maven property so it can be set from the command line.
3. **NEVER** rely on `export VAR=value` before `mvn test`.

### Where Test Output Goes

| Output | Location | Notes |
|---|---|---|
| **ALL test output** | **The `tee` log file** | Java logs, C++ DSP_DIAG, surefire summaries, everything |
| Native build log | `libnd4j/blasbuild/cuda/libnd4j-build.log` | Separate from Maven output |

### Writing Tests

- Write standalone isolation tests when debugging -- reproduce the bug minimally before fixing.
- Test ALL configuration combinations using parameterized/matrix-style tests (backends, data types, execution modes).
- Make individual configurations runnable: `@MethodSource` with named parameters or `-Dtest=TestClass#method[configName]`.
- ALL tests go in `platform-tests/` -- NEVER in the module being tested.

---

## Development Practices

### Investigate Before Coding

**Fully investigate** every task before writing code. Builds take too long to guess. Read the relevant code, trace values to their origins, understand the architecture. Use parallel agents to investigate competing hypotheses simultaneously when dealing with difficult bugs.

### Parallelize Work

When dealing with a difficult bug or complex task, **multi-task aggressively**. Dispatch parallel agents to:
- Investigate competing hypotheses simultaneously
- Fix discovered bugs while you continue the main task
- Run searches across different parts of the codebase

### Optimization and Crash Handling

When optimizing code or searching for optimal configurations, if you encounter a crash or bug, **dispatch a parallel task to fix it** rather than working around it or abandoning the optimization.

---

## C++ Conventions

### Header Discipline

**Avoid modifying headers whenever possible.** Header changes invalidate caches and cause hours of rebuilds. If you can refactor code to keep headers unchanged (move logic to .cpp/.cu files, use forward declarations, etc.), do that. Only modify headers when there is no alternative.

### Kernel and Helper Organization

When adding a new helper or kernel:
1. **Study existing patterns first.** Header in `helpers/`, CPU impl in `helpers/cpu/`, CUDA impl in `helpers/cuda/`.
2. Never add one-off standalone code -- follow established patterns.
3. Never add a kernel for one platform and stub the others -- implement for ALL platforms (CPU and CUDA minimum).
4. Use templates (`<typename T>`) + `BUILD_SINGLE_TEMPLATE` + `BUILD_SINGLE_SELECTOR` -- never hardcode types.
5. Use `getLaunchDims("op_name")` from `LaunchDims.h` -- never hardcode thread counts.
6. Register new ops in `LaunchDims.h` (macros) and `LaunchDims.cu` (both maps).

### Platform Abstraction Macros (MANDATORY)

Always use libnd4j's platform macros instead of raw compiler/platform keywords. These are defined in `libnd4j/include/system/` and ensure code compiles on CPU (GCC, Clang, MSVC) and GPU (NVCC) without `#ifdef` sprawl.

**CUDA function qualifiers** (`system/common.h`):

| Macro | Replaces | Notes |
|---|---|---|
| `SD_HOST` | `__host__` | Empty on CPU builds |
| `SD_DEVICE` | `__device__` | Empty on CPU builds |
| `SD_KERNEL` | `__global__` | Empty on CPU builds |
| `SD_HOST_DEVICE` | `__host__ __device__` | Empty on CPU builds |
| `SD_INLINE` | `__forceinline__` (NVCC) / `__forceinline` (MSVC) / `inline` (GCC/Clang) | Portable forced inline |
| `SD_LIB_EXPORT` | `__declspec(dllexport)` / `visibility("default")` | Public API |
| `SD_LIB_HIDDEN` | hidden visibility | |
| `SD_ALIGN32` | 32-byte alignment attribute | |

**OpenMP pragmas** (`system/openmp_pragmas.h`) -- no-ops on MSVC, fully expand on GCC/Clang:

| Macro | Replaces |
|---|---|
| `PRAGMA_OMP_PARALLEL_FOR` | `#pragma omp parallel for` |
| `PRAGMA_OMP_PARALLEL_FOR_SIMD` | `#pragma omp parallel for simd` |
| `PRAGMA_OMP_PARALLEL_FOR_THREADS(n)` | parallel for with thread count |
| `PRAGMA_OMP_PARALLEL_FOR_COLLAPSE(n)` | parallel for with loop collapsing |
| `PRAGMA_OMP_PARALLEL_FOR_REDUCTION(...)` | parallel for with reduction |
| `PRAGMA_OMP_SIMD` | `#pragma omp simd` |
| `PRAGMA_OMP_ATOMIC` | `#pragma omp atomic` |
| `PRAGMA_OMP_CRITICAL` | `#pragma omp critical` |
| `OMP_IF(args)` / `OMP_SCHEDULE(args)` | conditional / schedule wrappers |
| `PRAGMA_OMP_DECLARE_SIMD_SAFE` | SIMD declaration with unsupported-type warning suppression |

**Template instantiation** (`system/type_boilerplate.h`) -- never manually instantiate templates for each type:

| Macro | Purpose |
|---|---|
| `BUILD_SINGLE_TEMPLATE(NAME, SIG, TYPES)` | Instantiate a template for all types in a type list |
| `BUILD_SINGLE_SELECTOR(XTYPE, NAME, SIG, TYPES)` | Runtime type dispatch (switch on DataType) |
| `BUILD_DOUBLE_SELECTOR(XTYPE, YTYPE, ...)` | Two-type dispatch |
| `BUILD_TRIPLE_SELECTOR(XTYPE, YTYPE, ZTYPE, ...)` | Three-type dispatch |

**Other macros:**

| Macro | Source | Purpose |
|---|---|---|
| `SD_PROMOTE_FUNC(NAME, BODY)` | `math/templatemath.h` | Binary math with automatic type promotion |
| `DECLARE_PLATFORM(NAME, ENGINE)` | `system/platform_boilerplate.h` | Declare a platform-specific op |
| `PLATFORM_IMPL(NAME, ENGINE)` | `system/platform_boilerplate.h` | Implement the op |
| `PLATFORM_CHECK(NAME, ENGINE)` | `system/platform_boilerplate.h` | Check if platform can run the op |

### Configuration and Environment

When adding any new configuration option:
1. Add it to `libnd4j/include/system/Environment.h` and `Environment.cpp` (C++ side).
2. Add it to `nd4j/nd4j-backends/nd4j-api-parent/nd4j-api/src/main/java/org/nd4j/linalg/factory/Environment.java` (Java interface).
3. Add the system property constant to `ND4JSystemProperties.java`.
4. Wire it through the preset if it needs JNI exposure.

### Generated Code

**NEVER** edit generated code directly. Update the **presets** instead:
- `nd4j-native-preset`, `nd4j-cuda-preset`, `nd4j-minimizer-preset`, `nd4j-tpu-preset`
- Headers are automatically parsed by JavaCPP. Most of the time, modifying a header is sufficient -- the presets pick it up.

### Debugging

- Use `array->printIndexedBuffer()` for printing NDArray values -- never manual loops (wrong strides, wrong types, missing sync).
- Gate diagnostics behind `isVerbose`/`isDebug` -- no unconditional `syncToHost`.
- Make diagnostics reusable: add to DSP diagnostics or OpTimingTracker, not one-off prints. Diagnostic code should be toggleable via configuration, not commented-out code.

---

## DSP Development Rules

### Core Principles

**Maximize configuration optionality.** The goal is to blend different execution configurations (graph replay, slot-based, Triton-compiled, cuBLAS fallback) for optimal performance. Skipping kernels or falling back to slot-by-slot destroys this optionality. Every execution path must work correctly so configurations can be mixed freely.

**Fix bugs encountered during profiling.** When profiling and you encounter a crash, wrong result, or other bug, fix it immediately (dispatch a parallel task if needed). Profiling is not an excuse to ignore correctness.

### DSP Diagnostics

When debugging DSP (DynamicShapePlan) related issues, **always use DSP diagnostics**. Do NOT add ad-hoc printf/logging.

**Header:** `libnd4j/include/graph/DspDiagnostics.h` | **Impl:** `libnd4j/include/graph/impl/DspDiagnostics.cpp`

**How to enable** (via Maven `-D` properties, NEVER shell env vars):
```bash
cd /home/agibsonccc/Documents/GitHub/deeplearning4j/platform-tests && \
  mvn test -Dtest=MyTest \
  -Dnd4j.dsp.diagnostics=ALL \
  -Dnd4j.dsp.diagnostics.level=full \
  -Dnd4j.dsp.diagnostics.file=/tmp/dsp-report.json \
  2>&1 | tee /tmp/my-test.log
```

**CRITICAL: If you don't see DSP_DIAG output, the level is probably not `full`.** At `summary` (the default), events go to a ring buffer and are only printed in the plan report at the end. Set `full` for real-time event output.

**Diagnostic levels:**

| Level | Value | Behavior |
|---|---|---|
| `summary` | 0 | Category stats only -- printed at plan end or `printPlanReport()` |
| `detailed` | 1 | Per-step info tracked |
| **`full`** | **2** | **Every event echoed to stdout in real-time -- use this for debugging** |

**Diagnostic categories** (comma-separated, case-insensitive):

| Category | What it traces |
|---|---|
| `COMPILE` | Backend compilation (Triton, MLIR) |
| `JIT` | Kernel generation, PTX/cubin, cache hits/misses |
| `EXECUTE` | Per-step execution flow, segment dispatch |
| `TIMING` | Detailed timing breakdowns |
| `MEMORY` | Allocations, OOM, failover, pool state |
| `BACKEND` | Backend selection, device placement |
| `SHAPE` | Shape analysis, static/dynamic, frozen detection |
| `SEGMENT` | Segment building, boundaries, capturable analysis |
| `FUSION` | Op fusion, identity elimination |
| `VERIFY` | Golden comparison, output validation |
| `KV_CACHE` | KV cache config, retention, scattering |
| `FALLBACK` | Fallback events, error recovery |
| `STREAM_SYNC` | Stream ordering, event waits, sync points |
| `MULTI_DEVICE` | Device selection, P2P, migrations |
| `GRAPH_REPLAY` | Capture/instantiate/launch/address validation |
| `ALL` | All categories enabled |

**Maven properties -> env vars** (configured in `platform-tests/pom.xml`):

| Maven `-D` property | Env var in forked JVM |
|---|---|
| `nd4j.dsp.diagnostics` | `ND4J_DSP_DIAGNOSTICS` |
| `nd4j.dsp.diagnostics.level` | `ND4J_DSP_DIAGNOSTICS_LEVEL` |
| `nd4j.dsp.diagnostics.file` | `ND4J_DSP_DIAGNOSTICS_FILE` |

**Legacy env vars** (auto-mapped): `ND4J_DSP_TRACE` -> EXECUTE, `ND4J_TRITON_VERBOSE` -> COMPILE|JIT|BACKEND, `ND4J_DSP_EXECUTION_TIMING` -> TIMING, `ND4J_DSP_NATIVE_DUMP_OUTPUTS` -> VERIFY.

Ring buffer: 65,536 events, pre-allocated. `printPlanReport()` dumps stats regardless of level.

### DSP System Properties

| Property | Purpose |
|---|---|
| `nd4j.dsp.graphExecutionMode` | `AUTO`, `SLOT_BY_SLOT`, `CUDA_GRAPHS`, `TRITON` |
| `nd4j.dsp.cudaGraphs.enabled` | CUDA graph capture/replay |
| `nd4j.dsp.nativeExecutor.enabled` | Native plan execution |
| `nd4j.dsp.noFreeze` | Disable shape freezing |
| `nd4j.dsp.freezeRecompile` | Recompile on freeze |
| `nd4j.dsp.freezeMergeSegments` | Merge segments on freeze |
| `nd4j.dsp.batchZero` | Batch zero optimization |
| `nd4j.dsp.matmulSegmentation` | MatMul segmentation |
| `nd4j.dsp.castElimination` | Cast elimination |
| `nd4j.dsp.fp16Compute` | FP16 compute path |
| `nd4j.dsp.trace` | Execution trace (-> EXECUTE category) |
| `nd4j.dsp.executionTiming` | Timing (-> TIMING category) |
| `nd4j.op.timing` | Op timing |
| `nd4j.optimizer.enabled` | GraphOptimizer |
| `nd4j.optimizer.fp16` | FP16 weight pre-cast |
| `nd4j.triton.sectionFusion` | Triton section fusion |

---

## Architecture Reference

### Project Structure

```
libnd4j/              -- C++ native library (CPU and CUDA kernels)
  include/ops/        -- Op implementations (declarable/, helpers/, helpers/cpu/, helpers/cuda/)
  include/graph/      -- Graph execution engine, DSP
  include/system/     -- Platform macros, Environment
  include/loops/      -- Kernel loops
  include/array/      -- NDArray implementation
nd4j/                 -- Java ND4J API, backends, SameDiff
  samediff-llm/       -- LLM/VLM benchmark + generation infrastructure
  samediff-import/    -- ONNX model import (Kotlin)
  nd4j-ggml/          -- GGML/GGUF model import + quantization
deeplearning4j/       -- High-level DL4J layers and model import (Keras etc.)
platform-tests/       -- ALL tests go here (the only place to run tests)
codegen/op-codegen/   -- Op code generation (run `./generate.sh all` after changes)
ADRs/                 -- Architecture Decision Records
```

### ADRs (Architecture Decision Records)

New features and significant changes require ADR checks:
1. Read existing ADRs in `ADRs/` to understand the format (Status, Context, Decision sections).
2. Create a new ADR for new features or architectural decisions.
3. Update existing ADRs when modifying behavior covered by a prior decision.

### DSP Architecture

- **Plan cache**: Shape-keyed. One plan per (outputs, placeholder shape-info ptrs). `computeShapeKey()` gates value hashing on `outputShapeDependsOnInputValues`. Eviction must skip pinned plans (pin/unpin).
- **Execution flow**: `DynamicShapePlanCompiler.compile()` -> DAG -> classifies ops via JNI `getOpTraits()` (C++ `OpTraitTable.cpp`).
- **Executor lifecycle**: warmup -> freezeShapes -> pointer stability -> CUDA graph capture -> replay.
- **argTableStable**: When true, skip refresh + ext input sync (fast replay path).
- **Stream management**: `tl_dspExecutionStream` routes H2D to DSP stream (no per-call cudaStreamSync). `tl_dspGapStream` unifies gap ops onto same stream as island replay.
- **Triton dispatch**: `OpTraitTable.cpp` is the single source of truth for which ops can be Triton-compiled.

### Key DSP Classes

| Class | Location | Purpose |
|---|---|---|
| `DynamicShapePlan` | nd4j-api/.../execution/ | Plan representation |
| `DynamicShapePlanCompiler` | nd4j-api/.../execution/ | Compiles SameDiff -> plan |
| `DynamicShapePlanExecutor` | nd4j-api/.../execution/ | Executes plans (lifecycle manager) |
| `DspDiagnostics` | libnd4j/include/graph/ | C++ diagnostic framework |
| `DspPlanAssertions` | nd4j-api/.../execution/ | Test assertions (POINTERS_STABLE, REPLAYING) |
| `GraphOptimizer` | nd4j-api/.../optimize/ | Fusion/optimization entry point |
| `OpTraitTable.cpp` | libnd4j/include/ops/ | SSOT for Triton op mappability |
| `NativeDynamicShapePlan.cpp` | libnd4j/include/graph/impl/ | Native C++ plan execution |
| `GraphExecutionMode` | nd4j-api/.../execution/ | AUTO, SLOT_BY_SLOT, CUDA_GRAPHS, TRITON |

### DSP Replay Analytics + Device Stubbing

For multi-device testing on single-GPU machines, use `DeviceMemoryManager.configureStubTopology()` with `StubDeviceDescriptor` builders.

| Class | Package | Purpose |
|---|---|---|
| `StubDeviceDescriptor` | `o.n.linalg.api.device` | Fake device with mutable memory, configurable peer topology, per-peer bandwidth |
| `StubDeviceContextProvider` | `o.n.linalg.api.device` | Fake context provider with device switch history tracking |
| `DspReplayTransferAnalytics` | `o.n.autodiff.samediff.execution` | Per-step/segment transfer recording, memory pressure rerouting |
| `ReplayProfileManager` | `o.n.autodiff.samediff.execution` | Replay profiles enriched with transfer analytics |
| `DeviceMemoryManager` | `o.n.linalg.api.device` | Singleton with `configureStubTopology()` for test-time multi-device simulation |

**Usage pattern:**
```java
StubDeviceDescriptor gpu0 = StubDeviceDescriptor.builder(DeviceType.CUDA_GPU, 0)
    .deviceName("Stub RTX 4090")
    .totalMemory(24L * 1024 * 1024 * 1024)
    .availableMemory(20L * 1024 * 1024 * 1024)
    .addPeerDevice(1)
    .peerBandwidth(1, 300L * 1024 * 1024 * 1024)
    .build();

DeviceMemoryManager mgr = DeviceMemoryManager.getInstance();
mgr.configureStubTopology(Arrays.asList(gpu0, gpu1));
```

Always call `mgr.clearStubTopology()` in `@AfterEach`. Transfer recording uses `DspReplayTransferAnalytics.beginStep()`/`endStep()` brackets. Memory pressure rerouting via `checkMemoryPressureForTransfer()`. Progressive OOM simulation via `StubDeviceDescriptor.consumeMemory()`. Device switch tracking via `StubDeviceContextProvider.getSwitchHistory()`.

Tests: `DspReplayDeviceAnalyticsTest` in `platform-tests/.../framework/device/`.

### CUDA-Specific Notes

- Heap corruption is often from buffer overruns in native ops, not double-frees. The glibc `(!prev)` message means corrupted malloc metadata from a prior write.
- Views from `.get()` / `.getRow()` on CUDA may have stale device buffers. Use `.dup()` after view operations when the result will be used outside the current SameDiff execution scope.
- `Nd4j.argMax()` has issues with views/non-contiguous arrays. Manual iteration may be needed.
- `setPrimaryBuffer` / `setSpecialBuffer` must keep allocation sizes in sync -- mismatched sizes cause overruns during sync.
- `p()` method: writes host then syncToDevice (hidden H2D). Don't bypass without understanding this.
- `toFloatVector()` on CUDA views is extremely slow -- use `dup().data().asFloat()`.

### ONNX Import Notes

- ONNX Gather with 2D constant indices `[[0]]` produces higher-rank output than expected. Squeeze single-element constant indices.
- ONNX Softmax opset 13+ defaults axis to -1. The libnd4j softmax op normalizes negative dimensions.
- Mixed-type ops (FLOAT + LONG) silently truncate. Cast explicitly.
- Attention masks must be FLOAT, not LONG, to work with FLOAT attention scores.

### Double-Free and Shutdown Crashes

When debugging double-frees, use-after-free, or shutdown crashes:
- Always check the `DeallocatorService` for proper shutdown flags.
- Verify that deallocation ordering respects object lifetimes.
- Check that `setCloseable(false)` / `setConstant(true)` poisoning is properly undone.

---

## Error Diagnosis Strategy

### C++ Compile Errors
1. Read the error from the tee log -- find the FIRST error (ignore cascading ones).
2. Read the source file at the error line.
3. Understand the context -- read surrounding code, check includes.
4. Fix the root cause -- not a workaround.
5. If it's a header error, check if the fix can go in a .cpp/.cu instead (avoid cache invalidation).

### Java Compile Errors
1. Read the Maven output from the tee log.
2. Check for missing imports, type mismatches, API changes.
3. If an API changed in a dependency, grep for the new API signature.
4. Fix and rebuild.

### Linker Errors
1. Missing symbols -- usually a .cpp/.cu file not included in CMake.
2. Duplicate symbols -- usually a header with non-inline function definitions.
3. Check CMakeLists.txt if source files were added/removed.

### CMake Errors
1. Read the CMake output section of the build log.
2. Check CMakeLists.txt for syntax errors or missing dependencies.
3. Do NOT modify CMake configuration casually -- understand the build system first.

### Test Failures
1. Read the full output from the tee log (NOT surefire reports).
2. Assertion failure: trace expected vs actual back to origin.
3. Runtime crash: read stack trace, check for buffer overruns, null pointers.
4. DSP failure: enable diagnostics, check phase progression.
5. Timeout: check for infinite loops, deadlocks.
6. Fix production code (not test assertions, unless the test is genuinely wrong).

---

## Benchmark Scripts

All scripts live in `platform-tests/`:

### VLM Decode Benchmark (`run-benchmark.sh`)

Primary benchmark for SmolDocling VLM decode throughput.

```bash
cd /home/agibsonccc/Documents/GitHub/deeplearning4j/platform-tests
./run-benchmark.sh [OPTIONS]
```

| Flag | Purpose |
|---|---|
| `--tokens N` | Decode tokens (ALWAYS 250 for perf, fewer ONLY for debugging) |
| `--config NAME` | Config name (default: OPTIMAL) |
| `--op-timing` | Enable native op timing CSV export |
| `--op-timing-detailed` | Per-phase timing breakdown |
| `--op-breakdown OPS` | Per-op timing for specific comma-separated ops |
| `--op-histogram OPS` | Per-op timing histograms |
| `--fp16` / `--no-fp16` | FP16 weight pre-casting (default: ON) |
| `--no-optimizer` | Disable GraphOptimizer |
| `--triton-tf32` | Enable TF32 for Triton DotOps |
| `--debug` | Full DSP diagnostics + CUDA driver log |
| `--diag-replay` / `--diag-stream` / `--diag-device` / `--diag-all` | Targeted diagnostics |
| `--diag-json FILE` | JSON diagnostic report |
| `--nsys` | Nsight Systems profiling |
| `--clear-cache` / `--clear-decoder` / `--no-clear-decoder` | Cache management |
| `--backend cuda\|cpu` | Backend selection |

### LLM Multi-Model Benchmark (`run-llm-benchmarks.sh`)

Runs across model families: qwen (0.8B), gemma (1B), phi, mistral, lfm2-extract (350M).

| Flag | Purpose |
|---|---|
| `--test TEST` | import, baseline, cuda-graphs, triton, fusion, optimizer, matrix, perplexity, quant, prompts, device, all |
| `--models MODELS` | Comma-separated: qwen, gemma, phi, mistral, lfm2-extract, all |
| `--tokens N` | Decode tokens (default: 20) |
| `--backend cuda\|cpu` | Backend |
| `--config CONFIGS` | Config filter (supports * wildcard) |
| `--quant TYPE` | Quantization type (default: Q4_K_M) |
| `--op-timing` | Native op timing |
| `--debug` | DSP diagnostics at FULL level |

### CPU Benchmark (`run-benchmark-cpu.sh`)
Wrapper for `run-benchmark.sh --backend cpu`.

### Performance Analysis Workflow

1. **Baseline**: `./run-benchmark.sh --tokens 250` -> note `lateSteady tok/s`
2. **Hotspot identification**: add `--op-timing`
3. **Drill into specific ops**: add `--op-breakdown matmul,softmax`
4. **Compare configs**: different `--config` values (SLOT_BY_SLOT, OPTIMAL, TRITON, CUDA_GRAPHS)
5. **Profile sync overhead**: `--diag-stream`
6. **Profile graph replay**: `--diag-replay`
7. **Full diagnostic dump**: `--diag-all --diag-json /tmp/perf-diag.json`
8. **Nsight profiling**: `--nsys`

### Key Metrics

| Metric | Description |
|---|---|
| `overall tok/s` | End-to-end throughput |
| `decode tok/s` | Decode-phase only |
| `steady tok/s` | Excludes warmup steps |
| `lateSteady tok/s` | Most stable measurement |

---

## Validation and Regression Detection

### DSP Accuracy Validation (`run-validation.sh`)

Compares execution modes for token-level correctness.

| Flag | Purpose |
|---|---|
| `--test NAME` | outputAccuracy, perOpSlot, decodeStep, tf32Isolation, ALL |
| `--tokens N` | Max decode tokens per test |
| `--configs LIST` | Comma-separated configs for outputAccuracy |
| `--tolerance NAME` | Preset: standard, strict, tf32 |
| `--match-rate N` | Minimum token match rate % (default: 90) |
| `--verbose` | Per-step token logging |

### DSP Configuration Matrix (`run-dsp-matrix.sh`)

Sweeps 8 configs against golden SLOT_BY_SLOT baseline:

| Config | What it tests |
|---|---|
| `SLOT_BY_SLOT_baseline` | Baseline correctness |
| `SLOT_BY_SLOT_batchedGemm` | Batched GEMM integration |
| `AUTO_defaults` | AUTO resolution logic |
| `AUTO_frozen` | Frozen constants with AUTO |
| `TRITON_sectionFusion` | Triton section fusion pipeline |
| `TRITON_compileAll` | Triton compile-all mode |
| `TRITON_frozen_batchedGemm` | Full Triton + frozen + batched GEMM |
| `CUDA_GRAPHS_frozen` | CUDA graph capture + replay |

| Flag | Purpose |
|---|---|
| `--config NAME` | Run single config |
| `--list` | Print available configs |
| `--cpu` | CPU backend |
| `--no-triton` | Skip Triton kernels |
| `--diag-replay` / `--diag-segment` / `--diag-phase` / `--diag-all` | Diagnostics |
| `--diag-json FILE` | JSON diagnostic report |

### Domain Test Suites (in `platform-tests/`)

| Script | Scope |
|---|---|
| `run-all-tests.sh` | Everything |
| `run-nd4j-tests.sh` | ND4J core ops |
| `run-samediff-tests.sh` | SameDiff/autodiff |
| `run-vlm-tests.sh` | VLM (SmolDocling) |
| `run-llm-tests.sh` | LLM generation |
| `run-ggml-tests.sh` | GGML import |
| `run-onnx-tests.sh` | ONNX import |
| `run-validation.sh` | DSP accuracy validation |
| `run-dsp-matrix.sh` | DSP 8-config matrix |

### Key Regression Test Classes

| Class | Tests |
|---|---|
| `TestDspValidation` | outputAccuracy, perOpSlot, decodeStep, tf32Isolation |
| `TestDspConfigurationMatrix` | 8-entry config matrix sweep |
| `DspLifecycleValidationTest` | DSP lifecycle phase progression |
| `DspSlotLifecycleAuditTest` | Slot lifecycle audit |
| `TestDspPipelineFacets` | Pipeline facet integration |
| `TestNativeDecodeLoopRegression` | Native decode loop regression |
| `TestMythicPdfRegression` | Mythic PDF regression |
| `DspPlanAssertions` | Shared assertion helper (phases: POINTERS_STABLE, REPLAYING) |

---

## Kompile Agent Dispatch

When dispatching tasks to kompile agents, ALWAYS assign a DL4J role. Agents start blank -- the role injects DL4J rules and knowledge into their system prompt.

### Available DL4J Roles

| Role | When to use |
|---|---|
| `dl4j-fixer` | **DEFAULT for fixes.** Autonomous build->test->fix loop. Will NOT stop to ask. |
| `dl4j-dev` | General development -- features, refactoring, code changes |
| `dl4j-investigator` | Research only -- traces code, finds root causes, does NOT modify files |
| `dl4j-benchmarker` | Performance work -- runs benchmarks, analyzes tok/s, profiles hotspots |
| `dl4j-reviewer` | Code review -- checks for rule violations, safety issues, perf problems |

### Dispatch Pattern (Single Fix Task)

```
mcp__kompile__task:
  description: "Fix the regression in X"
  prompt: "Fix the regression...

Currently modified files (DO NOT touch): <list from git status>
Scope: only modify files in <path>

Build: /home/agibsonccc/dev-apps/mvn/bin/mvn -Pcuda -Dlibnd4j.triton=ON ...
Test: cd platform-tests && mvn test -Dtest=TestClass 2>&1 | tee /tmp/test.log

Success: TestClass all pass."
  agent: "qwen"
  role: "dl4j-fixer"
```

### Before Dispatching

1. Run `git status` to get modified files
2. Include modified files in the prompt so agents don't destroy them
3. Specify scope boundaries (what can/cannot be modified)
4. Include build and test commands if needed
5. Define success criteria

### Dispatch Tools

| Tool | Purpose |
|---|---|
| `mcp__kompile__task` | Single async task |
| `mcp__kompile__multi_task` | Parallel tasks (different prompts) |
| `mcp__kompile__quorum_task` | Consensus from multiple agents |

---

## Dispatching Subagents (Claude Code)

Subagents do NOT automatically inherit knowledge of AGENTS.md. When dispatching a subagent, you **MUST** include:

1. **Explicit rule reminders.** Copy the specific rules that apply. Do NOT say "follow AGENTS.md." Key rules to always include:
   - Git Safety: NEVER use `git checkout`, `git stash`, `git reset --hard`, or `git clean`
   - No Workarounds: Fix root causes directly
   - Build commands: Include the exact build command. NEVER use `make` directly
   - Test location: ALL tests from `platform-tests/`. Output in the `tee` log file
   - No jemalloc: NEVER use `LD_PRELOAD=libjemalloc.so`
   - No `tail`: NEVER pipe build or test output through `tail`

2. **Context about modified files.** Tell the subagent which files have uncommitted changes.

3. **Scope boundaries.** Tell the subagent exactly what it should and should NOT modify.

**If a subagent violates a rule**, it is YOUR fault for not including the rule in the prompt.
