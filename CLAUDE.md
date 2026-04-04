# CLAUDE.md - Development Guide for Deeplearning4j

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

**ALL tests go in `platform-tests`. ALWAYS run tests from there.**

**ALL test commands MUST be piped through `tee` to a known file.** This is the ONLY reliable way to capture ALL output (Java logs, C++ DSP_DIAG, surefire, everything):
```bash
cd /home/agibsonccc/Documents/GitHub/deeplearning4j/platform-tests && \
  mvn test -Dtest=<TestClass>#<method> 2>&1 | tee /tmp/<descriptive-name>.log
```

**To find test output, read the `tee` file.** Do NOT hunt through surefire report files — they split output across multiple files, may omit stdout/stderr, and are unreliable for C++ diagnostic output. The `tee` file has EVERYTHING in one place.

- **NEVER** run `mvn test` from the project root -- it triggers full rebuilds of native code and runs everything.
- **NEVER** use jemalloc (`LD_PRELOAD=libjemalloc.so`) unless the user explicitly asks for it.
- **NEVER** read surefire report files to find test output — use the `tee` log file instead.
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

### Where Test Output Goes

**USE TEE. ALWAYS USE TEE.** The `tee` log file is the SINGLE SOURCE OF TRUTH for all test output:
```bash
cd /home/agibsonccc/Documents/GitHub/deeplearning4j/platform-tests && \
  mvn test -Dtest=MyTest -Dnd4j.dsp.diagnostics=ALL -Dnd4j.dsp.diagnostics.level=FULL \
  2>&1 | tee /tmp/my-test.log
```
Then read `/tmp/my-test.log`. It contains EVERYTHING: Java logs, C++ DSP_DIAG, surefire summaries, pass/fail, all of it.

**Do NOT read surefire report files** (`target/surefire-reports/*`). They split output across multiple files, may omit stdout/stderr, and are unreliable for C++ diagnostic output. The `tee` file is always complete and reliable.

| Output Type | Location | Notes |
|---|---|---|
| **ALL test output** | **The `tee` log file** | **USE THIS. Java logs, C++ DSP_DIAG, everything.** |
| **Native build log** | `libnd4j/blasbuild/cuda/libnd4j-build.log` (when `-Dlibnd4j.log=libnd4j-build.log` is used) | Separate from Maven output |

**To enable DSP diagnostics**, pass via Maven `-D` properties (NOT shell env vars):
```bash
cd /home/agibsonccc/Documents/GitHub/deeplearning4j/platform-tests && \
  mvn test -Dtest=MyTest -Dnd4j.dsp.diagnostics=ALL -Dnd4j.dsp.diagnostics.level=FULL \
  2>&1 | tee /tmp/my-test.log
```

**CRITICAL RULES:**
- **ALWAYS pipe test commands through `tee`** — this is MANDATORY, not optional.
- **ALWAYS read the `tee` log file** for test output — NEVER hunt through surefire report files.
- **NEVER** use `export ND4J_DSP_DIAGNOSTICS=...` before `mvn test` — surefire replaces env, doesn't merge. Use `-D` properties.
- **NEVER** use `tail` or `grep` on live test output — read the `tee` log file AFTER the test completes.

### Writing Tests

- **Always write standalone isolation tests** when debugging. Reproduce the bug in a minimal test before fixing it.
- **Test all configuration combinations.** Use parameterized/matrix-style tests that enumerate all valid configurations (backends, data types, execution modes, etc.).
- **Make individual configurations runnable.** Structure parameterized tests so a specific broken configuration can be run directly (e.g., via `@MethodSource` with named parameters or `-Dtest=TestClass#method[configName]`).

## Development Rules

### No Workarounds -- EVER

**NEVER** work around a bug. Fix the root cause directly. A workaround is ANY compromise: a shortcut, a guard in the caller, reordering in test code, a "temporary" hack, forcing a particular approach to sidestep a problem, or disabling a feature because it has a bug. If you find an issue while working on something else, dispatch a subagent to fix it. Do not move on with a workaround in place.

**NEVER force a particular approach to avoid fixing a bug.** If graph replay crashes, fix graph replay -- do NOT bypass it and fall back to eager execution. If multi-device transfer fails, fix the transfer -- do NOT hardcode execution to a single GPU. If a kernel produces wrong results on a specific code path, fix the kernel -- do NOT route around that code path. The existing approach was chosen FOR A REASON (performance, correctness, architecture). Abandoning it is a workaround. There are NO compromises.

### Fix ALL Errors -- No Exceptions

**NEVER** dismiss a test error as "pre-existing" or "unrelated." An error is an error. If you encounter ANY failure while running tests -- whether you caused it or not -- **fix it immediately**. Dispatch a subagent if needed. Do NOT report errors back to the user without a fix. The phrase "pre-existing" is BANNED -- it means nobody fixed it yet, and now it's your job.

### NEVER Use EWS (elementWiseStride)

**NEVER** use `ews()` or `elementWiseStride` anywhere in the codebase -- it is **deprecated and unreliable**. EWS values in shape info are invalid for views, non-contiguous arrays, and many common tensor layouts. Code that checks `ews() == 1` as a fast-path condition will silently produce wrong results.

**Instead**, use stride-based contiguity checks:
- `shape::strideDescendingCAscendingF(shapeInfo)` -- checks if strides are contiguous in C or F order
- `ordering() == 'c'` + stride checks -- for C-contiguous verification
- Direct stride inspection via `strideAt(dim)` -- for specific layout requirements

This applies to ALL code: kernels, helpers, loop optimizations, offset calculations, and fast paths. If you see existing code using `ews()`, replace it with the proper stride-based check.

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

**NEVER bypass CUDA graph replay.** Graph replay exists for performance. If replay crashes, produces wrong results, or has capture errors, fix the replay infrastructure -- do NOT disable it, fall back to eager execution, or add a flag to skip it. The same applies to graph capture: if capture fails, fix WHY it fails.

**NEVER hardcode GPU device IDs.** Multi-device execution uses dynamic device selection, memory pressure routing, and peer-access topology for a reason. Do NOT hardcode `device=0`, force all work to one GPU, or skip cross-device transfers to avoid bugs. If a transfer between devices fails, fix the transfer. If memory pressure routing picks the wrong device, fix the routing logic. If peer-access detection is wrong, fix the detection. Hardcoding device IDs is a workaround -- BANNED.

**NEVER simplify multi-device memory transfers.** Cross-device transfers (D2D, H2D staging for non-peer GPUs, P2P direct access) are architected for specific performance and correctness reasons. Do NOT replace D2D with H2D+D2H to avoid a bug. Do NOT skip transfers and duplicate data on each device. Do NOT disable non-peer GPU support because transfers are complex. Fix the transfer code itself.

### DSP Diagnostics

When debugging DSP (DynamicShapePlan) related issues, **always use DSP diagnostics**. Do NOT add ad-hoc printf/logging — use the existing diagnostic infrastructure.

**Header:** `libnd4j/include/graph/DspDiagnostics.h` | **Impl:** `libnd4j/include/graph/impl/DspDiagnostics.cpp`

**How to enable** (via Maven `-D` properties, NEVER shell env vars):
```bash
cd /home/agibsonccc/Documents/GitHub/deeplearning4j/platform-tests && \
  mvn test -Dtest=MyTest \
  -Dnd4j.dsp.diagnostics=ALL \
  -Dnd4j.dsp.diagnostics.level=full
```

**Where to find the output:**
- `platform-tests/target/surefire-reports/<TestClass>-output.txt` — ALL DSP_DIAG output goes here
- JSON report: set `-Dnd4j.dsp.diagnostics.file=/tmp/dsp-report.json` for structured output
- **NEVER** look for DSP_DIAG output in the Maven console — surefire captures it in the report file

**Diagnostic levels** — controls verbosity:
| Level | Value | Behavior |
|---|---|---|
| `summary` | 0 | Category stats only — printed when plan ends or `printPlanReport()` called |
| `detailed` | 1 | Per-step info tracked |
| **`full`** | **2** | **Every event echoed to stdout in real-time** — this is what you want for debugging |

**CRITICAL: If you don't see DSP_DIAG output, the level is probably not `full`.** At `summary` (the default), events go to a ring buffer and are only printed in the plan report at the end. Set `-Dnd4j.dsp.diagnostics.level=full` to get real-time event output.

**Diagnostic categories** — comma-separated, case-insensitive:
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
| `ALL` | All categories enabled |

**Ring buffer:** 65,536 events stored in a pre-allocated ring buffer. At `summary`/`detailed`, events are only in the ring buffer. At `full`, events are ALSO echoed to stdout. The plan report (`printPlanReport()`) dumps stats from the ring buffer regardless of level.

**Maven properties → env vars** (configured in `platform-tests/pom.xml`):
| Maven `-D` property | Env var in forked JVM |
|---|---|
| `nd4j.dsp.diagnostics` | `ND4J_DSP_DIAGNOSTICS` |
| `nd4j.dsp.diagnostics.level` | `ND4J_DSP_DIAGNOSTICS_LEVEL` |
| `nd4j.dsp.diagnostics.file` | `ND4J_DSP_DIAGNOSTICS_FILE` |

**Legacy env vars** (auto-mapped to categories): `ND4J_DSP_TRACE` → EXECUTE, `ND4J_TRITON_VERBOSE` → COMPILE|JIT|BACKEND, `ND4J_DSP_EXECUTION_TIMING` → TIMING, `ND4J_DSP_NATIVE_DUMP_OUTPUTS` → VERIFY.

### Printing Array Values

**Use `array->printIndexedBuffer()` instead of manual loops** when you need to print NDArray values for debugging. This method handles all data types, formatting, and edge cases correctly. Manual `for` loops over buffer elements are error-prone (wrong strides, wrong types, missing sync) and wasteful.

### Git Safety

- **NEVER use `git checkout` on files — BANNED.** Use `git diff` to review changes and `Edit` tool to make targeted modifications. `git checkout` on a file destroys ALL uncommitted changes including the user's own work. There is no undo.
- **NEVER use `git stash` — BANNED.** Stashing silently hides uncommitted changes and risks losing the user's work. If you need to set aside changes, ask the user.
- **NEVER use `git reset --hard` — BANNED.** This destroys uncommitted work irreversibly.
- **NEVER use `git clean` — BANNED.** This deletes untracked files irreversibly.
- **If you need to undo YOUR changes to a file**, use `Edit` to restore the specific lines you changed. Do NOT use git commands that affect the entire file.

### Additional Rules

- **No `.arr` or `.shape` in model import code** -- use `sd.shape(..)` and `sd.rank(..)`. Everything must be variable-based for dynamic shape support.
- **No fully qualified class names in code** -- use imports.
- **Trace values to roots** -- always search for the origin of a value before attempting a fix.
- **`MALLOC_CHECK_=3` does NOT work reliably** -- don't rely on it.
- **Make diagnostics reusable.** When adding diagnostic or debug output, add it to the appropriate diagnostic framework (DSP diagnostics, OpTimingTracker, etc.) rather than one-off prints. Diagnostic code should be toggleable via configuration, not commented-out code.

### Dispatching Subagents

Subagents do NOT automatically inherit knowledge of this CLAUDE.md. When dispatching a subagent, you **MUST** include the following in the prompt:

1. **Explicit rule reminders.** Copy the specific rules that apply to the subagent's task directly into the prompt. Do NOT say "follow CLAUDE.md" — the subagent may not read it. Key rules to always include:
   - **Git Safety:** NEVER use `git checkout`, `git stash`, `git reset --hard`, or `git clean` on files. Use `Edit` tool to make targeted modifications. These git commands destroy uncommitted work irreversibly.
   - **No Workarounds:** Fix root causes directly. NEVER work around a bug.
   - **Build commands:** Include the exact build command if the subagent needs to build. NEVER use `make` directly.
   - **Test location:** ALL tests run from `platform-tests/`. Test output is in `platform-tests/target/surefire-reports/<TestClass>-output.txt`.
   - **No jemalloc:** NEVER use `LD_PRELOAD=libjemalloc.so`.
   - **No `tail`:** NEVER pipe build or test output through `tail`.

2. **Context about what files are modified.** Tell the subagent which files have uncommitted changes so it does not destroy them with git commands.

3. **Scope boundaries.** Tell the subagent exactly what it should and should NOT modify. If it should only investigate, say "DO NOT modify any files — research only."

**Example subagent prompt:**
```
Investigate why X crashes in Y.

RULES (mandatory):
- NEVER use git checkout, git stash, git reset --hard, or git clean — BANNED
- NEVER modify files outside of libnd4j/include/ops/ — research only for other files
- If you need to undo changes, use Edit tool to restore specific lines
- Test output is in platform-tests/target/surefire-reports/<TestClass>-output.txt
- Do NOT use workarounds — fix root causes

Currently modified files (DO NOT git checkout these): <list>
```

**If a subagent violates a rule**, it is YOUR fault for not including the rule in the prompt. Always be explicit.

### Optimization and Crash Handling

When optimizing code or searching for optimal configurations, if you encounter a crash or bug, **dispatch a subagent to fix it** rather than working around it or abandoning the optimization.

## DSP Replay Analytics + Device Stubbing

The DSP replay analytics system bridges DSP replay execution with transfer tracking and device memory management. It enables per-segment transfer profiling during replay and multi-device testing on single-GPU machines.

### Key Classes

| Class | Package | Purpose |
|---|---|---|
| `StubDeviceDescriptor` | `o.n.linalg.api.device` | Fake device with mutable memory, configurable peer topology, per-peer bandwidth |
| `StubDeviceContextProvider` | `o.n.linalg.api.device` | Fake `DeviceContextProvider` with device switch history tracking |
| `DspReplayTransferAnalytics` | `o.n.autodiff.samediff.execution` | Per-step/per-segment transfer recording, memory pressure rerouting |
| `ReplayProfileManager` | `o.n.autodiff.samediff.execution` | Replay profiles enriched with transfer analytics data |
| `DeviceMemoryManager` | `o.n.linalg.api.device` | Singleton with `configureStubTopology()` for test-time multi-device simulation |

### Setting Up a Stub Multi-Device Topology

Use `DeviceMemoryManager.configureStubTopology()` to simulate multiple GPUs with configurable memory and peer access:

```java
StubDeviceDescriptor gpu0 = StubDeviceDescriptor.builder(DeviceType.CUDA_GPU, 0)
    .deviceName("Stub RTX 4090")
    .totalMemory(24L * 1024 * 1024 * 1024)
    .availableMemory(20L * 1024 * 1024 * 1024)
    .addPeerDevice(1)               // NVLink P2P to GPU 1
    .peerBandwidth(1, 300L * 1024 * 1024 * 1024)  // optional: custom bandwidth
    .build();

StubDeviceDescriptor gpu1 = StubDeviceDescriptor.builder(DeviceType.CUDA_GPU, 1)
    .totalMemory(24L * 1024 * 1024 * 1024)
    .addPeerDevice(0)
    .build();

DeviceMemoryManager mgr = DeviceMemoryManager.getInstance();
mgr.configureStubTopology(Arrays.asList(gpu0, gpu1));
// mgr.isMemorySimulationEnabled() == true
// mgr.getStubContextProvider().getDeviceCount() == 2
```

This auto-registers devices, sets simulated memory, injects the stub context provider, and enables simulation mode. **Always call `mgr.clearStubTopology()` in `@AfterEach`** to restore normal state.

### Recording Transfers During Replay Steps

Use `DspReplayTransferAnalytics` to bracket replay steps and record transfers:

```java
TransferSubsystem transferSub = new TransferSubsystem();
transferSub.setEnabled(true);
DspReplayTransferAnalytics analytics = new DspReplayTransferAnalytics(transferSub, memMgr);

analytics.beginStep(segmentIdx, shapeHash);
// ... execute segment ...
analytics.recordTransfer(TransferEvent.builder()
    .variableName("weight_0")
    .direction(TransferDirection.D2D)
    .reason(TransferReason.CAPTURE_BUFFER_COPY)
    .bytes(4096).durationNanos(1000)
    .build());
StepTransferSummary step = analytics.endStep();

// Per-segment accumulation
SegmentTransferSummary seg = analytics.getSegmentSummary(segmentIdx);
// Full report
ReplayTransferReport report = analytics.getReport();
```

### Memory Pressure Rerouting

`checkMemoryPressureForTransfer()` detects pressure on the target device and selects an alternative, recording a `RoutingDecision`:

```java
DeviceDescriptor actual = analytics.checkMemoryPressureForTransfer(
    gpu0, 1L * 1024 * 1024 * 1024, TransferReason.CONSTANT_REPLICATION);
// If gpu0 is under pressure, actual != gpu0 and a RoutingDecision is recorded
```

### Enriching Replay Profiles with Analytics

`ReplayProfileManager.captureProfileWithAnalytics()` merges per-segment transfer stats into the profile. `SegmentReplayInfo` now includes `executionDeviceId`, `transferBytes`, `transferCount`, `transferDurationNanos`, and `transferBytesByReason`. `ReplayProfile` now includes `primaryDeviceId` and `deviceMemoryAtCapture`. All new fields are backward-compatible in JSON (default to 0/null if missing).

### Progressive Memory Exhaustion

`StubDeviceDescriptor.consumeMemory(bytes)` decrements `availableMemory` and returns actual bytes consumed (capped at available). Use this to simulate progressive OOM across replay steps.

### Device Switch History

`StubDeviceContextProvider` records every `switchDevice()` call as a `DeviceSwitchRecord` (previousDeviceId, newDeviceId, caller, reason, timestamp). Inspect with `getSwitchHistory()`, clear with `clearSwitchHistory()`.

### Test Location

`DspReplayDeviceAnalyticsTest` in `platform-tests/.../framework/device/` covers all 10 scenarios: stub topology setup, transfer recording, memory pressure rerouting, P2P vs non-P2P analytics, per-segment breakdown, JSON round-trip, switch history, progressive exhaustion, and reset.

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
