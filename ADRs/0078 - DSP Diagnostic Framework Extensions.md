# ADR 0078 - DSP Diagnostic Framework Extensions

## Status
Accepted

## Context

The DSP (DynamicShapePlan) execution pipeline has three critical phases that are difficult to debug in production: graph replay (CUDA graph capture, instantiation, and launch), stream synchronization (cross-stream ordering, event waits, stale host/device data), and multi-device orchestration (device selection, P2P transfers, memory pressure rerouting). The existing DSP diagnostic framework provides 14 categories covering compilation, JIT, execution, timing, memory, backend selection, shapes, segments, fusion, verification, KV cache, fallback, device transfers, and emulated replay. However, the three areas above were not explicitly categorized - stream sync events were logged under EXECUTE/FALLBACK, multi-device events under TRANSFER/BACKEND, and graph replay phases under SEGMENT/EXECUTE. This made it impossible to selectively enable diagnostics for a specific concern without also enabling unrelated noise.

Additionally, the `DspDebugger` Java API only supported plan structure analysis (`analyzePlan`) and basic step validation (`validateStep`, `validateMultipleSteps`). There was no programmatic way to:
- Inspect graph replay readiness per segment (capture state, replay count, pointer stability)
- Track phase transitions over multiple execution steps and detect regressions
- Validate stream synchronization correctness via deterministic input patterns
- Analyze multi-device placement and transfer patterns

## Decision

### 1. Three New Diagnostic Categories (C++ + Java)

Added three new bitfield categories to `DspDiagCategory` in `DspDiagnostics.h` and mirrored in `DspDiagnostics.java`:

| Category | Bit | Purpose |
|----------|-----|---------|
| `STREAM_SYNC` | 1<<14 | Stream synchronization events: `cudaStreamSynchronize`, `cudaEventRecord/Wait`, cross-stream ordering dependencies, sync-before-read patterns |
| `MULTI_DEVICE` | 1<<15 | Multi-device orchestration: device selection decisions, P2P topology detection, cross-device migrations, memory pressure rerouting |
| `GRAPH_REPLAY` | 1<<16 | Graph replay lifecycle phases: capture begin/end, instantiation, launch, address snapshot validation, capture buffer identification |

Updated `DSP_DIAG_ALL` from `0x3FFF` (14 categories) to `0x1FFFF` (17 categories) and `DSP_DIAG_NUM_CATEGORIES` from 14 to 17. The existing `parseCategories()` function automatically supports the new names since it iterates the `sCategoryNames[]` array.

The categories are designed to be composable. Enabling `GRAPH_REPLAY` alone gives focused output about capture/replay phases without execution noise. Combining `STREAM_SYNC,GRAPH_REPLAY` isolates sync issues during replay. The `ALL` mask still captures everything.

### 2. Enhanced DspDebugger Java API

Extended `DspDebugger` (attached to a `SameDiff` instance via `DspDebugger.attach(sd)`) with four new analysis methods:

**`analyzeGraphReplay()`** - Queries the native plan handle for per-segment replay state:
- Capture readiness (`isPlanSegmentCapturable`)
- Capture failure detection (`isPlanSegmentCaptureFailed`)
- Replay state codes and counts
- Execution phase per segment (WARMUP, COMPILING, COMPILED, REPLAYING, SLOT_BY_SLOT)
- Capture buffer counts and tracked pointer snapshots
- Plan-level pointer stability and frozen execution count

Returns a `GraphReplayReport` with helper methods: `getCaptureFailures()`, `getStuckSegments()`, `getReplayingSegments()`, `isFullyReplaying()`.

**`trackReplayProgression(numSteps, placeholders, outputs)`** - Executes N steps while recording the `ExecutionPhase` of every segment and the `PlanPhase` at each step. Returns a `GraphReplayProgressReport` with:
- `getPlanPhaseTransitions()` - Steps where the plan phase advanced (e.g., SLOT_BY_SLOT -> SHAPES_FROZEN)
- `getPhaseRegressions()` - Segments that regressed to an earlier phase (indicates instability)

**`validateStreamSync(numSteps, placeholderName, shape, dtype, outputs)`** - Runs deterministic input patterns (constant * step_index) and checks outputs for:
- `NAN_DETECTED` - Partial writes from async kernels visible before completion
- `INF_DETECTED` - Numerical issues or stale data overflow
- `STALE_DATA` - Consecutive steps producing identical output (missing stream sync)
- `OUTPUT_REGRESSION` - Output norm dropping unexpectedly (out-of-order execution)
- `CLOSED_BUFFER` / `NULL_OUTPUT` - Use-after-free or allocation failure

Returns a `StreamSyncReport` with per-issue-type filtering and counting.

**`analyzeMultiDevice()`** - Reports backend distribution across segments, capture buffer counts, INT/LONG sync slot identification (potential cross-device transfer points), and replay cache device statistics.

**`runFullDiagnostics(warmupSteps, validateSteps, placeholders, outputs)`** - Combined sweep that enables ALL categories at FULL level, runs warmup, then collects plan report + replay report + device report + progression report + native C++ plan report + JSON report. Returns a `FullDiagnosticReport` with `getAllIssues()` for a quick summary of everything wrong.

### 3. Diagnostic Category Groupings

The categories are grouped by concern for targeted activation. Each diagnostic mode enables a primary category plus supporting categories that provide necessary context:

| Concern | Primary | Supporting |
|---------|---------|-----------|
| Graph replay | `GRAPH_REPLAY` | `SEGMENT`, `EXECUTE` |
| Stream sync | `STREAM_SYNC` | `EXECUTE`, `TIMING` |
| Multi-device | `MULTI_DEVICE` | `TRANSFER`, `BACKEND`, `MEMORY` |

These groupings are encoded in the benchmark scripts as `--diag-replay`, `--diag-stream`, `--diag-device` flags. The `--diag-all` flag enables all 17 categories at FULL detail level with automatic JSON report generation.

### 4. Integration with Benchmark Infrastructure

Both GPU and CPU benchmark scripts (`run-benchmark.sh`, `run-benchmark-cpu.sh`) accept diagnostic flags that map to Maven `-D` properties, which flow through the surefire `<environmentVariables>` configuration to the forked JVM's `ND4J_DSP_DIAGNOSTICS` environment variable, which the C++ `DspDiagnostics::parseEnvVars()` reads at startup.

The diagnostic output summary in the benchmark results section parses the tee log file for category-specific patterns (capture failures, sync events, device switches) to provide a quick at-a-glance assessment.

## Consequences

- **Selective debugging**: Engineers can now enable `--diag-replay` to focus exclusively on graph capture/replay issues without wading through memory allocation or timing noise. This reduces diagnostic log volume by approximately 10x compared to `--debug` (which enables ALL categories).

- **Phase regression detection**: The `trackReplayProgression()` method can programmatically detect when a segment regresses from REPLAYING to WARMUP, which indicates pointer instability that would cause silent corruption in CUDA graph replay. Previously this required manual log inspection.

- **Stream sync validation**: The deterministic input pattern approach can detect intermittent stream sync bugs that only manifest under specific timing conditions. Running `validateStreamSync(100, ...)` exercises the pipeline enough to surface race conditions.

- **Multi-device readiness**: The `analyzeMultiDevice()` report identifies INT/LONG sync slots (which require host-device synchronization for shape-dependent values) and capture buffer distribution, enabling multi-GPU deployment planning without trial-and-error.

- **No header changes for most C++ code**: The new categories are only used via the existing `DSP_DIAG` macro family. C++ code that wants to emit events under the new categories just uses `DSP_DIAG(STREAM_SYNC, "...")` or `DSP_DIAG(GRAPH_REPLAY, "...")` — no new macros or includes needed.

- **Backward compatible**: The `int` mask type used in the JNI bridge (`dspDiagSetCategories(int)`) supports up to 31 categories. The category name parser in both C++ and Java dynamically iterates the name arrays, so no parsing code changes were needed.

## Files Added/Modified

### Modified Files
- `libnd4j/include/graph/DspDiagnostics.h` - Added 3 category enum values, updated ALL mask and NUM_CATEGORIES
- `libnd4j/include/graph/impl/DspDiagnostics.cpp` - Added 3 category name strings to sCategoryNames[]
- `nd4j/nd4j-backends/nd4j-api-parent/nd4j-api/src/main/java/org/nd4j/autodiff/samediff/diagnostics/DspDiagnostics.java` - Added 3 Java category constants, updated ALL mask and CATEGORY_NAMES[]
- `nd4j/nd4j-backends/nd4j-api-parent/nd4j-api/src/main/java/org/nd4j/autodiff/samediff/execution/DspDebugger.java` - Added analyzeGraphReplay(), trackReplayProgression(), validateStreamSync(), analyzeMultiDevice(), runFullDiagnostics() methods and 10 report/enum inner classes
- `platform-tests/run-benchmark.sh` - Added --diag-replay, --diag-stream, --diag-device, --diag-all, --diag-json flags and diagnostic summary output
- `platform-tests/run-benchmark-cpu.sh` - Same diagnostic flags for CPU backend
- `platform-tests/src/test/java/org/eclipse/deeplearning4j/nd4j/autodiff/samediff/EmulatedReplayTest.java` - Added tests 37-44 exercising new categories and DspDebugger analysis methods
