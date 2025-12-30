# ADR: Op Execution Timing Tracker

## Status
Accepted

## Context

The deeplearning4j/nd4j framework executes thousands of operations during neural network training and inference. Understanding performance characteristics at the operation level is critical for:

1. **Graph-level optimization**: Identifying which operations consume the most time across a computation graph
2. **Within-op analysis**: Understanding where time is spent inside individual operations (validation, memory allocation, helper execution vs native execution)
3. **Platform helper effectiveness**: Measuring how often optimized platform helpers (oneDNN, cuDNN) are used and their performance impact
4. **Regression detection**: Tracking performance over time to detect regressions

### Previous State

The existing profiling infrastructure had several limitations:

1. **Required graph execution context**: Timing only worked when `Environment::getInstance().isProfiling()` was true AND a `FlowPath` with `GraphProfile` existed
2. **No standalone op timing**: Operations executed via the direct `execute(inputs, outputs)` path had no timing
3. **Coarse granularity**: Only captured total execution time, not phase breakdown
4. **No aggregation by op type**: Each node was tracked individually, making it hard to answer "how much time do all matmul ops take?"
5. **No distribution analysis**: Only min/max/average, no percentiles or histograms
6. **No export capabilities**: Data was only available via `printOut()` to stdout

## Decision

Implement a new `OpTimingTracker` system with the following design principles:

### 1. Always-Available Timing

Timing works regardless of execution mode (graph or standalone) with a simple enable check:

```cpp
auto& tracker = OpTimingTracker::getInstance();
if (tracker.isEnabled()) {
    // Record timing
}
```

### 2. Lock-Free Hot Path

Use a fixed-size ring buffer for recording to minimize overhead:

```cpp
static constexpr int RING_SIZE = 8192;
OpTimingRecord _ringBuffer[RING_SIZE];
std::atomic<uint64_t> _ringIndex{0};
```

Recording is a single atomic increment + copy, no locks in the hot path.

### 3. Phase-Level Granularity

Break down op execution into distinct phases:

| Phase | What It Measures |
|-------|------------------|
| `VALIDATION` | Input validation, argument checking, datatype validation |
| `SHAPE_CALC` | Output shape calculation |
| `MEMORY_ALLOC` | Output array allocation |
| `HELPER_CHECK` | Platform helper `isUsable()` check |
| `HELPER_EXEC` | Platform helper execution (oneDNN, cuDNN) |
| `NATIVE_EXEC` | Native C++ implementation execution |
| `TOTAL` | End-to-end operation time |

### 4. Statistical Analysis

Provide rich statistics beyond simple averages:

- **Welford's algorithm** for numerically stable variance calculation
- **Logarithmic histogram** for timing distribution (24 buckets, powers of 2, from <1μs to >4M μs)
- **Percentiles** (p50, p90, p99) computed from histogram
- **Per-thread statistics** for parallel execution analysis

### 5. Export Capabilities

Support multiple export formats:

- **Chrome Trace JSON**: Visualizable in `chrome://tracing` or Perfetto
- **CSV**: For spreadsheet/data analysis tools
- **Console output**: Human-readable hotspot reports

### 6. CUDA Event Timer (Future)

Provide infrastructure for accurate GPU kernel timing:

```cpp
#ifdef SD_CUDA
class CudaEventTimer {
    void* _startEvent;  // cudaEvent_t
    void* _stopEvent;   // cudaEvent_t
    void start();
    void stop();
    float elapsedMillis();
};
#endif
```

## Implementation

### Core Components

```
libnd4j/include/graph/profiling/
├── OpTimingTracker.h      # Header with all structures and classes
└── impl/
    └── OpTimingTracker.cpp # Implementation
```

### Data Structures

```cpp
// Single execution record (fixed-size for ring buffer)
struct OpTimingRecord {
    LongType hash;                       // Op identifier
    const char* name;                    // Op name (static string)
    LongType phaseNanos[OP_PHASE_COUNT]; // Per-phase timing
    LongType inputBytes, outputBytes;    // I/O sizes
    LongType memoryAllocated;            // Memory tracking
    LongType timestampNanos;             // For trace export
    std::thread::id threadId;            // Thread tracking
    bool usedHelper;                     // Helper vs native path
    int deviceId;                        // CUDA device
};

// Aggregated statistics per op type
struct AggregatedOpStats {
    // Counts
    uint64_t callCount, helperUsedCount;

    // Timing (all phases)
    LongType totalTime, validationTime, shapeTime, ...;

    // Distribution
    TimingHistogram histogram;
    double m2, mean;  // For stddev

    // Memory
    LongType totalMemoryAllocated, totalMemoryDeallocated;
};
```

### Instrumentation Point

All timing is captured in `DeclarableOp::execute()`:

```cpp
Status DeclarableOp::execute(Context* block) {
    auto& tracker = OpTimingTracker::getInstance();
    const bool timing = tracker.isEnabled();
    const bool detailed = timing && tracker.isDetailedMode();

    OpTimingRecord rec{};
    if (timing) {
        rec.hash = this->getOpHash();
        rec.name = this->getOpName()->c_str();
        rec.threadId = std::this_thread::get_id();
        rec.timestampNanos = tracker.getTraceTimestamp();
    }

    // Phase: VALIDATION
    { OpPhaseTimer timer(detailed ? &rec : nullptr, OpPhase::VALIDATION);
      validateNonEmptyInput(*block);
      validateArguments(*block);
      validateDataTypes(*block);
    }

    // Phase: MEMORY_ALLOC
    { OpPhaseTimer timer(detailed ? &rec : nullptr, OpPhase::MEMORY_ALLOC);
      prepareOutputs(*block);
    }

    // Phase: HELPER_CHECK + HELPER_EXEC or NATIVE_EXEC
    if (helperAvailable && helper->isUsable(*block)) {
        OpPhaseTimer timer(detailed ? &rec : nullptr, OpPhase::HELPER_EXEC);
        status = helper->invokeHelper(*block);
        rec.usedHelper = true;
    } else {
        OpPhaseTimer timer(detailed ? &rec : nullptr, OpPhase::NATIVE_EXEC);
        status = this->validateAndExecute(*block);
    }

    if (timing) {
        rec.phaseNanos[TOTAL] = currentTimeNanos() - startTime;
        tracker.record(rec);
    }
}
```

### API Surface

#### C++ API

```cpp
#include <graph/profiling/OpTimingTracker.h>

auto& tracker = sd::graph::OpTimingTracker::getInstance();

// Enable timing
tracker.enable(true);           // detailed mode
tracker.enableWithTrace(true);  // with Chrome trace export

// Run operations...

// Analyze
tracker.flush();
tracker.printHotspots(20);
tracker.printOpBreakdown("matmul");
tracker.printHistogram("conv2d");
tracker.printThreadStats();

// Export
tracker.exportChromeTrace("trace.json");
tracker.exportCSV("timing.csv");

// Reset
tracker.reset();
```

#### Java/JNI API

```java
NativeOps ops = Nd4j.getNativeOps();

// Enable
ops.setOpTimingEnabled(1, 1);        // (enabled, detailed)
ops.setOpTimingEnabledWithTrace(1);  // with trace mode

// Run operations...

// Analyze
ops.flushOpTiming();
ops.printOpTimingStats(20);
ops.printOpTimingBreakdown("matmul");
ops.printOpTimingHistogram("conv2d");
ops.printOpTimingThreadStats();

// Export
ops.exportOpTimingChromeTrace("trace.json");
ops.exportOpTimingCSV("timing.csv");

// Query
int numOps = ops.getOpTimingNumOps();
long totalExecs = ops.getOpTimingTotalExecutions();

// Reset
ops.resetOpTiming();
```

### Output Examples

#### Hotspots Report

```
=== Op Timing Hotspots (Top 20) ===
Rank  Op Name                   Calls     Total(ms)     Avg(us)  StdDev(us)   Helper%
----  ------------------------  ----------  ------------  ----------  ----------  --------
   1  conv2d                        847       4523.40     5341.20     1234.50    94.3%
   2  matmul                       2134       2891.10     1354.80      456.20     0.0%
   3  batchnorm                     423        892.30     2109.70      789.10   100.0%
```

#### Phase Breakdown

```
=== conv2d Breakdown (847 calls) ===
Phase            Total(ms)     Avg(us)   % of Op
---------------  ------------  ----------  --------
Validation            12.30       14.52     0.3%
Memory Alloc         156.80      185.12     3.5%
Helper Check          23.10       27.28     0.5%
Helper Exec         4285.90     5061.27    94.7%
Native Exec            0.00        0.00     0.0%
---------------  ------------  ----------  --------
TOTAL               4523.40     5341.20   100.0%

Statistics:
  Helper used: 94.3% of calls
  Min: 2341.50 us, Max: 12453.80 us, StdDev: 1234.50 us
  p50: 4096.00 us, p90: 8192.00 us, p99: 16384.00 us
```

#### Histogram

```
=== conv2d Timing Histogram (847 samples) ===

  Timing Distribution (logarithmic buckets):
  Range (us)            Count     Percent  Histogram
  --------------------  ----------  --------  ----------
  1024 - 2048               45      5.3%  ####
  2048 - 4096              312     36.8%  ##############################
  4096 - 8192              389     45.9%  ########################################
  8192 - 16384              87     10.3%  ########
  16384 - 32768             14      1.7%  #

  Percentiles: p50=4096.0us, p90=8192.0us, p99=16384.0us
```

## Consequences

### Positive

1. **Universal coverage**: All op executions are timed, not just graph-based
2. **Low overhead**: Lock-free ring buffer, cheap enable check
3. **Rich analysis**: Histograms, percentiles, per-thread stats, phase breakdown
4. **Actionable insights**: Easily identify whether slowness is in validation, allocation, or actual compute
5. **Export flexibility**: Chrome trace for visualization, CSV for analysis
6. **Helper visibility**: Track platform helper usage and effectiveness

### Negative

1. **Memory footprint**: Ring buffer (8192 entries * ~128 bytes = ~1MB) always allocated
2. **Timestamp overhead**: `std::chrono::high_resolution_clock` calls in hot path when enabled
3. **Phase timing overhead**: RAII timer objects for each phase when detailed mode enabled
4. **Aggregation cost**: `flush()` iterates entire ring buffer under mutex

### Mitigations

- Ring buffer size is configurable via `RING_SIZE` constant
- Phase timing only active in detailed mode
- Aggregation is on-demand (call `flush()` only when needed)
- Chrome trace events capped at 100,000 to bound memory

### Design Note: Op Name Storage

The `OpTimingRecord` stores the op name as a **fixed-size char array** (`char name[64]`) rather than a `const char*` pointer. This prevents dangling pointer bugs that can occur when:

1. The op's `std::string` name is reallocated during execution
2. The op object is destroyed before timing data is aggregated
3. JNI or other memory is reused in the location of the original string

The 64-character limit is sufficient for all standard op names (e.g., "matmul", "conv2d", "layer_norm").

## Alternatives Considered

### 1. External Profiler Integration (nvprof, VTune, perf)

**Rejected because**: Requires external tools, doesn't capture op-level semantics, harder to integrate with Java layer.

### 2. Sampling-Based Profiler

**Rejected because**: Loses precision for short operations, adds complexity for accurate attribution.

### 3. Extend Existing GraphProfile

**Rejected because**: Too tightly coupled to graph execution, would require significant refactoring to support standalone ops.

## Usage Guide

### Quick Start

#### Java - Basic Profiling

```java
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.nativeblas.NativeOps;

// Get NativeOps instance
NativeOps ops = Nd4j.getNativeOps();

// Enable timing (simple mode - total time only)
ops.setOpTimingEnabled(1, 0);

// Run your neural network or operations
INDArray input = Nd4j.rand(100, 784);
INDArray weights = Nd4j.rand(784, 10);
INDArray result = input.mmul(weights);
// ... more operations ...

// Get results
ops.flushOpTiming();           // Aggregate ring buffer data
ops.printOpTimingStats(10);    // Print top 10 hotspots

// Clean up
ops.resetOpTiming();
ops.setOpTimingEnabled(0, 0);  // Disable
```

#### Java - Detailed Phase Breakdown

```java
// Enable detailed mode (per-phase timing)
ops.setOpTimingEnabled(1, 1);  // second param = 1 for detailed

// Run operations...

ops.flushOpTiming();
ops.printOpTimingStats(20);           // Overall hotspots
ops.printOpTimingBreakdown("mmul");   // Phase breakdown for matrix multiply
ops.printOpTimingBreakdown("conv2d"); // Phase breakdown for convolution
```

#### Java - Chrome Trace Export

```java
// Enable with trace mode for timeline visualization
ops.setOpTimingEnabledWithTrace(1);  // 1 = detailed mode

// Run operations...

ops.flushOpTiming();
ops.exportOpTimingChromeTrace("/tmp/nd4j_trace.json");

// Open chrome://tracing in Chrome browser and load the JSON file
```

#### C++ - Direct Usage

```cpp
#include <graph/profiling/OpTimingTracker.h>

using namespace sd::graph;

// Get singleton instance
auto& tracker = OpTimingTracker::getInstance();

// Enable with detailed phase timing
tracker.enable(true);  // true = detailed mode

// Run operations...

// Analyze results
tracker.flush();
tracker.printHotspots(20);
tracker.printOpBreakdown("matmul");
tracker.printHistogram("conv2d");
tracker.printThreadStats();

// Export for external analysis
tracker.exportChromeTrace("trace.json");
tracker.exportCSV("timing_data.csv");

// Reset for next profiling session
tracker.reset();
```

### Timing Modes

| Mode | Enable Method | Phase Timing | Trace Events | Overhead | Use Case |
|------|---------------|--------------|--------------|----------|----------|
| **Disabled** | `setOpTimingEnabled(0, 0)` | No | No | None | Production |
| **Simple** | `setOpTimingEnabled(1, 0)` | No | No | Low | Quick hotspot identification |
| **Detailed** | `setOpTimingEnabled(1, 1)` | Yes | No | Medium | Deep performance analysis |
| **Trace** | `setOpTimingEnabledWithTrace(1)` | Yes | Yes | Higher | Timeline visualization |

### Understanding the Output

#### Hotspots Report Columns

| Column | Meaning |
|--------|---------|
| `Rank` | Position by total time consumed |
| `Op Name` | Operation name (e.g., "matmul", "conv2d", "add") |
| `Calls` | Number of times this operation was executed |
| `Total(ms)` | Cumulative wall-clock time for all executions |
| `Avg(us)` | Average time per execution in microseconds |
| `StdDev(us)` | Standard deviation - high values indicate variable performance |
| `Helper%` | Percentage of calls that used platform helpers (oneDNN/cuDNN) |

#### Phase Breakdown Interpretation

| Phase | If High, Consider... |
|-------|---------------------|
| `Validation` | Simplifying input shapes, reducing number of inputs |
| `Memory Alloc` | Pre-allocating outputs, using workspaces |
| `Helper Check` | Platform helper overhead (usually negligible) |
| `Helper Exec` | Helper is being used - check if it's faster than native |
| `Native Exec` | The actual computation - expected to be highest |

### Best Practices

#### 1. Ring Buffer Considerations

The tracker uses a fixed-size ring buffer (8192 entries by default). If you execute more ops than this between `flush()` calls, older entries will be overwritten.

```java
// For long-running workloads, flush periodically
for (int epoch = 0; epoch < 100; epoch++) {
    trainOneEpoch();

    if (epoch % 10 == 0) {
        ops.flushOpTiming();  // Aggregate data
        ops.printOpTimingStats(5);
    }
}
```

#### 2. Warm-up Before Profiling

JIT compilation and cache warming can skew initial measurements:

```java
// Warm-up phase (timing disabled or ignored)
for (int i = 0; i < 10; i++) {
    model.fit(warmupData);
}

// Reset and start fresh profiling
ops.resetOpTiming();
ops.setOpTimingEnabled(1, 1);

// Actual profiling
for (int i = 0; i < 100; i++) {
    model.fit(trainingData);
}

ops.flushOpTiming();
ops.printOpTimingStats(20);
```

#### 3. Comparing Helper vs Native Performance

```java
// Profile WITH helpers
Nd4j.getEnvironment().allowHelpers(true);
ops.resetOpTiming();
ops.setOpTimingEnabled(1, 1);
runWorkload();
ops.flushOpTiming();
ops.exportOpTimingCSV("with_helpers.csv");

// Profile WITHOUT helpers
Nd4j.getEnvironment().allowHelpers(false);
ops.resetOpTiming();
runWorkload();
ops.flushOpTiming();
ops.exportOpTimingCSV("without_helpers.csv");

// Compare the CSV files to see helper impact
```

#### 4. Multi-threaded Analysis

```java
// Enable timing
ops.setOpTimingEnabled(1, 1);

// Run parallel workload
ExecutorService executor = Executors.newFixedThreadPool(4);
for (int i = 0; i < 4; i++) {
    executor.submit(() -> runOperations());
}
executor.shutdown();
executor.awaitTermination(1, TimeUnit.HOURS);

// Analyze per-thread performance
ops.flushOpTiming();
ops.printOpTimingThreadStats();  // Shows timing by thread
```

### CSV Export Format

The CSV export contains the following columns for programmatic analysis:

```
OpName,Hash,Calls,TotalMs,AvgUs,StdDevUs,MinUs,MaxUs,HelperPct,
ValidationMs,ShapeCalcMs,MemoryAllocMs,HelperCheckMs,HelperExecMs,NativeExecMs,
TotalInputBytes,TotalOutputBytes,TotalMemoryAllocated,
P50Us,P90Us,P99Us
```

Example Python analysis:

```python
import pandas as pd

df = pd.read_csv('timing.csv')

# Top 10 by total time
print(df.nlargest(10, 'TotalMs')[['OpName', 'Calls', 'TotalMs', 'AvgUs']])

# Operations with high variance (potential optimization targets)
df['CV'] = df['StdDevUs'] / df['AvgUs']  # Coefficient of variation
print(df.nlargest(10, 'CV')[['OpName', 'AvgUs', 'StdDevUs', 'CV']])

# Helper effectiveness
helper_ops = df[df['HelperPct'] > 0]
print(helper_ops[['OpName', 'HelperPct', 'AvgUs']])
```

### Chrome Trace Visualization

1. Export the trace: `ops.exportOpTimingChromeTrace("trace.json")`
2. Open Chrome browser and navigate to `chrome://tracing`
3. Click "Load" and select your `trace.json` file
4. Use WASD keys to navigate, mouse to zoom
5. Click on events to see details

The trace shows:
- Each operation as a horizontal bar
- Duration represented by bar width
- Thread lanes for multi-threaded execution
- Helper vs native execution color-coded

### Troubleshooting

| Issue | Solution |
|-------|----------|
| No output from `printOpTimingStats` | Call `flushOpTiming()` first |
| Missing operations in output | Ring buffer may have overflowed - flush more frequently |
| Phase times don't add up to total | Some overhead not captured in phases is normal |
| Zero helper percentage unexpectedly | Check `Nd4j.getEnvironment().helpersAllowed()` |
| High validation time | May indicate complex input shapes or many inputs |

## Related Decisions

- This complements but does not replace the existing `GraphProfile`/`NodeProfile` system
- Future work may integrate CUDA event timing for accurate GPU kernel measurement
- Memory allocation tracking can be expanded by instrumenting `DataBuffer` creation

## Files Changed

| File | Purpose |
|------|---------|
| `libnd4j/include/graph/profiling/OpTimingTracker.h` | Core header with all structures and classes |
| `libnd4j/include/graph/profiling/impl/OpTimingTracker.cpp` | Full implementation |
| `libnd4j/include/ops/declarable/impl/DeclarableOp.cpp` | Instrumentation in execute() |
| `libnd4j/include/legacy/NativeOps.h` | JNI function declarations |
| `libnd4j/include/legacy/impl/NativeOpsHelpers_OpTiming.cpp` | JNI implementations |
| `nd4j/nd4j-backends/nd4j-api-parent/nd4j-api/src/main/java/org/nd4j/nativeblas/NativeOps.java` | Java interface |

## Date

2025-12-26

## Authors

- Implementation: Claude Code (AI Assistant)
- Architecture: deeplearning4j team
