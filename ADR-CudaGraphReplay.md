# ADR: CUDA Graph Replay and Visualization

## Status
Accepted

## Date
2026-02-18

## Context

DeepLearning4J's SameDiff execution engine processes computation graphs through a native `NativeDynamicShapePlan` executor. For inference workloads like LLM token generation, the same graph is executed thousands of times with identical structure but varying input data (e.g., growing KV cache sequences).

Each op execution incurs overhead from:
1. **CPU-side launch overhead**: Java→JNI→C++ traversal, context setup
2. **Kernel launch latency**: Each CUDA kernel requires a separate launch call (~5-10μs each)
3. **Memory operations**: Individual allocations and transfers for each op

For autoregressive LLM generation with hundreds of ops per forward pass, this overhead becomes significant. PyTorch addresses this with CUDA Graphs support via `torch.cuda.CUDAGraph` and `torch.cuda.graph()`, providing:
- Single API call to replay entire captured kernel sequences
- Reduced CPU overhead by bypassing per-kernel launches
- Improved GPU utilization through optimized scheduling

### Requirements

1. **Transparent Integration**: Work with existing SameDiff execution without API changes
2. **Shape Dynamic Handling**: Support graphs with dynamic shapes (e.g., growing KV cache)
3. **Segmentation**: Handle non-capturable ops (shape-dependent, host callbacks) by segmenting the graph
4. **Debugging Support**: Provide visibility into capture status, node contributions, and replay statistics
5. **Visualization**: PyTorch-style Chrome trace export and HTML visualization

## Decision

Implement CUDA Graph capture and replay for the native execution plan with comprehensive visualization support, following PyTorch's API patterns.

### Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                     SameDiff Execution Flow                         │
├─────────────────────────────────────────────────────────────────────┤
│  1. Graph Compilation (Java)                                        │
│     SameDiff → DynamicShapePlan → serialized bytes → JNI           │
│                                                                     │
│  2. Native Plan Creation (C++)                                      │
│     NativeDynamicShapePlan::fromSerializedPlan()                   │
│     └── buildSegments() - partition into capturable/non-capturable │
│                                                                     │
│  3. Execution Loop                                                  │
│     ┌──────────────────────────────────────────────────────────┐   │
│     │  For each segment:                                       │   │
│     │    ├─ WARMUP (1st exec): slot-by-slot, populate cache   │   │
│     │    ├─ CAPTURE (2nd exec): cudaStreamBeginCapture()      │   │
│     │    │                      executeSlot() × N             │   │
│     │    │                      cudaStreamEndCapture()        │   │
│     │    │                      cudaGraphInstantiate()        │   │
│     │    └─ REPLAY (3rd+): cudaGraphLaunch() on cached graph  │   │
│     └──────────────────────────────────────────────────────────┘   │
│                                                                     │
│  4. Visualization Export                                            │
│     └── Chrome Trace / HTML / DOT / JSON                           │
└─────────────────────────────────────────────────────────────────────┘
```

### Core Components

#### 1. Graph Segmentation (`NativeDynamicShapePlan`)

The plan is partitioned into segments based on capturability:

```cpp
struct GraphSegment {
    int startSlot, endSlot;
    bool isCapturable;           // No shape-dependent ops, no host callbacks
    bool captureFailed;          // Permanent failure (e.g., unsupported op)
    int captureOomRetries;       // OOM retry counter
    int captureRetryAfterExec;   // Execution threshold for OOM retry
    
    std::shared_ptr<sd::cuda::CudaGraphHandle> cachedGraph;
    LongType cachedShapeKey;     // Hash of input shapes for invalidation
    
    std::vector<CaptureBuffer> captureBuffers;  // Fixed-address input buffers
    std::unordered_map<int, size_t> capturedOutputSizes;  // For KV cache validation
    
    int64_t executionCount;
};
```

**Segmentation Rules:**
- Segment boundaries at non-capturable ops (shape_of with host sync, dynamic reshapes)
- Minimum segment size (default: 10 ops) to amortize capture overhead
- Maximum segment size (default: 50 ops) to limit capture memory

#### 2. CUDA Graph Handle (`CudaGraphHandle`)

Wraps `cudaGraph_t` and `cudaGraphExec_t` with lifecycle management:

```cpp
class CudaGraphHandle {
    cudaGraph_t _graph;
    cudaGraphExec_t _graphExec;
    GraphState _state;  // EMPTY → CAPTURING → CAPTURED → INSTANTIATED
    
    // Visualization data
    std::vector<ExecutionTimelineEntry> _executionTimeline;
    double _captureStartTimeMs, _captureEndTimeMs, _instantiateTimeMs;
    
    // Methods
    bool beginCapture(cudaStream_t stream, cudaStreamCaptureMode mode);
    bool endCapture(cudaStream_t stream);
    bool instantiate();
    bool launchAsync(cudaStream_t stream);
    
    // Visualization
    bool exportToChromeTrace(const std::string& filename) const;
    bool exportToHtml(const std::string& filename) const;
    bool debugDump(const std::string& basePath) const;
    std::string getChromeTraceJson() const;
};
```

#### 3. Capture Buffer Management

CUDA graphs record exact GPU memory addresses. External inputs (position_ids, attention_mask) are recreated each step with new addresses. Solution: fixed-address "capture buffers":

```cpp
struct CaptureBuffer {
    int externalInputIndex;    // Source: external input index (< 0)
    int crossSegmentSlotIdx;   // Source: cross-segment output slot (>= 0)
    NDArray* buffer;           // Fixed-address buffer
    size_t capturedSize;       // Size at capture time
};

// Before replay: copy current inputs into capture buffers
for (auto& cb : seg.captureBuffers) {
    NDArray* src = externalArrays[cb.externalInputIndex];
    cudaMemcpyAsync(cb.buffer->specialBuffer(), src->specialBuffer(), 
                    srcBytes, cudaMemcpyDeviceToDevice, stream);
}
```

#### 4. Shape Key Invalidation

Graphs are invalidated when input shapes change:

```cpp
LongType computeSegmentShapeKey(const GraphSegment& seg, 
                                 NDArray** externalArrays, int numExt) {
    LongType key = 0;
    for (int s = seg.startSlot; s <= seg.endSlot; s++) {
        for (int i = 0; i < slots_[s].numInputs; i++) {
            int srcIdx = slots_[s].inputSourceIndices[i];
            if (srcIdx < 0) {  // External input
                NDArray* arr = externalArrays[-(srcIdx + 1)];
                key ^= hashShape(arr->shapeInfo());
            }
        }
    }
    return key;
}
```

### Visualization (PyTorch-style)

Following PyTorch's `torch.cuda.CUDAGraph.debug_dump()` pattern:

#### Chrome Trace Export

```cpp
bool CudaGraphHandle::exportToChromeTrace(const std::string& filename) const;
```

Outputs JSON loadable in `chrome://tracing`:
- Per-node timing (kernels, memcpy, memset)
- Flow events for dependencies
- Execution timeline showing replay history
- Metadata (device ID, node counts, statistics)

#### HTML Visualization

```cpp
bool CudaGraphHandle::exportToHtml(const std::string& filename) const;
```

Standalone HTML with:
- Summary statistics cards
- Color-coded node list by type
- Execution timeline bars
- Embedded CSS styling

#### Debug Dump

```cpp
bool CudaGraphHandle::debugDump(const std::string& basePath) const;
```

Creates multiple files:
- `{basePath}.dot` - GraphViz DOT format
- `{basePath}.json` - Chrome trace JSON
- `{basePath}.html` - HTML visualization
- `{basePath}_nodes.json` - Detailed node information

### Java API

```java
NativeOps nativeOps = NativeOpsHolder.getInstance().getDeviceNativeOps();

// Enable CUDA graphs for a plan
nativeOps.setPlanCudaGraphsEnabled(planHandle, true);

// After execution, get statistics
int numCaptured = nativeOps.getPlanNumCapturedGraphSegments(planHandle);
int totalReplays = nativeOps.getPlanTotalGraphReplays(planHandle);
String stats = nativeOps.getPlanCaptureStats(planHandle);

// Visualization exports
nativeOps.exportPlanCudaGraphChromeTrace(planHandle, "trace.json");
nativeOps.exportPlanCudaGraphHtml(planHandle, "graph.html");
nativeOps.debugDumpPlanCudaGraph(planHandle, "debug/graph");

// Programmatic access
String json = nativeOps.getPlanCudaGraphChromeTraceJson(planHandle);
nativeOps.clearPlanCudaGraphTimeline(planHandle);

// Debug output
nativeOps.printPlanCapturedGraphDebug(planHandle);
```

## Implementation Details

### Execution Flow

```cpp
Status NativeDynamicShapePlan::execute(
    NDArray** externalInputs, int numExternalInputs,
    NDArray** requestedOutputs, int numRequestedOutputs,
    void* stream) 
{
    // Pre-execution cleanup
    flushPendingClose(stream);
    invalidateStaleGraphs(externalInputs, numExternalInputs);
    
    // Execute segments
    for (auto& segment : segments_) {
        if (segment.cachedGraph && shapeKeyMatches) {
            // REPLAY path
            updateCaptureBuffers(segment, externalInputs);
            segment.cachedGraph->launchAsync(cudaStream);
            totalGraphReplays_++;
        } else {
            // WARMUP or CAPTURE path
            executeSegmentSlotBySlot(segment, externalInputs, stream);
            
            if (shouldCapture(segment)) {
                captureSegment(segment, externalInputs, stream);
            }
        }
    }
    
    return Status::OK;
}
```

### Host-Only Op Detection

Some ops do host-side work that doesn't capture (e.g., shape_of syncs to host):

```cpp
struct CaptureAuditEntry {
    int slotIndex;
    std::string opName;
    size_t nodesBefore, nodesAfter;
    size_t nodesContributed;  // nodesAfter - nodesBefore
    
    bool isHostOnly() const { return nodesContributed == 0; }
};

// During capture, track node count before/after each op
size_t nodesBefore = graph->getNumNodesDuringCapture(stream);
executeSlot(slotIdx, ...);
size_t nodesAfter = graph->getNumNodesDuringCapture(stream);
```

### Memory Management

1. **Capture Buffers**: Allocated once, reused across replays
2. **Graph Handles**: Shared pointers with automatic cleanup
3. **Pinned Host Memory**: For H2D copies during capture, freed on graph destruction

## Consequences

### Positive

1. **Reduced Latency**: Eliminates per-op launch overhead (10-100x for small ops)
2. **Transparent**: Works with existing SameDiff code without changes
3. **Dynamic Shape Support**: Handles KV cache growth via shape key invalidation
4. **Debugging**: Comprehensive audit trail for host-only ops
5. **Visualization**: PyTorch-compatible Chrome trace export

### Negative

1. **Memory Overhead**: Capture requires all intermediate buffers simultaneously
2. **Warmup Cost**: First 1-2 executions are slower due to capture
3. **Shape Sensitivity**: Shape changes trigger re-capture (can't handle arbitrary dynamics)
4. **Debugging Complexity**: Errors during replay are harder to diagnose

### Mitigations

1. **Pre-capture Memory Check**: Estimate segment memory needs before capture
2. **OOM Retry Cooldown**: Back off after OOM, retry after several executions
3. **Segment Size Limits**: Balance between capture benefit and memory cost
4. **Verbose Logging**: Detailed capture audit helps identify host-only ops

## Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `minCaptureSegmentSize` | 10 | Minimum ops for capture (smaller → slot-by-slot) |
| `maxCaptureSegmentSize` | 50 | Maximum ops per segment |
| `cudaGraphsEnabled` | false | Master switch for graph capture |
| `shapesFrozen` | false | Skip shape invalidation checks |

System property: `nd4j.dsp.cudaGraphs.enabled=true/false`

## Alternatives Considered

### 1. Full Graph Capture (No Segmentation)
**Rejected**: Can't handle shape-dependent ops (reshape, gather with dynamic indices)

### 2. JIT Compilation (Triton)
**Partial**: Implemented separately for kernel fusion; complements CUDA graphs

### 3. Persistent Execution Plans (cuDNN-style)
**Rejected**: Less flexible than CUDA graphs, doesn't handle arbitrary op sequences

## Usage Guide

### Basic Usage

```java
// Compile and execute with CUDA graphs enabled
SameDiff sd = SameDiff.create();
// ... build graph ...

DynamicShapePlan plan = sd.compilePlan();
Pointer handle = nativeOps.compileDynamicShapePlan(serializedPlan);

// Enable CUDA graphs
nativeOps.setPlanCudaGraphsEnabled(handle, true);
nativeOps.setPlanMinCaptureSegmentSize(handle, 5);  // For testing

// Warmup + capture happens automatically
for (int i = 0; i < 100; i++) {
    Map<String, INDArray> outputs = executeNativePlan(handle, plan, inputs);
}

// Check capture status
String stats = nativeOps.getPlanCaptureStats(handle);
// "captured=3(45slots)|oomRetrying=0(0slots)|..."

// Export visualization
nativeOps.debugDumpPlanCudaGraph(handle, "llm_graph");
```

### Chrome Trace Visualization

1. Export trace: `nativeOps.exportPlanCudaGraphChromeTrace(handle, "trace.json")`
2. Open Chrome: `chrome://tracing`
3. Load `trace.json`
4. Navigate with WASD keys, click events for details

The trace shows:
- Kernel launches (green bars)
- Memory operations (yellow bars)
- Graph replay events (blue bars)
- Flow arrows for dependencies

### Troubleshooting

| Issue | Solution |
|-------|----------|
| No graph capture | Check segment size ≥ minCaptureSegmentSize |
| Stale outputs on replay | Check `validatePlanCapturedGraph()` for host-only ops |
| OOM during capture | Reduce maxCaptureSegmentSize or enable OOM retry |
| Shape mismatch errors | Shapes changed → graph invalidated, re-capture on next exec |

## Files Changed

| File | Purpose |
|------|---------|
| `libnd4j/include/execution/cuda/CudaGraphScheduler.h` | CudaGraphHandle and scheduler declarations |
| `libnd4j/include/execution/cuda/CudaGraphHandle.cu` | Graph handle implementation + visualization |
| `libnd4j/include/execution/cuda/CudaGraphScheduler.cu` | Scheduler implementation |
| `libnd4j/include/graph/NativeDynamicShapePlan.h` | Graph segment structure, segmentation API |
| `libnd4j/include/graph/impl/NativeDynamicShapePlan.cpp` | Segment execution, capture, replay logic |
| `libnd4j/include/legacy/NativeOps.h` | JNI function declarations |
| `libnd4j/include/legacy/cuda/NativeOps_dsp.cu` | CUDA-specific JNI implementations |
| `libnd4j/include/legacy/cpu/NativeOps_dsp.cpp` | CPU stubs (no-op) |
| `nd4j/.../nativeblas/NativeOps.java` | Java interface |
| `nd4j/.../bindings/Nd4jCuda.java` | Native method declarations |
| `platform-tests/.../CudaGraphVisualizationTest.java` | Visualization tests |

## Related Decisions

- **ADR-OpTimingTracker**: Complementary per-op timing for profiling
- **Triton Graph Backend**: Alternative optimization path for kernel fusion
- **Dynamic Shape Plan Executor**: Java-side orchestration layer

## References

- [CUDA Graphs Documentation](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#cuda-graphs)
- [PyTorch CUDA Graphs](https://pytorch.org/blog/accelerating-pytorch-with-cuda-graphs/)
- [PyTorch CUDAGraph.debug_dump()](https://pytorch.org/docs/stable/generated/torch.cuda.CUDAGraph.html)

## Authors

- Implementation: deeplearning4j team
- Visualization: Added 2026-02-18
