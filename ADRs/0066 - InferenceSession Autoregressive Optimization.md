# ADR: InferenceSession Autoregressive Optimization

## Status

Implemented

Proposed by: Adam Gibson (January 2025)

Discussed with: Development Team

## Context

InferenceSession is the primary execution engine for SameDiff graph inference. Its original design targeted batch inference workloads where a graph executes once with fixed inputs and produces outputs. Autoregressive generation — where the graph executes thousands of times in a loop with incrementally changing inputs — exposed several performance bottlenecks:

**Redundant Shape Calculation**: Every execution step recalculated output shapes for all ops via JNI calls to C++, even though 95%+ of shapes don't change between decode steps (only the KV cache dimension grows).

**Redundant Array Allocation**: Intermediate arrays were allocated fresh on every step, even when the previous step's arrays had the same shape and could be reused.

**Unnecessary GPU Synchronization**: INT/LONG tensor values were synchronized from GPU to host on every step for shape-dependent ops, even when those values hadn't changed.

**Reshape Copy Overhead**: Reshape operations created new contiguous arrays even when the source was already contiguous and a view would suffice.

**Pool Trim Overhead**: CudaMemoryPool trim operations (which synchronize CUDA streams) were called too frequently, blocking GPU execution unnecessarily.

## Decision

We implement a suite of optimizations in InferenceSession specifically targeting autoregressive generation workloads, controlled by system properties for gradual rollout.

### Shape Caching

Thread-local shape caches eliminate redundant `calculateOutputShape` JNI calls:

```java
private static final ThreadLocal<Map<Long, long[][]>> outputShapeCacheTl =
    ThreadLocal.withInitial(HashMap::new);

// Cache key: hash(opName, inputShapes, inputDtypes)
// For value-dependent ops: also includes INT/LONG input values
long cacheKey = computeShapeCacheKey(op, inputs);
long[][] cachedShapes = outputShapeCache.get(cacheKey);
if (cachedShapes != null) {
    return cachedShapes;  // Skip JNI call
}
```

Cache hit rate exceeds 95% after the first decode step because most op shapes are determined by model architecture, not input data. Only ops depending on the growing KV cache dimension see cache misses.

### Output Array Caching

Intermediate arrays whose shapes don't change between steps are reused:

```java
private static final ThreadLocal<Map<String, INDArray>> outputArrayCacheTl =
    ThreadLocal.withInitial(HashMap::new);

// Only cache non-output (intermediate) arrays
if (!isRequestedOutput && cachedArray != null && shapeMatches(cachedArray, expectedShape)) {
    return cachedArray;  // Reuse without allocation
}
```

This works in concert with ArrayCacheMemoryMgr — arrays that aren't reused by the output cache are returned to the capacity cache for potential reuse by other ops.

### INT/LONG Sync Deferral

Synchronizing INT/LONG tensors from GPU to host is deferred until actually needed:

```java
// Only sync when shape cache misses AND op is value-dependent
if (shapeCacheMiss && op.isValueDependent()) {
    syncIntLongInputs(op.getIntLongInputs());
    timingIntLongSyncCount++;
} else {
    timingIntLongSyncSkipCount++;
}
```

When all INT/LONG inputs come from external sources (constants, variables), the sync is skipped entirely since no prior GPU op has written to these buffers. This avoids blocking the GPU pipeline for ops like `shape_of` that always receive the same constant input.

### Reshape View Optimization

Reshape operations use views instead of copies when the source is contiguous:

```java
if (isContiguous(source) && !requiresCopy(source, targetShape)) {
    return source.reshape(targetShape);  // View — no allocation, no copy
} else {
    return source.dup().reshape(targetShape);  // Must copy for non-contiguous
}
```

Tracking counters (`timingReshapeViewUsed`, `timingReshapeViewSkipped`) show that 80%+ of reshapes can use views in autoregressive workloads, saving both allocation time and memory bandwidth.

### Conditional Pool Trimming

Pool trimming is rate-limited to avoid excessive CUDA stream synchronization:

```java
private static final int TRIM_INTERVAL = Integer.getInteger(
    "org.nd4j.inference.trimInterval", 10);

if (dspStepCount % TRIM_INTERVAL == 0 || dspStepCount <= 1) {
    CudaMemoryPool.trimPool(currentDevice);
}
```

**Step 0-1**: Always trim. These are the prefill→decode transition steps where large encoder buffers are freed, and trimming recovers significant memory.

**Every N steps**: Periodic trim prevents unbounded pool growth while minimizing sync overhead. The default interval of 10 steps balances memory recovery against GPU pipeline continuity.

### Execution Plan Selection

InferenceSession automatically selects the best execution strategy:

```java
if (hasDynamicShapes && !hasControlFlow) {
    // DynamicShapePlan: pre-compiled wiring, per-slot shape cache
    plan = DynamicShapePlanCompiler.compile(dag);
} else if (!hasDynamicShapes) {
    // ExecutionPlan: pre-allocated fixed workspaces (ORT-style)
    plan = ExecutionPlanCompiler.compile(dag);
} else {
    // Standard interpreted execution (control flow graphs)
    // No pre-compilation possible
}
```

DynamicShapePlan is enabled by default (`-Dorg.nd4j.inference.dynamicShapePlan=true`) and provides 62% faster inference for autoregressive generation.

### Diagnostic Timing

Comprehensive timing instrumentation is available via `-Dorg.nd4j.inference.timing=true`:

```
=== Inference Timing Summary ===
  Shape cache hits: 18,648 (97.2%)
  Shape cache misses: 536 (2.8%)
  INT/LONG syncs: 24
  INT/LONG sync skips: 512
  Reshape views used: 3,924 (82.1%)
  Reshape views skipped: 856 (17.9%)
  Pool trims: 26
  Total decode steps: 256
  Avg step time: 1284ms
```

This makes it easy to identify remaining bottlenecks and validate that optimizations are effective.

## Consequences

### Advantages

**62% Faster Inference**: Combined optimizations reduce per-step overhead from ~3400ms to ~1284ms for SmolDocling on RTX 4090.

**95%+ Shape Cache Hit Rate**: Eliminates the dominant source of JNI overhead in autoregressive decoding.

**80%+ Reshape View Rate**: Reduces allocation count and memory bandwidth consumption for reshape-heavy graphs.

**Tunable**: All optimizations are controlled by system properties and can be disabled individually for debugging.

### Disadvantages

**Thread-Local State**: Shape and output caches are thread-local, consuming memory proportional to graph size per thread. Multi-threaded inference multiplies this overhead.

**Cache Invalidation**: Shape cache entries are never explicitly invalidated. If model weights change (fine-tuning during inference), stale cache entries could produce incorrect shapes. This is safe for inference-only workloads.

**Trim Interval Sensitivity**: Too-infrequent trimming causes gradual memory growth; too-frequent trimming blocks the GPU pipeline. The default interval of 10 is empirically tuned for VLM workloads.

## References

- InferenceSession.java in nd4j-api
- DynamicShapePlanCompiler.java
- DynamicShapePlanExecutor.java
- ADR 0061 - DynamicShapePlan Execution
- ADR 0062 - Java-Side Shape Inference
