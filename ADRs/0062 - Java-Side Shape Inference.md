# ADR: Java-Side Shape Inference

## Status

Implemented

Proposed by: Adam Gibson (January 2025)

Discussed with: Development Team

## Context

SameDiff's shape inference pipeline historically required a JNI round-trip to C++ for every `calculateOutputShape` call. Each round-trip involves:

1. Marshalling Java arrays to native pointers (~2μs)
2. JNI call overhead (~1μs)
3. C++ shape calculation (varies, 1-50μs depending on op)
4. Marshalling results back to Java (~2μs)
5. CUDA device synchronization for ops that need input values (~100-1000μs)

For autoregressive generation workloads, this overhead is paid on every decode step for every op in the graph. A vision encoder with 1962 ops executing at ~5μs per shape call adds ~10ms per frame just for shape inference — time that could be eliminated for ops with trivially predictable output shapes.

Additionally, certain ops like `reshape` require synchronizing INT/LONG input tensors from GPU to host to read their values for shape calculation. This `cudaMemcpy` D2H transfer blocks the GPU pipeline and adds 100-1000μs per sync. For autoregressive decoding where most shapes don't change between steps, these syncs are almost always wasted.

## Decision

We implement Java-side shape inference for common ops, combined with a per-slot shape cache in DynamicShapePlan, to eliminate unnecessary JNI calls and CUDA synchronization.

### Two-Tier Shape Calculation

**Tier 1: Shape Cache (Per-Slot)**

Each DynamicShapeSlot maintains a cached shape result keyed by a hash of its inputs:

```java
// Shape-only ops: key = hash(opName, inputShapes, inputDtypes)
// Value-dependent ops: key = hash(opName, inputShapes, inputDtypes, intLongValues)
long[] cachedShapeKey;
long[][] cachedOutputShapes;
```

On cache hit (95%+ of decode steps after step 0), shape inference is skipped entirely. The cache key distinguishes between shape-only ops (where output shape depends only on input shapes) and value-dependent ops (where output shape depends on actual input values like reshape targets).

**Tier 2: Java Shape Functions**

For cache misses, Java-side shape functions handle common ops without JNI:

```java
public static long[][] calculateOutputShapesFromJava(String opName,
        List<long[]> inputShapes, List<DataType> inputDtypes,
        long[] iArgs, double[] tArgs) {
    switch (opName) {
        case "reshape": return reshapeShape(inputShapes, iArgs);
        case "shape_of": return shapeOfShape(inputShapes);
        case "cast":     return castShape(inputShapes, dArgs);
        case "matmul":   return matmulShape(inputShapes, bArgs);
        // ... other common ops
    }
    return null; // Fall through to C++ for unknown ops
}
```

### Op Classification

Ops are classified along two dimensions that affect shape inference behavior:

**Shape-Only vs. Value-Dependent**:
- Shape-only (matmul, add, relu, softmax, concat): Output shape determinable from input shapes and op parameters alone
- Value-dependent (reshape, squeeze, expand_dims, gather, strided_slice, pad): Output shape depends on INT/LONG input tensor values

**Fully-Writing vs. Sparse-Output**:
- Fully-writing (matmul, add, multiply, softmax, argmax): Op guarantees every output element is written
- Sparse/partial (scatter, pad, gather, unique): Some output elements may not be written, requiring pre-zeroed buffers

These classifications are determined at compile time and stored as flags in each DynamicShapeSlot.

### INT/LONG Sync Deferral

For value-dependent ops, the shape cache key must include INT/LONG input values. Retrieving these values from GPU memory requires a D2H sync. The system optimizes this in several ways:

1. **External-Only Optimization**: If all INT/LONG inputs come from external sources (constants, variables, placeholders), skip the `commit()` call since no prior GPU op has written to these buffers.

2. **Sync Counting**: Track `timingIntLongSyncCount` vs `timingIntLongSyncSkipCount` for diagnostics. When skip rate is high, the optimization is working well.

3. **Lazy Sync**: Only sync INT/LONG inputs when the shape cache misses AND the op is value-dependent. Shape-only ops never trigger INT/LONG sync regardless of input types.

### ShapeOverride Flag

When Java-side shape inference provides the output shape, the OpContext is flagged with `shapeFunctionOverride=true`:

```cpp
// In C++ op execution:
if (ctx->shapeFunctionOverride()) {
    // Skip calculateOutputShape() and prepareOutputs()
    // Use pre-allocated outputs from Java
}
```

This eliminates redundant C++ shape calculation, saving ~1-2ms per op. For the vision encoder (1962 ops), this saves ~2-4 seconds per frame.

## Consequences

### Advantages

**Latency Reduction**: Eliminates JNI round-trips for 80%+ of shape calculations. Combined with shape caching, reduces shape inference overhead from ~10ms to ~0.5ms per frame.

**GPU Pipeline Continuity**: Deferring INT/LONG syncs keeps the GPU pipeline running uninterrupted. Only value-dependent ops with cache misses trigger synchronization.

**Diagnostic Visibility**: Sync counting and timing instrumentation (enabled via `-Dorg.nd4j.inference.timing=true`) make it easy to identify remaining shape inference bottlenecks.

### Disadvantages

**Maintenance Burden**: Java shape functions must stay synchronized with their C++ counterparts. Adding a new op to C++ requires adding a corresponding Java shape function for optimal performance.

**Correctness Risk**: If a Java shape function produces a different result than the C++ version, the op will execute with incorrectly-shaped output buffers, causing silent data corruption or crashes. Extensive testing is required.

**Limited Coverage**: Only common ops have Java-side implementations. Uncommon ops still require JNI round-trips. The system is designed to degrade gracefully — unknown ops fall through to C++.

## References

- DynamicShapePlanCompiler.java (op classification logic)
- DynamicShapeSlot.java (shape cache implementation)
- DynamicShapePlanExecutor.java (shape override integration)
- NativeDynamicShapePlan.cpp (C++ shape override handling)
