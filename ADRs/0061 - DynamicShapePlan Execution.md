# ADR: DynamicShapePlan Execution

## Status

Implemented

Proposed by: Adam Gibson (January 2025)

Discussed with: Development Team

## Context

SameDiff's execution engine supports two primary inference modes: standard interpreted execution (InferenceSession) and pre-compiled ExecutionPlan. Both have significant limitations for autoregressive generation workloads like LLM/VLM token decoding:

**Interpreted Execution (InferenceSession)**: Re-analyzes the graph structure on every execution step. Uses string-keyed HashMaps for variable lookup, dependency tracking, and array management. For a vision encoder with 1962 ops, this overhead adds ~100ms per frame from HashMap operations alone.

**Static ExecutionPlan**: Pre-allocates fixed workspaces assuming static shapes. Works well for batch inference with constant shapes, but fails for autoregressive generation where the KV cache grows by one position each decode step. Shape changes force full plan recompilation, defeating the purpose of pre-compilation.

**The Fundamental Problem**: Autoregressive generation has shapes that change predictably (KV cache grows) but the graph topology stays constant. We need a system that pre-compiles the graph wiring once and handles shape changes without recompilation.

## Decision

We implement DynamicShapePlan — a pre-compiled, index-based graph execution system that separates graph topology (compiled once) from shape handling (evaluated per step with caching).

### Architecture

```
┌───────────────────────────────────────────────────────────────┐
│                   DynamicShapePlanCompiler                      │
│  Input: ForwardExecutionDAG                                    │
│  Output: DynamicShapePlan (flat-indexed slots + release sched) │
└──────────────────────────┬────────────────────────────────────┘
                           │
                           ▼
┌───────────────────────────────────────────────────────────────┐
│                     DynamicShapePlan                            │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ DynamicShapeSlot[] slots  (1962 slots for vision enc)    │  │
│  │  - opName, opHash                                        │  │
│  │  - inputSlotIndices[] (>=0: op output, <0: external)     │  │
│  │  - outputSlotIndex                                       │  │
│  │  - iArgs, tArgs, bArgs, dArgs                            │  │
│  │  - shapeCache (hash → cached output shapes)              │  │
│  │  - needsIntLongSync flag                                 │  │
│  │  - needsZeroedOutput flag                                │  │
│  └──────────────────────────────────────────────────────────┘  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ int[][] releaseAtStep  (pre-computed liveness schedule)   │  │
│  │  releaseAtStep[i] = slot indices dead after step i        │  │
│  └──────────────────────────────────────────────────────────┘  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ INDArray[] outputSlots  (flat result storage)             │  │
│  └──────────────────────────────────────────────────────────┘  │
└──────────────────────────┬────────────────────────────────────┘
                           │
              ┌────────────┴────────────┐
              ▼                         ▼
┌──────────────────────┐  ┌──────────────────────────────┐
│ DynamicShapePlan     │  │ NativeDynamicShapePlan       │
│ Executor (Java)      │  │ (C++ native executor)        │
│ - Fallback path      │  │ - Fast path                  │
│ - Full diagnostics   │  │ - Binary serialization       │
│ - Shape override     │  │ - CUDA graph support         │
└──────────────────────┘  └──────────────────────────────┘
```

### Index-Based Slot Architecture

The core optimization replaces string-keyed HashMap lookups with flat integer-indexed arrays:

```java
public class DynamicShapeSlot {
    int slotIndex;              // Position in outputSlots[]
    String opName;              // For logging/debugging only
    long opHash;                // For native executor dispatch
    int[] inputSlotIndices;     // >=0: outputSlots[idx], <0: externalInputs[-idx-1]
    int outputSlotIndex;        // Where to store result in outputSlots[]
    long[] cachedShapeKey;      // Shape cache key for fast comparison
    long[][] cachedOutputShapes; // Cached output shapes (skip calculateOutputShape)
}
```

Input wiring uses a sign convention: positive indices reference other op outputs in `outputSlots[]`, negative indices reference external inputs (constants, variables, placeholders) via `externalInputs[-idx-1]`. This eliminates all string-based variable resolution at execution time.

### Pre-Computed Liveness Schedule

Instead of maintaining a dependency tracker at runtime (O(n) per release check), the compiler pre-computes exactly which slots become dead after each step:

```java
// releaseAtStep[i] = indices of slots whose last consumer is step i
int[][] releaseAtStep = compiler.computeLivenessSchedule(slots, dag);
```

At runtime, after executing step `i`, the executor iterates `releaseAtStep[i]` and immediately frees those slot arrays. This is O(k) where k is the number of slots dying at that step, typically 1-3.

### Per-Slot Shape Caching

Each slot maintains a shape cache that avoids redundant `calculateOutputShape` JNI calls:

**Shape-Only Ops** (matmul, add, relu, softmax): Cache key is `hash(opName) ^ hash(inputShapes) ^ hash(inputDtypes)`. Input VALUES are excluded — this avoids false cache misses from changing batch dimensions that don't affect output shapes.

**Value-Dependent Ops** (reshape, squeeze, gather, strided_slice): Cache key additionally includes INT/LONG input values, since these ops' output shapes depend on input data, not just input shapes.

On cache hit (95%+ of steps after the first), shape inference is skipped entirely, saving 4-6ms per vision encoder frame.

### Control Flow Exclusion

DynamicShapePlan returns `null` during compilation if any control flow ops are detected (Switch, Merge, Enter, Exit, NextIteration, LoopCond). These ops require frame-based execution semantics that the flat slot architecture cannot support. The system falls back to standard InferenceSession execution for graphs with control flow.

### Native C++ Executor

The plan can be serialized to a compact binary format and sent to C++ via a single JNI call:

```
Binary Layout:
  24-byte header: magic, version, numSlots, numExternalInputs
  Per slot: opHash(8B), numInputs(4B), inputIndices(4B*n),
            numIArgs(4B), iArgs(8B*n), numTArgs(4B), tArgs(8B*n),
            flags(4B: needsSync, needsZero, fullyWriting)
```

The native executor (`NativeDynamicShapePlan.cpp`) processes the entire plan without per-op JNI round-trips, eliminating ~15-20μs overhead per op. For 1962 ops, this saves ~30ms per frame.

### Multi-GPU Device Assignment

The compiler's `assignDevices()` method distributes ops across available GPUs:

1. Query each device's free memory (accounting for pool reservations)
2. Proportionally assign ops based on memory budgets
3. Non-P2P secondary GPUs receive 0% budget by default (configurable via `nd4j.dsp.nonP2pBudgetFraction`)
4. Split parallel groups (ops with identical predecessors) across GPUs for concurrent execution

Non-P2P GPUs are still used for memory spillover via `allocateFailover`, but are not assigned compute work by default due to the 100x slowdown from host-staged data transfers.

### Cached Array Zeroing

Reused arrays from `slotArrayCache_` must be zeroed before reuse. Some ops marked as `FULLY_WRITING_OPS` were assumed to overwrite all output elements, but this assumption proved incorrect for certain ops, causing stale data accumulation across decode steps.

**Fix**: Always `nullify()` reused/cached arrays regardless of the fully-writing flag. New allocations use conditional zeroing (only if `slot.needsZeroedOutput`).

## Consequences

### Advantages

**Execution Speed**: 2-3x faster than interpreted InferenceSession. Vision encoder (1962 ops): ~150ms with DSP vs ~300ms interpreted. Overall 62% faster inference when enabled by default.

**Memory Efficiency**: Pre-computed liveness schedule enables immediate intermediate release, reducing peak GPU memory by ~50% compared to end-of-execution release.

**Shape Cache Efficiency**: 95%+ cache hit rate after first step. Eliminates redundant JNI calls for shape calculation, saving 4-6ms per frame.

**Predictable Allocation**: Index-based slot architecture has zero HashMap overhead. Array indexing is O(1) vs O(1) amortized for HashMap (but with better constants — no hashing, no collision handling, no boxing).

**Replay Path Optimizations**: Runtime replay now includes direct C++ KV-scatter execution (avoiding repeated JNI crossings), frozen-constant detection for value-independent ops, and copy skipping for unchanged graph-capture buffers.

**Allocation and Mode Control Improvements**: Decode-heavy workloads benefit from additional zero-copy view reuse and tighter output pre-allocation behavior, while explicit DSP compilation/execution modes allow safer fallback and targeted performance tuning.

### Disadvantages

**No Control Flow Support**: Graphs with Switch/Merge/Enter/Exit operations must fall back to interpreted execution. This limits DSP to feedforward models (which covers most inference workloads).

**Compilation Cost**: First execution pays a one-time compilation cost (~10-20ms for large graphs). Amortized over hundreds of decode steps, this is negligible.

**Memory Overhead**: The flat `outputSlots[]` array reserves space for all intermediate results simultaneously, though liveness-based release keeps actual occupancy low.

## References

- DynamicShapePlan.java, DynamicShapePlanCompiler.java, DynamicShapePlanExecutor.java
- DynamicShapeSlot.java
- NativeDynamicShapePlan.h, NativeDynamicShapePlan.cpp
- ADR 0048 - Improved SameDiff Execution Framework (predecessor)
