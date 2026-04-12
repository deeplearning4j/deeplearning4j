# Multi-Task Results: Investigate DSP divergence root causes

**Subtasks:** 4

---

## context-pool-contamination (qwen)

Subagent 'qwen' completed in 173.5s

## Summary
Investigation complete. The `contextPool_` indexing and iArgs/tArgs handling are architecturally sound — no contamination vector found there. The divergence is likely from a different source (stale input pointers, shape key issues, or buffer-sharing through view ops).

**Full output (268 chars) written to:** `/home/agibsonccc/Documents/GitHub/deeplearning4j/.kompile/task-results/qwen-20260412-115206.md`
Use the `read` tool to access the full result if needed.

---

## syncToDevice-missing (qwen)

Subagent 'qwen' completed in 291.1s

## Summary
Now I have a comprehensive picture. Let me compile the findings:

---

## Investigation Results: Missing `syncToDevice` Causing Stale GPU Buffer Reads in DSP

### Finding 1: **NO `prepareSpecialUse` / `registerSpecialUse` in DSP slot execution** ✅ CONFIRMED GAP

`NativeDynamicShapePlan_slotexec.cpp` contains **zero calls** to `prepareSpecialUse` or `registerSpecialUse`. These are the standard CUDA coherency primitives used by every legacy NativeOps entry point (pairwise, transform, reduce, etc.) and all NDArray math operations.

**What they do:**
- `prepareSpecialUse({writes}, {reads})` → calls `syncToDevice()` on all read arrays, allocates device buffers for writes
- `registerSpecialUse({writes}, {reads})` → ticks `tickReadDevice()` / `tickWriteDevice()` actuality counters

**Impact:** Without these, the DSP path skips the systematic device coherency that the standard path gets for free.

---

### Finding 2: **Selective `syncToDevice` — only for data-dependent/control inputs** ✅ CONFIRMED GAP


... (5747 chars total) ...


1. **Producers** write to device but never call `tickWriteDevice()` 
2. **Consumers** don't call `prepareSpecialUse` → don't call `syncToDevice()` → don't verify device data is current
3. **`syncToSpecial()` early-exits** based on stale `isSpecialActual()` state

This causes small numerical errors (reading partially-stale GPU buffers) that compound through 30 transformer layers.

### Recommended Fix

Add `NDArray::prepareSpecialUse(outputs, inputs)` before `slot.ident.op->execute(&ctx)` and `NDArray::registerSpecialUse(outputs, inputs)` after, mirroring the standard legacy NativeOps pattern. This ensures all inputs are synced to device and actuality counters stay accurate.

**Full output (5747 chars) written to:** `/home/agibsonccc/Documents/GitHub/deeplearning4j/.kompile/task-results/qwen-20260412-115403.md`
Use the `read` tool to access the full result if needed.

---

## shapeFunctionOverride-effect (qwen)

Subagent 'qwen' completed in 570.5s

## Summary
## Investigation Results

### Hypothesis Verdict: **CONFIRMED** — Multiple ops depend on output shape/strides during execution

---

### 1. What `setShapeFunctionOverride(true)` Actually Skips

**File:** `libnd4j/include/ops/declarable/impl/DeclarableOp.cpp`

When `shapeFunctionOverride()` is true, `prepareOutputs()` returns immediately at **line 410**:

```cpp
if (fp && ctx.shapeFunctionOverride()) {
  return (int)ctx.fastpath_out().size();
}
```

This skips:


... (5661 chars total) ...


**Option A — Per-op opt-out:** Add a virtual method to `DeclarableOp`:
```cpp
virtual bool requiresShapeValidationDuringExecute() const { return false; }
```
Override to `true` for `matmul` and any other affected ops. In `prepareOutputs()`, skip the early return for these ops, ensuring `calculateOutputShape` runs and validation happens.

**Option B — Always validate shapes in execute:** In the warmup path, instead of setting `shapeFunctionOverride(true)`, set a flag that tells `prepareOutputs` to skip **allocation** but still run `calculateOutputShape` and validate. The pre-allocated arrays are kept if shapes match; replaced if they don't.

**Option B is safer** — it preserves the DSP pre-allocation intent while catching shape mismatches. **Option A is more targeted** but requires auditing every op for correctness.

**Full output (5661 chars) written to:** `/home/agibsonccc/Documents/GitHub/deeplearning4j/.kompile/task-results/qwen-20260412-115843.md`
Use the `read` tool to access the full result if needed.

---

## execution-order-diff (qwen)

Subagent 'qwen' completed in 171.3s

## Summary
## Investigation Summary: DSP Execution Order vs Standard Path

I've completed a thorough investigation of the DSP execution order compared to the standard InferenceSession path. Here are the findings:

### 1. **Slot Order Determination (NativePlanCompiler.cpp)**

**How DSP determines order:**
- The Java `DynamicShapePlanCompiler` receives a `ForwardExecutionDAG` with a pre-computed topological order
- Slot N in DSP = the N-th op in the filtered execution order (skipping VARIABLE_INIT, PLACEHOLDER_SET nodes)
- The native `NativePlanCompiler::compile()` iterates `opNodes` in array order and assigns `stepIdx` sequentially
- **Critical:** The topological sort happens in Java via `ForwardExecutionDAGBuilder.createTopologicalOrder()` (line 462-480)

**How standard path determines order:**
- `ForwardExecutionDAGBuilder.buildForwardDAG()` performs backward traversal from requested outputs
- Uses `findRequiredSubgraph()` to find all producer ops transitively
- Creates topological order via recursive DFS (`topologicalSort()`)

**Key finding:** Both paths use the **same** `ForwardExecutionDAG`, so they should execute ops in the same order **if the DAG is built correctly**.

### 2. **Potential Order Divergence Issues**

... (6148 chars total) ...


2. **Add logging** to print the actual execution order in both paths:
   - Java: Log `dag.getExecutionOrder()` before DSP compilation
   - Native: Add DSP_DIAG to print slot order in `NativePlanCompiler::compile()`

3. **Check if `attn_mask_reformat` output slot is shared** across layers by examining the ONNX model's variable names

4. **Verify `hasControlFlow_`** is not set incorrectly for transformer graphs

5. **Test with `-Dorg.nd4j.inference.dynamicShapePlan=false`** to confirm divergence disappears with standard path

**Full output (6148 chars) written to:** `/home/agibsonccc/Documents/GitHub/deeplearning4j/.kompile/task-results/qwen-20260412-115204.md`
Use the `read` tool to access the full result if needed.

---

**Summary:** 4/4 subtasks completed successfully.