# Multi-Task Results: Investigate 3 remaining DSP divergence causes

**Subtasks:** 3

---

## external-input-mapping (qwen)

Subagent 'qwen' completed in 246.7s

## Summary
Now I have all the information I need. Let me write up the complete investigation findings:

## **INVESTIGATION COMPLETE: External Input Index Mapping Analysis**

### Summary

I've thoroughly investigated the external input index mapping in DSP plan compilation. Here are the findings:

---

### **CRITICAL FINDING: Non-Deterministic Set Iteration**

**The root cause is in `ForwardExecutionDAGBuilder.findConstants()` / `findVariables()` / `findPlaceholders()`:**

```java
// Line 654-676 in ForwardExecutionDAGBuilder.java
private Set<String> findConstants() {
    return sameDiff.getVariables().values().stream()
        .filter(v -> v.getVariable().getVariableType() == VariableType.CONSTANT)
        .map(v -> v.getVariable().name())

... (7048 chars total) ...

    return requiredVariables.stream()
        .filter(varName -> {
            Variable var = sameDiff.getVariables().get(varName);
            return var != null && var.getVariable().getVariableType() == VariableType.PLACEHOLDER;
        })
        .collect(Collectors.toCollection(LinkedHashSet::new));  // ← PRESERVE ORDER
}
```

This ensures **deterministic, reproducible ordering** across DAG rebuilds while preserving the PatriciaTrie's natural lexicographic order.

**Full output (7048 chars) written to:** `/home/agibsonccc/Documents/GitHub/deeplearning4j/.kompile/task-results/qwen-20260412-134638.md`
Use the `read` tool to access the full result if needed.

---

## op-execution-order (qwen)

Subagent 'qwen' completed in 195.1s

## Summary
Now I have a clear picture. Let me compile my findings into a comprehensive report.

---

## Investigation Results: DSP vs Standard InferenceSession Execution Order

### 1. How DSP Determines Slot Order

**`DynamicShapePlanCompiler.java`** (line ~131):
```java
List<ExecutionNode> executionOrder = dag.getExecutionOrder();
```

The compiler iterates through `dag.getExecutionOrder()` sequentially. Each node in this list becomes a slot at index `stepIdx` (0, 1, 2, ...). The slot index directly corresponds to the position in the `executionOrder` list. The compiler builds:
- `slots[stepIdx]` = the op at `executionOrder.get(stepIdx)`
- Input wiring resolves to either negative external indices or positive output slot indices
- The compiled plan is then executed slot 0, 1, 2, ... by `executeSegmentSlotBySlot` in C++ (`NativeDynamicShapePlan_segments.cpp`, line ~819: `while (stepIdx <= seg.def.endSlot)`)

### 2. How Standard InferenceSession Determines Execution Order


... (5690 chars total) ...

### Summary of Findings

| Aspect | DSP Path | Standard Path | Same? |
|--------|----------|---------------|-------|
| DAG source | `dag.getExecutionOrder()` | `dag.getExecutionOrder()` | **YES** |
| Topological sort | DFS post-order | DFS post-order | **YES** |
| Tiebreaking | HashMap iteration order | HashMap iteration order | **SAME DAG = same order** |
| Subgraph inclusion | Backward-only from outputs | May include forward-expanded placeholders | **POTENTIALLY DIFFERENT** |

**Most likely root cause:** The `attn_mask_reformat` subgraph is **pruned** from the DSP's DAG (backward-only traversal from outputs) but may be **present** in the standard path's DAG. If the standard path includes it and DSP doesn't, the attention mask is computed differently, causing divergence that grows with each layer.

**Full output (5690 chars) written to:** `/home/agibsonccc/Documents/GitHub/deeplearning4j/.kompile/task-results/qwen-20260412-134546.md`
Use the `read` tool to access the full result if needed.

---

## fastpath-vs-variablespace (qwen)

Subagent 'qwen' completed in 783.3s

## Summary
I now have enough information to compile the full analysis. Let me write it up:

---

## Investigation Results: DSP Fastpath vs Standard Path Divergence

### 1. Full Execution Path for Fastpath Mode (`DeclarableOp::execute()`)

The execution sequence is:

```
execute(Context*) 
  → validateDataTypes()       // validates input/output types match
  → prepareOutputs()          // allocates/validates output arrays
    → if (shapeFunctionOverride && isFastPath): EARLY RETURN (skip shape inference + allocation)
    → else: calculateOutputShape() → allocate output if needed
  → KernelDispatchHelper::dispatchWithAutoTune()  // try helper (e.g., FlashAttentionHelper)
  → validateAndExecute()      // calls op's executeOp() if no helper found
```


... (7826 chars total) ...


3. **CUDA graph capture vs replay address mismatch**. `MmulHelper.cu` has explicit handling for `tl_graphExecutionActive` (line ~1277). If DSP uses CUDA graphs and the standard path doesn't, or if the graph was captured with different array addresses than replay, the cached cuBLAS Lt epilogue state or cast cache could use wrong pointers.

### Recommended Next Steps

1. **Log strides**: Add debug logging in both paths just before `MmulHelper::matmul()` to print `x->strideAt(0)`, `y->strideAt(0)`, `z->strideAt(0)`, `z->ordering()` for both DSP and standard paths.

2. **Force contiguous allocation**: As a test, have DSP call `output->dup()` to create a fresh contiguous copy before passing to the op, and compare results.

3. **Check AttentionWorkspace**: Log the workspace key and buffer shapes to verify no stale buffer reuse.

**Full output (7826 chars) written to:** `/home/agibsonccc/Documents/GitHub/deeplearning4j/.kompile/task-results/qwen-20260412-135535.md`
Use the `read` tool to access the full result if needed.

---

**Summary:** 3/3 subtasks completed successfully.