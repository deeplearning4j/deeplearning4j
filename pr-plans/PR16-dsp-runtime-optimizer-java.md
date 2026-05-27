# PR16: DSP Runtime & Graph Optimizer (Java)

**Estimated files:** ~131
**Merge layer:** 5
**Complexity:** High — core DSP execution path + optimization passes
**Reviewers:** DSP team, optimizer specialists

## Description

Java-side DSP runtime (DynamicShapePlan compiler/executor, execution phases,
slot management, replay analytics, autotuner) and the graph optimizer framework
(25 optimization passes, fusion patterns, algebraic simplifications).

## DSP Runtime (103 files)

### C++ files — assigned to other PRs (cross-reference only)
- `libnd4j/include/execution/` — **assigned to PR08** (platform backends / execution infrastructure)
- `libnd4j/include/graph/execution/Logic*` — **assigned to PR09** (DSP/graph execution)
- `libnd4j/include/graph/GraphExecutioner.*` — **assigned to PR09**
- `libnd4j/include/graph/DspDiagnostics.*` — **assigned to PR09**

PR16 is **Java-only** — the C++ files above are listed here for context but belong to PR08/PR09.

### Java DSP execution (~36)
- `DynamicShapePlan.java`
- `DynamicShapePlanCompiler.java`
- `DynamicShapePlanExecutor.java`
- `DynamicShapeSlot.java`
- `GraphExecutionMode.java`
- `ForwardExecutionDAG.java`
- `ForwardExecutionDAGBuilder.java`
- `ExecutionPhase.java`
- `PlanPhase.java`
- `PlanIntrospection.java`
- `DspPlanIntrospection.java`
- `SlotState.java`
- `DspHandle.java`
- `DspDebugger.java`
- `DspPlanAssertions.java`
- `DspReplayTransferAnalytics.java`
- `DspCompilationMode.java`
- `ReplayProfileManager.java`
- `AutoTuner.java`
- `BackendPlanManager.java`
- `BufferAllocation.java`
- `BufferAllocKind.java`
- `ModelMemoryEstimator.java`
- `UpdaterOpsAppender.java`
- `GraphNodePhase.java`
- `TritonCacheManager.java`
- `CollectiveCommunicator.java`
- `CollectiveCommunicatorFactory.java`
- `LocalCollectiveCommunicator.java`
- `NcclCommunicator.java`
- `DevicePlacementPlanner.java`
- `TensorParallelConfig.java`
- `TensorParallelRunner.java`
- `PipelineParallelRunner.java`

### Java graph executioners (~2)
- `GraphExecutioner.java`
- `NativeGraphExecutioner.java`

## Graph Optimizer (28 files)

### Framework (~3)
- `GraphOptimizer.java`
- `OptimizationHelper.java`
- `Optimizer.java`

### Optimization passes (~25)
- `ActivationFusion.java`
- `AlgebraicOptimizations.java`
- `ArithmeticChainOptimizations.java`
- `AttentionFusionOptimizations.java`
- `BroadcastElimination.java`
- `CommonSubexpressionElimination.java`
- `ConcatSplitOptimizations.java`
- `ConstantFunctionOptimizations.java`
- `CuDNNFunctionOptimizations.java`
- `GatedDeltaNetFusionOptimizations.java`
- `HorizontalFusionOptimizations.java`
- `IdentityFunctionOptimizations.java`
- `LinearFusionOptimizations.java`
- `MatMulChainOptimizations.java`
- `NormalizationFusionOptimizations.java`
- `OptimizationUtils.java`
- `PeepholeOptimizations.java`
- `QuantizationOptimizations.java`
- `RedundancyElimination.java`
- `Rematerialization.java`
- `ReorderingOptimizations.java`
- `SelectWhereOptimizations.java`
- `ShapeFunctionOptimizations.java`
- `StrengthReduction.java`
- `UnusedFunctionOptimizations.java`

### ADRs (2)
- `ADRs/0062 - Java-Side Shape Inference.md` — Java-side shape calculation eliminating JNI round-trips for stable-shape ops
- `ADRs/0066 - InferenceSession Autoregressive Optimization.md` — Shape caching, array reuse, pool-trim throttling for autoregressive throughput

## Review Focus

- DynamicShapePlan lifecycle (compile → warmup → freeze → replay)
- ForwardExecutionDAG — DAG construction from SameDiff graph
- Optimizer passes — each must preserve semantic equivalence
- Fusion patterns — verify they don't break accuracy (run DSP matrix)
