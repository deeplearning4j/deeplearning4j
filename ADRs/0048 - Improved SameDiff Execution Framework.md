# Improved SameDiff Execution Framework

## Status
**Accepted**

Proposed by: Adam Gibson (Date: Current)

Discussed with: Team

## Context

SameDiff's execution of complex graphs with control flow (loops, conditionals) has historically been challenging to debug and optimize. Key issues include:

1. **Loop termination analysis**: Infinite loops are difficult to detect and debug, requiring manual inspection of execution traces
2. **Variable evolution tracking**: Understanding how variables change across loop iterations is crucial for convergence analysis but was previously opaque
3. **Execution planning**: The original execution was interpreted without optimization for repeated subgraph patterns
4. **Cross-frame references**: Operations that reference variables across loop boundaries or conditional scopes were handled ad-hoc
5. **Convergence issues**: The original `initSubgraph` method had convergence problems that prevented proper graph initialization

This ADR documents the new execution framework that addresses these limitations through systematic analysis and optimization.

## Decision

We introduce a comprehensive execution analysis and optimization framework with the following components:

### 1. Variable Evolution Analysis

The framework tracks how variables change across loop iterations to detect patterns:

```java
public class VariableEvolutionAnalysis {
    private Map<String, List<VariableStateInfo>> variableHistory;
    private Map<String, VariablePattern> detectedPatterns;
    
    public enum VariablePattern {
        CONVERGING,      // Values approaching a limit
        DIVERGING,       // Values growing without bound
        OSCILLATING,     // Values alternating in a pattern
        STABLE,          // Values remaining constant
        CHAOTIC          // No discernible pattern
    }
}
```

This enables automatic detection of:
- Convergence rates and predicted iterations to convergence
- Divergent behavior that may indicate bugs
- Oscillating patterns that prevent termination
- Numerical stability issues

### 2. Loop Termination Analysis

A dedicated termination analyzer predicts and diagnoses loop behavior:

```java
public class LoopTerminationAnalyzer {
    public TerminationPrediction analyzeLoop(LoopInfo loop, IterationSnapshot[] snapshots);
    public RootCauseAnalysis diagnoseInfiniteLoop(LoopInfo loop);
    public LoopTerminationErrorReport generateReport(LoopInfo loop);
}
```

Key features:
- **Prediction**: Estimates remaining iterations based on variable evolution
- **Root cause analysis**: Identifies which variables/conditions prevent termination
- **Multi-loop analysis**: Handles nested loops and their interactions
- **Error reporting**: Provides actionable feedback for debugging

### 3. Enhanced DAG-based Execution

The new execution framework replaces the broken convergence process with a robust DAG construction and caching mechanism:

```java
public class InferenceSession {
    private final DAGCache dagCache = new DAGCache();
    
    public ExecutionResult output(...) {
        // Build corrected DAG with caching (replaces broken initSubgraph)
        ForwardExecutionDAG dag = dagCache.getOrCompute(allRequired, () -> {
            ForwardExecutionDAGBuilder builder = new ForwardExecutionDAGBuilder(sameDiff);
            return builder.buildForwardDAG(allRequired);
        });
        
        // Execute with corrected ordering
        Map<String, SDValue> results = executeOperations(dag, processedPlaceholders,
                processedOtherPlaceholders, allRequired, listeners, at, batch);
    }
}
```

#### Execution Node Types

The DAG consists of different node types representing various execution stages:

```java
public class ExecutionNode {
    public enum ExecutionNodeType {
        VARIABLE_INIT,      // Initialize constants/variables
        PLACEHOLDER_SET,    // Set placeholder values
        OPERATION_EXEC,     // Execute operations
        CONTROL_FLOW        // Handle control flow operations
    }
}
```

#### Frame-Aware Execution Order

The new execution uses frame-aware topological ordering to handle control flow:

```java
List<ExecutionNode> executionOrder = dag.getFrameAwareExecutionOrder();

for (ExecutionNode node : executionOrder) {
    if (!node.isReadyToExecute(completedOps)) {
        Set<String> missing = new HashSet<>(node.getDependsOnOperations());
        missing.removeAll(completedOps);
        throw new IllegalStateException("Operation " + node.getOperationName() +
                " not ready. Missing dependencies: " + missing);
    }
    
    executeNode(node, variableValues, allRequired, listeners, at, batch);
    completedOps.add(node.getOperationName());
}
```

#### Control Flow Operation Handling

Special handling for control flow operations with proper frame management:

```java
// Switch operation with branch tracking
if (op instanceof Switch) {
    boolean predicateValue = predicate.getDouble(0) != 0.0;
    String branchTaken = predicateValue ? "RIGHT" : "LEFT";
    executionStatus = "SWITCH_" + branchTaken;
    detailedStatus = String.format("SWITCH decision: %s branch taken (frame: %s, iter: %d)",
            branchTaken, outputFrameIter.getFrame(), outputFrameIter.getIteration());
}

// Merge operation with multi-frame input resolution
if (op instanceof Merge) {
    // Strategy 1: Current frame lookup
    // Strategy 2: Cross-frame lookup for Enter operations  
    // Strategy 3: Look for alias mappings in dependency tracker
}

// Enter operation with cross-frame dependency creation
if (op instanceof Enter) {
    // Create explicit dependency aliases for cross-frame access
    ExecStep expectedStep = new ExecStep(ExecType.OP, outputVar, enterOutFrameIter);
    ExecStep actualStep = new ExecStep(ExecType.OP, e.getOwnName(), enterOutFrameIter);
    dt.createDependeeAlias(expectedStep, actualStep);
}
```

### 4. Cross-Frame Reference Management

Enhanced cross-frame reference handling with explicit dependency tracking:

```java
// Multi-frame input resolution for Merge operations
List<VarId> candidateInputs = new ArrayList<>();
List<SDValue> availableValues = new ArrayList<>();

for (String inputName : in) {
    SDValue foundValue = null;
    VarId foundVarId = null;
    
    // Strategy 1: Current frame lookup
    VarId currentFrameVid = outputFrameIter.toVarId(inputName);
    foundValue = getSdValue(currentFrameVid);
    
    // Strategy 2: Cross-frame lookup for Enter operations
    if (foundValue == null) {
        for (Map.Entry<VarId, SDValue> entry : nodeValueOutputs.entrySet()) {
            VarId storedVid = entry.getKey();
            if (storedVid.getFrame().equals(outputFrameIter.getFrame())) {
                String producerOp = findVariableProducer(storedVid.getVariable());
                if (producer != null && producer.getOp() instanceof Enter) {
                    // Found cross-frame reference
                }
            }
        }
    }
}
```

### 5. Enhanced Memory Management

The framework includes improved memory tracking to prevent double-frees:

```java
protected static Set<Long> freedArrays = new LinkedHashSet<>();

// Track array dependencies and lifecycle
private AbstractDependencyTracker<SDValue, Dep> arrayUseTracker = new HashDependencyTracker<>();

// Close arrays when no longer needed
if (arrayUseTracker.hasNewAllSatisfied()) {
    List<SDValue> canClose = arrayUseTracker.getNewAllSatisfiedList();
    for (SDValue value : canClose) {
        if (!freedArrays.contains(value.getTensorValue().getId()) && 
            sameDiff.isEnableCache()) {
            mmgr.release(value.getTensorValue());
            freedArrays.add(value.getTensorValue().getId());
        }
    }
}
```

### 6. Execution Visualization Integration

Comprehensive execution tracking for debugging:

```java
if (visualizationEnabled && visualizer != null) {
    visualizer.recordStep(
        ExecType.OP,
        op.getOwnName(),
        outputFrameIter,
        stepInputs,
        stepOutputs,
        executionStatus + " | " + detailedStatus
    );
    
    // Enhanced failure analysis for control flow operations
    if (op instanceof Switch || op instanceof Merge || op instanceof Enter ||
        op instanceof Exit || op instanceof NextIteration || op instanceof LoopCond) {
        visualizer.analyzeControlFlowFailure(op, opInputs, allIterInputs, 
            constAndPhInputs, outputFrameIter, nodeValueOutputs, e);
    }
}
```

### 7. Dependent Value Tracking

New utilities for debugging variable dependencies:

```java
public String getDependentValuesString(Map<String, SDValue> variableValues, String variableName) {
    Map<String, String> deps = getDependentValuesMap(variableValues, variableName);
    // Recursively collect all dependent values
}

private void collectDependentValues(Map<String, SDValue> variableValues, String varName,
                                   Map<String, String> result, Set<String> visited) {
    // Find the op that produces this variable and collect all inputs
}
```

## Implementation Structure

The implementation is organized into several key packages:

- `org.nd4j.autodiff.samediff.execution`: Core execution planning and DAG construction
  - `ForwardExecutionDAG`: Frame-aware execution graph
  - `ForwardExecutionDAGBuilder`: Builds optimized execution plans
  - `DAGCache`: Caches execution plans for reuse
  - `ExecutionNode`: Represents operations in the execution graph
- `org.nd4j.autodiff.samediff.internal`: Enhanced internal execution with frame tracking
  - `InferenceSession`: Main execution engine with corrected DAG handling
  - `AbstractDependencyTracker`: Tracks cross-frame dependencies
- `org.nd4j.autodiff.samediff`: Analysis classes for termination, evolution, and cross-references

## Consequences

### Advantages

1. **Correct graph initialization**: The new DAG builder replaces the broken `initSubgraph` convergence process
2. **Improved debuggability**: Developers can understand why loops don't terminate and how variables evolve
3. **Performance optimization**: DAG caching eliminates repeated graph construction overhead
4. **Robustness**: Automatic detection of common issues like infinite loops and numerical instability
5. **Better error messages**: Root cause analysis provides actionable feedback with dependency tracking
6. **Visualization**: Complex control flow becomes understandable through detailed execution traces
7. **Memory safety**: Improved tracking prevents double-frees and memory leaks

### Disadvantages

1. **Memory overhead**: Tracking variable evolution, cross-frame references, and DAG cache requires additional memory
2. **Analysis cost**: Initial DAG construction phase adds latency to first execution (mitigated by caching)
3. **Complexity**: The framework adds significant complexity to the codebase
4. **Learning curve**: Developers need to understand new concepts like frame contexts, cross-frame references, and execution nodes

### Migration Path

Existing SameDiff graphs will automatically benefit from:
- Corrected graph initialization (automatic)
- Loop termination analysis (opt-in via configuration)
- DAG caching and optimization (transparent)
- Improved error messages with dependency tracking (automatic)

New features like variable evolution tracking require explicit enablement:

```java
SameDiff sd = SameDiff.create();
sd.enableExecutionAnalysis(AnalysisLevel.FULL);
```

## Future Work

1. **Distributed execution**: Extend DAG planning for multi-device execution
2. **Advanced patterns**: Detect more complex patterns like periodic orbits
3. **Automatic fixing**: Suggest or apply fixes for common issues
4. **Integration with profiling**: Connect execution analysis with performance profiling
5. **Lazy evaluation**: Optimize DAG execution to compute only required values
6. **Incremental updates**: Support efficient re-execution when only inputs change
