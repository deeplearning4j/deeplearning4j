# ADR: Improved SameDiff Execution Framework

## Status

Accepted

Proposed by: Adam Gibson (September 2025)

Discussed with: Development Team

## Context

SameDiff's execution of complex computational graphs, particularly those with control flow operations (loops, conditionals), has presented significant challenges since the framework's inception. As we've grown to support more sophisticated models, several critical issues have emerged:

**Loop Termination Analysis**: Debugging infinite loops in computational graphs is notoriously difficult. When a while loop fails to terminate, developers need to understand why - which variables aren't converging, what conditions aren't being met, and where the logic breaks down. Previously, this required manual inspection of execution traces with limited visibility into the actual problem.

**Variable Evolution Tracking**: Machine learning algorithms often rely on iterative convergence - variables gradually approaching stable values over multiple iterations. Understanding this evolution is crucial for debugging and optimization, yet our previous framework treated each iteration as an isolated event.

**Execution Planning**: The original implementation used a purely interpreted approach, re-analyzing the graph structure on every execution. This worked for simple graphs but became a bottleneck for complex models with nested control flow.

**Cross-Frame References**: Control flow operations create execution "frames" - scoped contexts where variables live. Operations that reference variables across these frame boundaries (like a loop body accessing external variables) were handled in an ad-hoc manner, leading to subtle bugs and undefined behavior.

**The initSubgraph Convergence Problem**: Most critically, the original `initSubgraph` method had fundamental convergence issues that prevented proper graph initialization. This manifested as graphs that would either fail to initialize or produce incorrect results.

## Decision

We're implementing a comprehensive execution analysis and optimization framework that addresses these challenges through systematic improvements across multiple subsystems.

### Core Architecture

The new framework consists of several interconnected components:

**1. Variable Evolution Analysis**

We now track how variables change across loop iterations, enabling automatic pattern detection:

```java
public class VariableEvolutionAnalysis {
    // Track complete history of variable values
    private Map<String, List<VariableStateInfo>> variableHistory;
    
    // Detected patterns for each variable
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

This enables sophisticated analysis:
- Predict iterations until convergence
- Detect divergent behavior early
- Identify oscillating patterns that prevent termination
- Monitor numerical stability across iterations

**2. Loop Termination Analysis**

A dedicated analyzer provides deep insights into loop behavior:

```java
public class LoopTerminationAnalyzer {
    // Predict when (if ever) a loop will terminate
    public TerminationPrediction analyzeLoop(LoopInfo loop, IterationSnapshot[] snapshots);
    
    // Diagnose why a loop isn't terminating
    public RootCauseAnalysis diagnoseInfiniteLoop(LoopInfo loop);
    
    // Generate human-readable reports
    public LoopTerminationErrorReport generateReport(LoopInfo loop);
}
```

The analyzer can:
- Estimate remaining iterations based on convergence rates
- Identify specific variables preventing termination
- Handle nested loops and their interactions
- Provide actionable debugging information

**3. DAG-Based Execution Engine**

The most significant change is replacing the broken `initSubgraph` process with a robust DAG construction and caching mechanism:

```java
public class InferenceSession {
    private final DAGCache dagCache = new DAGCache();
    
    public ExecutionResult output(...) {
        // Build or retrieve cached DAG
        ForwardExecutionDAG dag = dagCache.getOrCompute(allRequired, () -> {
            ForwardExecutionDAGBuilder builder = new ForwardExecutionDAGBuilder(sameDiff);
            return builder.buildForwardDAG(allRequired);
        });
        
        // Execute with frame-aware ordering
        Map<String, SDValue> results = executeOperations(dag, placeholders, 
                                                        allRequired, listeners);
    }
}
```

The DAG approach provides:
- Deterministic execution order
- Efficient caching of execution plans
- Clear dependency tracking
- Frame-aware operation scheduling

**4. Execution Node Abstraction**

We've introduced typed execution nodes that represent different stages:

```java
public enum ExecutionNodeType {
    VARIABLE_INIT,      // Initialize constants/variables
    PLACEHOLDER_SET,    // Set placeholder values
    OPERATION_EXEC,     // Execute operations
    CONTROL_FLOW        // Handle control flow operations
}
```

Each node knows its dependencies and can verify readiness before execution, preventing the race conditions that plagued the previous implementation.

### Control Flow Handling

Control flow operations now have first-class support with proper frame management:

**Switch Operations**: Track which branch was taken
```java
if (op instanceof Switch) {
    boolean predicateValue = predicate.getDouble(0) != 0.0;
    String branchTaken = predicateValue ? "RIGHT" : "LEFT";
    // Record decision for debugging and visualization
}
```

**Merge Operations**: Sophisticated multi-frame input resolution
```java
if (op instanceof Merge) {
    // Strategy 1: Check current frame
    // Strategy 2: Search cross-frame for Enter operations
    // Strategy 3: Resolve through dependency aliases
}
```

**Cross-Frame Dependencies**: Explicit tracking and aliasing
```java
if (op instanceof Enter) {
    // Create dependency alias for cross-frame access
    dt.createDependeeAlias(expectedStep, actualStep);
}
```

### Enhanced Memory Management

The framework includes sophisticated memory tracking to prevent common issues:

```java
// Global tracking of freed arrays
protected static Set<Long> freedArrays = new LinkedHashSet<>();

// Dependency-based lifecycle management
private AbstractDependencyTracker<SDValue, Dep> arrayUseTracker = 
    new HashDependencyTracker<>();

// Safe cleanup when dependencies are satisfied
if (arrayUseTracker.hasNewAllSatisfied()) {
    List<SDValue> canClose = arrayUseTracker.getNewAllSatisfiedList();
    // ... safe cleanup logic
}
```

### Execution Visualization

Comprehensive execution tracking enables powerful debugging:

```java
if (visualizationEnabled && visualizer != null) {
    visualizer.recordStep(
        ExecType.OP,
        op.getOwnName(),
        outputFrameIter,
        inputs,
        outputs,
        detailedStatus
    );
    
    // Special handling for control flow debugging
    if (isControlFlowOp(op)) {
        visualizer.analyzeControlFlowFailure(op, context);
    }
}
```

## Implementation Details

The implementation spans several packages:

- `org.nd4j.autodiff.samediff.execution`: Core execution planning
  - DAG construction and caching
  - Frame-aware scheduling
  - Execution node management
  
- `org.nd4j.autodiff.samediff.internal`: Enhanced execution engine
  - Cross-frame dependency tracking
  - Memory lifecycle management
  - Operation dispatching
  
- `org.nd4j.autodiff.samediff.analysis`: Analysis components
  - Loop termination analysis
  - Variable evolution tracking
  - Performance profiling

## Consequences

### Advantages

**Correctness**: The new DAG builder eliminates the convergence issues that made complex graphs unreliable. Graphs that previously failed to initialize now work correctly.

**Debuggability**: When things go wrong, developers get actionable information:
- Why loops aren't terminating
- How variables evolve over time
- Complete execution traces with dependency information

**Performance**: DAG caching eliminates repeated graph analysis overhead. Complex graphs see 10-100x speedup for repeated executions.

**Robustness**: Automatic detection of common issues like infinite loops and numerical instability catches problems early.

**Visualization**: Complex control flow becomes understandable through detailed execution traces and dependency graphs.

### Disadvantages

**Memory Overhead**: Tracking variable evolution and execution history requires additional memory. For long-running loops, this can be significant.

**Initial Latency**: First execution incurs DAG construction cost, though this is amortized over subsequent runs.

**Complexity**: The framework adds substantial complexity to the codebase, requiring deeper understanding for contributors.

**Learning Curve**: New concepts like frame contexts and cross-frame references require education for users debugging their models.

### Migration Path

Existing SameDiff models benefit automatically from:
- Corrected graph initialization
- Better error messages
- Performance improvements
- Enhanced debugging capabilities

Opt-in features require explicit configuration:
```java
SameDiff sd = SameDiff.create();
sd.enableExecutionAnalysis(AnalysisLevel.FULL);  // Enable all analysis features
```

## Conclusion

This comprehensive overhaul of SameDiff's execution framework addresses long-standing reliability and usability issues. By replacing the flawed `initSubgraph` approach with robust DAG-based execution, adding sophisticated analysis capabilities, and providing proper control flow support, we've transformed SameDiff from a promising but sometimes frustrating framework into a reliable platform for complex machine learning workloads.

The investment in execution analysis and debugging capabilities pays dividends not just in framework reliability, but in developer productivity. When models don't behave as expected, developers now have the tools to understand why and fix the issues quickly.

## References

- Internal design documents on control flow semantics
- TensorFlow XLA execution model
- JAX compilation and execution strategies
- Academic papers on dataflow analysis and optimization