package org.nd4j.autodiff.samediff;

import lombok.Data;

// Data classes for the error report structure
@Data
public class LoopTerminationErrorReport {
    private String frameName;
    private int iteration;
    private long timestamp;
    private TerminationType terminationType;
    private String triggerOperation;
    private String terminationReason;
    private boolean wasEarlyTermination;
    private String earlyTerminationCause;

    // Analysis sections
    private VariableStateAnalysis variableStateAnalysis;
    private OperationAnalysis operationAnalysis;
    private FrameContextInfo frameContext;
    private VariableEvolutionAnalysis variableEvolution;
    private PerformanceMetrics performanceMetrics;
    private RootCauseAnalysis rootCauseAnalysis;
    private VisualizationData visualizationData;

    // Loop-specific metrics
    private long loopExecutionTime;
    private int expectedIterations;
    private int maxIterationsObserved;
}
