package org.nd4j.autodiff.samediff;

import lombok.Data;

@Data
public class PerformanceMetrics {
    private long totalMemory;
    private long freeMemory;
    private long usedMemory;
    private long maxMemory;
    private long totalVariableMemory;
    private long loopExecutionTime;
    private double averageIterationTime;
    private double iterationsPerSecond;
}
