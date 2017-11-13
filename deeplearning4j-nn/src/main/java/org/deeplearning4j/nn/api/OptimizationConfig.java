package org.deeplearning4j.nn.api;


public interface OptimizationConfig {

    boolean isMiniBatch();

    int getMaxNumLineSearchIterations();

    OptimizationAlgorithm getOptimizationAlgo();

    org.deeplearning4j.nn.conf.stepfunctions.StepFunction getStepFunction();

    // TODO: this is unused
    boolean isMinimize();

    int getIterationCount();

    int getEpochCount();

    void setIterationCount(int iterationCount);

    void setEpochCount(int epochCount);

}
