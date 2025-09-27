package org.nd4j.autodiff.samediff;

import lombok.Data;

import java.util.HashMap;
import java.util.Map;

@Data
public class VisualizationData {
    private Map<String, String> variableEvolutionPlots = new HashMap<>();
    private String conditionEvaluationTimeline;
    private String memoryUsageVisualization;
    private String operationExecutionFlow;
}
