package org.nd4j.autodiff.samediff;

import lombok.Data;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

@Data
public class VariableEvolutionAnalysis {
    private Map<String, List<Object>> variableEvolution = new HashMap<>();
    private Map<String, VariablePattern> detectedPatterns = new HashMap<>();
    private List<ConditionEvaluation> conditionEvaluationHistory = new ArrayList<>();
}
