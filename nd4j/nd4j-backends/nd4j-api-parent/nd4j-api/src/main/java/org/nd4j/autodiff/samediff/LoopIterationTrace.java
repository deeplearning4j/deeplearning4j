package org.nd4j.autodiff.samediff;

import lombok.Data;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

@Data
public class LoopIterationTrace {
    private String frameName;
    private List<IterationSnapshot> iterations = new ArrayList<>();
    private Map<String, List<Object>> variableEvolution = new HashMap<>();
    private List<ConditionEvaluation> conditionEvaluations = new ArrayList<>();
    private Map<String, Integer> operationExecutionCounts = new HashMap<>();
}
