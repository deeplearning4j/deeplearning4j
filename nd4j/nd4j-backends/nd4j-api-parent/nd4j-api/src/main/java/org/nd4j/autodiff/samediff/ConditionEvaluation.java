package org.nd4j.autodiff.samediff;

import lombok.Data;

import java.util.HashMap;
import java.util.Map;

@Data
public class ConditionEvaluation {
    private int iteration;
    private String conditionOperation;
    private Object conditionValue;
    private String conditionSource;
    private boolean terminationTriggered;
    private Map<String, Object> inputValues = new HashMap<>();
    private String evaluationContext;
    private long timestamp;
}
