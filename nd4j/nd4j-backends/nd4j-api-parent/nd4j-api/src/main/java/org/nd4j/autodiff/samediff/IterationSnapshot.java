package org.nd4j.autodiff.samediff;

import lombok.Data;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

@Data
public class IterationSnapshot {
    private int iteration;
    private long timestamp;
    private Map<String, Object> variableValues = new HashMap<>();
    private Map<String, String> variableShapes = new HashMap<>();
    private List<String> executedOperations = new ArrayList<>();
    private boolean conditionEvaluated;
    private Object conditionValue;
    private String conditionSource;
    private Map<String, Object> debugInfo = new HashMap<>();
}
