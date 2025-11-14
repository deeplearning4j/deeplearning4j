package org.nd4j.autodiff.samediff;

import lombok.Data;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

@Data
public class LoopState {
    private int iteration;
    private Map<String, Object> variableStates = new HashMap<>();
    private Map<String, String> operationStates = new HashMap<>();
    private List<String> activeOperations = new ArrayList<>();
    private Map<String, Object> frameContext = new HashMap<>();
}
