package org.nd4j.autodiff.samediff;

import lombok.Data;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

@Data
public class OperationAnalysis {
    private String loopConditionOp;
    private List<String> exitOperations = new ArrayList<>();
    private List<String> switchOperations = new ArrayList<>();
    private List<String> nextIterationOperations = new ArrayList<>();
    private List<String> enterOperations = new ArrayList<>();
    private List<String> mergeOperations = new ArrayList<>();
    private Map<String, Integer> operationExecutionCounts = new HashMap<>();
    private Map<Integer, List<String>> recentExecutionHistory = new HashMap<>();
    private OperationInfo triggerOperationInfo;
}
