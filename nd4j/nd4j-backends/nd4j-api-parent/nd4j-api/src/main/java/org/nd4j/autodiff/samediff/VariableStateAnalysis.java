package org.nd4j.autodiff.samediff;

import lombok.Data;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

@Data
public class VariableStateAnalysis {
    private Map<String, VariableStateInfo> currentVariables = new HashMap<>();
    private Map<String, VariableStateInfo> inputVariables = new HashMap<>();
    private Map<String, VariableStateInfo> outputVariables = new HashMap<>();
    private List<String> loopVariables = new ArrayList<>();
    private List<String> loopConstants = new ArrayList<>();
    private List<String> invariantVariables = new ArrayList<>();
    private Map<String, List<String>> problematicVariables = new HashMap<>();
    private Map<String, List<String>> variableDependencies = new HashMap<>();
}
