package org.nd4j.autodiff.samediff;

import lombok.Data;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

@Data
public class CrossLoopAnalysis {
    private Map<TerminationType, Long> terminationTypeDistribution = new HashMap<>();
    private List<String> terminationCorrelations = new ArrayList<>();
    private List<String> systemWideIssues = new ArrayList<>();
    private Map<String, Integer> commonProblematicVariables = new HashMap<>();
}
