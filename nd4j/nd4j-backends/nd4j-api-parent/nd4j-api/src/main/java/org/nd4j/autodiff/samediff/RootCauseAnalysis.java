package org.nd4j.autodiff.samediff;

import lombok.Data;

import java.util.ArrayList;
import java.util.List;

@Data
public class RootCauseAnalysis {
    private String primaryCause;
    private List<String> contributingFactors = new ArrayList<>();
    private List<String> recommendedActions = new ArrayList<>();
    private List<String> similarPatternsInHistory = new ArrayList<>();
    private double confidenceLevel;
}
