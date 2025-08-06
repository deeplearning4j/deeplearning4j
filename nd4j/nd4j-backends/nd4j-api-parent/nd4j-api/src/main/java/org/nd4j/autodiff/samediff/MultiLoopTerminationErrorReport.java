package org.nd4j.autodiff.samediff;

import lombok.Data;

import java.util.HashMap;
import java.util.Map;

@Data
public class MultiLoopTerminationErrorReport {
    private Map<String, LoopTerminationErrorReport> individualReports = new HashMap<>();
    private CrossLoopAnalysis crossLoopAnalysis;
    private long totalAnalysisTime;
    private int totalLoopsAnalyzed;
}
