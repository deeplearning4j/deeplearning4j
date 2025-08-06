package org.nd4j.autodiff.samediff;

import lombok.Data;

import java.util.HashMap;
import java.util.Map;

@Data
public class TerminationPrediction {
    private int predictedAtIteration;
    private int predictedTerminationIteration;
    private double confidence;
    private String predictionMethod;
    private String reasoning;
    private Map<String, Object> evidenceData = new HashMap<>();
}
