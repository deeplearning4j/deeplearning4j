package org.deeplearning4j.rl4j.environment;

import lombok.Value;

import java.util.Map;

@Value
public class StepResult {
    private Map<String, Object> channelsData;
    private double reward;
    private boolean terminal;
}
