package org.deeplearning4j.rl4j.support;

import lombok.AllArgsConstructor;
import lombok.Getter;
import lombok.Value;
import org.deeplearning4j.rl4j.learning.async.AsyncConfiguration;

@AllArgsConstructor
@Value
public class MockAsyncConfiguration implements AsyncConfiguration {

    private Integer seed;
    private int maxEpochStep;
    private int maxStep;
    private int numThread;
    private int nstep;
    private int targetDqnUpdateFreq;
    private int updateStart;
    private double rewardFactor;
    private double gamma;
    private double errorClamp;
}
