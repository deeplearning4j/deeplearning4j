package org.deeplearning4j.rl4j.support;

import org.deeplearning4j.rl4j.learning.async.AsyncConfiguration;

public class MockAsyncConfiguration implements AsyncConfiguration {

    private final int nStep;
    private final int maxEpochStep;

    public MockAsyncConfiguration(int nStep, int maxEpochStep) {
        this.nStep = nStep;

        this.maxEpochStep = maxEpochStep;
    }

    @Override
    public int getSeed() {
        return 0;
    }

    @Override
    public int getMaxEpochStep() {
        return maxEpochStep;
    }

    @Override
    public int getMaxStep() {
        return 0;
    }

    @Override
    public int getNumThread() {
        return 0;
    }

    @Override
    public int getNstep() {
        return nStep;
    }

    @Override
    public int getTargetDqnUpdateFreq() {
        return 0;
    }

    @Override
    public int getUpdateStart() {
        return 0;
    }

    @Override
    public double getRewardFactor() {
        return 0;
    }

    @Override
    public double getGamma() {
        return 0;
    }

    @Override
    public double getErrorClamp() {
        return 0;
    }
}
