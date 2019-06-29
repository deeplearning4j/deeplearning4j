package org.deeplearning4j.rl4j.observation.support;

import org.deeplearning4j.rl4j.observation.Observation;
import org.deeplearning4j.rl4j.observation.SimpleObservation;
import org.deeplearning4j.rl4j.observation.transforms.ObservationTransform;

public class TestSourceTransform implements ObservationTransform {

    private final boolean isReady;
    public boolean isResetCalled;

    public TestSourceTransform(boolean isReady) {
        this.isReady = isReady;
    }

    public void reset() {
        isResetCalled = true;
    }

    public Observation transform(Observation input) {
        return new SimpleObservation(input.toNDArray().addi(1.0));
    }

    public boolean isReady() {
        return isReady;
    }
}

