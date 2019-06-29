package org.deeplearning4j.rl4j.observation.support;

import org.deeplearning4j.rl4j.observation.Observation;
import org.deeplearning4j.rl4j.observation.SimpleObservation;
import org.deeplearning4j.rl4j.observation.transforms.ObservationTransform;
import org.nd4j.linalg.factory.Nd4j;

public class TestPreviousTransform implements ObservationTransform {

    private final boolean isReady;
    public boolean isResetCalled;
    public boolean getObservationCalled;
    public double getObservationInput;
    public boolean isReadyCalled;

    public TestPreviousTransform(boolean isReady) {
        this.isReady = isReady;
    }

    public void reset() {
        isResetCalled = true;
    }

    public Observation transform(Observation input) {
        getObservationCalled = true;
        getObservationInput = input.toNDArray().getDouble(0);
        return new SimpleObservation(input.toNDArray());
    }

    public boolean isReady() {
        isReadyCalled = true;
        return true;
    }
}