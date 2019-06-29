package org.deeplearning4j.rl4j.observation.support;

import org.deeplearning4j.rl4j.observation.Observation;
import org.deeplearning4j.rl4j.observation.SimpleObservation;
import org.deeplearning4j.rl4j.observation.transforms.ObservationTransform;
import org.deeplearning4j.rl4j.observation.transforms.PassthroughTransform;

public class TestPassthroughTransform extends PassthroughTransform {

    private final boolean isReady;
    public boolean isResetCalled;
    public boolean isHandledCalled = false;
    public boolean getIsReadyCalled = false;
    public double handleInput;

    public TestPassthroughTransform(boolean isReady) {
        super();
        this.isReady = isReady;
    }

    @Override
    protected void performReset() {
        isResetCalled = true;
    }

    @Override
    protected Observation handle(Observation input) {
        isHandledCalled = true;
        handleInput = input.toNDArray().getDouble(0);
        return new SimpleObservation(input.toNDArray().addi(1.0));
    }

    @Override
    protected boolean getIsReady() {
        getIsReadyCalled = true;
        return isReady;
    }
}
