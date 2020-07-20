package org.deeplearning4j.rl4j.support;

import org.deeplearning4j.rl4j.space.ObservationSpace;
import org.nd4j.linalg.api.ndarray.INDArray;

public class MockObservationSpace implements ObservationSpace {

    private final int[] shape;

    public MockObservationSpace() {
        this(new int[] { 1 });
    }

    public MockObservationSpace(int[] shape) {
        this.shape = shape;
    }

    @Override
    public String getName() {
        return null;
    }

    @Override
    public int[] getShape() {
        return shape;
    }

    @Override
    public INDArray getLow() {
        return null;
    }

    @Override
    public INDArray getHigh() {
        return null;
    }
}