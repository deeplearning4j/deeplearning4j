package org.deeplearning4j.rl4j.support;

import org.deeplearning4j.rl4j.space.ObservationSpace;
import org.nd4j.linalg.api.ndarray.INDArray;

public class MockObservationSpace implements ObservationSpace {

    @Override
    public String getName() {
        return null;
    }

    @Override
    public int[] getShape() {
        return new int[] { 1 };
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
