package org.deeplearning4j.rl4j.observation;

import org.nd4j.linalg.api.ndarray.INDArray;

public class VoidObservation implements Observation {

    private static final VoidObservation instance = new VoidObservation();

    public static VoidObservation getInstance() {
        return instance;
    }

    private VoidObservation() {

    }

    @Override
    public INDArray toNDArray() {
        return null;
    }
}
