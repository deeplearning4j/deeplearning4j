package org.deeplearning4j.rl4j.observation;

import org.nd4j.linalg.api.ndarray.INDArray;

public class BasicObservation implements Observation {

    private final INDArray value;

    public BasicObservation(INDArray value) {
        this.value = value;
    }

    @Override
    public INDArray toNDArray() {
        return value;
    }
}
