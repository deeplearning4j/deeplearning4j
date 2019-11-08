package org.deeplearning4j.rl4j.observation;

import org.nd4j.linalg.api.ndarray.INDArray;

public class Observation {
    // TODO: Presently only a dummy container. Will contain observation channels when done.

    private final INDArray obsArray;

    public Observation(INDArray obsArray) {
        this.obsArray = obsArray;
    }

    public INDArray toINDArray() {
        return obsArray;
    }
}
