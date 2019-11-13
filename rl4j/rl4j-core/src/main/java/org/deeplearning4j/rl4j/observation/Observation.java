package org.deeplearning4j.rl4j.observation;

import org.deeplearning4j.rl4j.learning.Learning;
import org.deeplearning4j.rl4j.learning.sync.Transition;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.util.ArrayUtil;

public class Observation {
    // TODO: Presently only a dummy container. Will contain observation channels when done.

    private final INDArray[] obsArray;

    private INDArray hstack = null;

    public Observation(INDArray[] obsArray) {
        this.obsArray = obsArray;
    }

    public INDArray[] toINDArray() {
        return obsArray;
    }

    public INDArray toHStack() {
        if(hstack != null) {
            return hstack;
        }

        //concat the history into a single INDArray input
        hstack = Transition.concat(Transition.dup(obsArray));

        //if input is not 2d, you have to append that the batch is 1 length high
        if (hstack.shape().length > 2)
            hstack = hstack.reshape(Learning.makeShape(1, ArrayUtil.toInts(hstack.shape())));

        return hstack;
    }
}
