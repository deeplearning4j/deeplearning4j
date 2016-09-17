package org.deeplearning4j.rl4j.space;

import org.nd4j.linalg.api.ndarray.INDArray;

/**
 * @author rubenfiszel (ruben.fiszel@epfl.ch) on 8/11/16.
 *         An observation space contains the basic informations about the state space
 */
public interface ObservationSpace<O> {

    String getName();

    int[] getShape();

    INDArray getLow();

    INDArray getHigh();

}
