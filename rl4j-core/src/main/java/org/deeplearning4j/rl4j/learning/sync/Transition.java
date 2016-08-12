package org.deeplearning4j.rl4j.learning.sync;

import lombok.Value;
import org.nd4j.linalg.api.ndarray.INDArray;

/**
 * @author rubenfiszel (ruben.fiszel@epfl.ch) 7/12/16.
 */
@Value
public class Transition<A> {

    INDArray[] observation;
    A action;
    double reward;
    boolean isTerminal;
    INDArray[] nextObservation;

}
