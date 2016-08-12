package org.deeplearning4j.rl4j.learning.async;

import lombok.AllArgsConstructor;
import lombok.Value;
import org.nd4j.linalg.api.ndarray.INDArray;

/**
 * @author rubenfiszel (ruben.fiszel@epfl.ch) on 8/5/16.
 */
@AllArgsConstructor
@Value
public class MiniTrans<A> {
    INDArray obs;
    A action;
    INDArray[] output;
    double reward;
}
