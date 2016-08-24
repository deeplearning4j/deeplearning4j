package org.deeplearning4j.rl4j.learning.async;

import lombok.AllArgsConstructor;
import lombok.Value;
import org.nd4j.linalg.api.ndarray.INDArray;

/**
 * @author rubenfiszel (ruben.fiszel@epfl.ch) on 8/5/16.
 *
 * Its called a MiniTrans because it is similar to a Transition
 * but without a next observation
 *
 * It is stacked and then processed by AsyncNStepQL or A3C
 * following the paper implementation https://arxiv.org/abs/1602.01783 paper.
 *
 */
@AllArgsConstructor
@Value
public class MiniTrans<A> {
    INDArray obs;
    A action;
    INDArray[] output;
    double reward;
}
