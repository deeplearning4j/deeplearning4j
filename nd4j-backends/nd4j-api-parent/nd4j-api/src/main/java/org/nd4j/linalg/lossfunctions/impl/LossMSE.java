package org.nd4j.linalg.lossfunctions.impl;

import lombok.EqualsAndHashCode;
import org.nd4j.linalg.activations.IActivation;
import org.nd4j.linalg.api.ndarray.INDArray;

/**
 * Mean Squared Error loss function: L = 1/N sum_i (actual_i - predicted)^2
 * See also {@link LossL2} for a mathematically similar loss function (LossL2 does not have division by N, where N is output size)
 *
 * @author Susan Eraly
 */
@EqualsAndHashCode(callSuper = true)
public class LossMSE extends LossL2 {

    public LossMSE() {}

    /**
     * Mean Squared Error loss function where each the output is (optionally) weighted/scaled by a fixed scalar value.
     * Note that the weights array must be a row vector, of length equal to the labels/output dimension 1 size.
     * A weight vector of 1s should give identical results to no weight vector.
     *
     * @param weights Weights array (row vector). May be null.
     */
    public LossMSE(INDArray weights) {
        super(weights);
    }

    @Override
    public double computeScore(INDArray labels, INDArray preOutput, IActivation activationFn, INDArray mask,
                    boolean average) {

        double score = super.computeScore(labels, preOutput, activationFn, mask, average);
        score /= (labels.size(1));
        return score;
    }

    @Override
    public INDArray computeScoreArray(INDArray labels, INDArray preOutput, IActivation activationFn, INDArray mask) {
        INDArray scoreArr = super.computeScoreArray(labels, preOutput, activationFn, mask);
        return scoreArr.divi(labels.size(1));
    }

    @Override
    public INDArray computeGradient(INDArray labels, INDArray preOutput, IActivation activationFn, INDArray mask) {
        INDArray gradients = super.computeGradient(labels, preOutput, activationFn, mask);
        return gradients.divi(labels.size(1));
    }

    @Override
    public String toString() {
        if (weights == null)
            return "LossMSE()";
        return "LossMSE(weights=" + weights + ")";
    }
}
