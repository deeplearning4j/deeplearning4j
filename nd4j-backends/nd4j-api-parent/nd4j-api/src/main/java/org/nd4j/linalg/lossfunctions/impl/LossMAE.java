package org.nd4j.linalg.lossfunctions.impl;

import lombok.EqualsAndHashCode;
import org.nd4j.linalg.activations.IActivation;
import org.nd4j.linalg.api.ndarray.INDArray;

/**
 * Mean absolute error loss function: L = 1/N sum_i abs(predicted_i - actual_i)
 * See also {@link LossL1} for a mathematically similar loss function (LossL1 does not have division by N, where N is output size)
 *
 * @author Susan Eraly
 */
@EqualsAndHashCode(callSuper = true)
public class LossMAE extends LossL1 {

    public LossMAE() {

    }

    /**
     * Mean Absolute Error loss function where each the output is (optionally) weighted/scaled by a fixed scalar value.
     * Note that the weights array must be a row vector, of length equal to the labels/output dimension 1 size.
     * A weight vector of 1s should give identical results to no weight vector.
     *
     * @param weights Weights array (row vector). May be null.
     */
    public LossMAE(INDArray weights) {
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
        scoreArr.divi(scoreArr.size(1));
        return scoreArr;
    }

    @Override
    public INDArray computeGradient(INDArray labels, INDArray preOutput, IActivation activationFn, INDArray mask) {
        INDArray gradients = super.computeGradient(labels, preOutput, activationFn, mask);
        gradients.divi(labels.size(1));
        return gradients;
    }

    @Override
    public String toString() {
        if (weights == null)
            return "LossMAE()";
        return "LossMAE(weights=" + weights + ")";
    }
}
