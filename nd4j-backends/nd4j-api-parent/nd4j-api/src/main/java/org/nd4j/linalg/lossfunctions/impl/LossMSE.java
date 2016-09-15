package org.nd4j.linalg.lossfunctions.impl;

import lombok.EqualsAndHashCode;
import org.nd4j.linalg.api.ndarray.INDArray;

/**
 * Created by Alex on 08/08/2016.
 */
@EqualsAndHashCode
public class LossMSE extends LossL2normSq {

    @Override
    public double computeScore(INDArray labels, INDArray preOutput, String activationFn, INDArray mask, boolean average) {

        double score = super.computeScore(labels,preOutput,activationFn,mask,average);
        score /= (labels.size(1));
        return score;
    }

    @Override
    public INDArray computeScoreArray(INDArray labels, INDArray preOutput, String activationFn, INDArray mask) {
        INDArray scoreArr = super.computeScoreArray(labels,preOutput,activationFn,mask);
        return scoreArr.divi(labels.size(1));
    }

    @Override
    public INDArray computeGradient(INDArray labels, INDArray preOutput, String activationFn, INDArray mask) {
        INDArray gradients = super.computeGradient(labels,preOutput,activationFn,mask);
        return gradients.divi(labels.size(1));
    }

    @Override
    public String toString() {
        return "LossMSE()";
    }
}
