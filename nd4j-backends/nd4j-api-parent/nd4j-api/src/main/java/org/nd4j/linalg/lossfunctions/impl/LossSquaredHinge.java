package org.nd4j.linalg.lossfunctions.impl;

import lombok.EqualsAndHashCode;
import org.apache.commons.math3.util.Pair;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.BooleanIndexing;
import org.nd4j.linalg.indexing.conditions.Conditions;
import org.nd4j.linalg.lossfunctions.ILossFunction;

/**
 * Created by susaneraly on 9/9/16.
 */
@EqualsAndHashCode
public class LossSquaredHinge implements ILossFunction {

    public INDArray scoreArray(INDArray labels, INDArray preOutput, String activationFn, INDArray mask) {
        /* y_hat is -1 or 1
        hinge loss is max(0,1-y_hat*y)
         */
        INDArray output = Nd4j.getExecutioner().execAndReturn(Nd4j.getOpFactory().createTransform(activationFn, preOutput.dup()));

        INDArray scoreArr = output.muli(labels); //y*yhat
        scoreArr.rsubi(1.0); //1 - y*yhat

        if (mask != null) {
            scoreArr.muliColumnVector(mask);
        }
        return scoreArr; // 1 - y*yhat
    }

    @Override
    public double computeScore(INDArray labels, INDArray preOutput, String activationFn, INDArray mask, boolean average) {
        INDArray scoreArr = computeScoreArray(labels, preOutput, activationFn, mask);
        double score = scoreArr.sumNumber().doubleValue();
        if (average) score /= scoreArr.size(0);
        return score;
    }

    @Override
    public INDArray computeScoreArray(INDArray labels, INDArray preOutput, String activationFn, INDArray mask) {
        INDArray scoreArr = scoreArray(labels, preOutput, activationFn, mask);
        BooleanIndexing.replaceWhere(scoreArr, 0.0, Conditions.lessThan(0.0));//max(0,1-y*yhat)
        scoreArr.muli(scoreArr);
        return scoreArr.sum(1);
    }

    @Override
    public INDArray computeGradient(INDArray labels, INDArray preOutput, String activationFn, INDArray mask) {
        INDArray sigmaPrimeZ = Nd4j.getExecutioner().execAndReturn(Nd4j.getOpFactory().createTransform(activationFn, preOutput.dup()).derivative());
        /*
        gradient is 0 if yhaty is >= 1
        else gradient is gradient of the loss function = (1-yhaty) wrt preOutput = -y*derivative_of_yhat wrt preout
        */

        INDArray scoreArr = scoreArray(labels, preOutput, activationFn, mask);

        INDArray bitMaskRowCol = scoreArr.dup();
        /*
            bit mask is 0 if 1-sigma(y*yhat) is neg, bit mask is 1 if 1-sigma(y*yhat) is +ve
         */
        BooleanIndexing.replaceWhere(bitMaskRowCol, 0.0, Conditions.lessThan(0.0));
        BooleanIndexing.replaceWhere(bitMaskRowCol, 1.0, Conditions.greaterThan(0.0));

        INDArray gradients = scoreArr.muli(2);
        gradients.muli(labels.neg());
        gradients.muli(bitMaskRowCol).muli(sigmaPrimeZ);

        if (mask != null) {
            gradients.muliColumnVector(mask);
        }

        return gradients;
    }

    @Override
    public org.apache.commons.math3.util.Pair<Double, INDArray> computeGradientAndScore(INDArray labels, INDArray preOutput, String activationFn, INDArray mask, boolean average) {
        //TODO: probably a more efficient way to do this...
        //Yes - will implement in round two. Just want to get done now.

        return new Pair<>(
                computeScore(labels, preOutput, activationFn, mask, average),
                computeGradient(labels, preOutput, activationFn, mask));
    }

    @Override
    public String toString() {
        return "LossSquaredHinge()";
    }
}
