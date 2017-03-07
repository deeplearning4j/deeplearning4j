package org.nd4j.linalg.lossfunctions.impl;

import lombok.EqualsAndHashCode;
import org.apache.commons.math3.util.Pair;
import org.nd4j.linalg.activations.IActivation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.indexing.BooleanIndexing;
import org.nd4j.linalg.indexing.conditions.Conditions;
import org.nd4j.linalg.lossfunctions.ILossFunction;
import org.nd4j.linalg.lossfunctions.LossUtil;

/**
 * Created by susaneraly on 8/15/16.
 */
@EqualsAndHashCode
public class LossHinge implements ILossFunction {

    public INDArray scoreArray(INDArray labels, INDArray preOutput, IActivation activationFn, INDArray mask) {
        /* y_hat is -1 or 1
        hinge loss is max(0,1-y_hat*y)
         */
        //INDArray output = Nd4j.getExecutioner().execAndReturn(Nd4j.getOpFactory().createTransform(activationFn, preOutput.dup()));
        INDArray output = activationFn.getActivation(preOutput.dup(), true);

        INDArray scoreArr = output.muli(labels); //y*yhat
        scoreArr.rsubi(1.0); //1 - y*yhat

        if (mask != null) {
            LossUtil.applyMask(scoreArr, mask);
        }
        return scoreArr; // 1 - y*yhat
    }

    @Override
    public double computeScore(INDArray labels, INDArray preOutput, IActivation activationFn, INDArray mask,
                    boolean average) {
        INDArray scoreArr = computeScoreArray(labels, preOutput, activationFn, mask);
        double score = scoreArr.sumNumber().doubleValue();
        if (average)
            score /= scoreArr.size(0);
        return score;
    }

    @Override
    public INDArray computeScoreArray(INDArray labels, INDArray preOutput, IActivation activationFn, INDArray mask) {
        INDArray scoreArr = scoreArray(labels, preOutput, activationFn, mask);
        BooleanIndexing.replaceWhere(scoreArr, 0.0, Conditions.lessThan(0.0));//max(0,1-y*yhat)
        return scoreArr.sum(1);
    }

    @Override
    public INDArray computeGradient(INDArray labels, INDArray preOutput, IActivation activationFn, INDArray mask) {
        /*
        gradient is 0 if yhaty is >= 1
        else gradient is gradient of the loss function = (1-yhaty) wrt preOutput = -y*derivative_of_yhat wrt preout
        */

        INDArray bitMaskRowCol = scoreArray(labels, preOutput, activationFn, mask);
        /*
            bit mask is 0 if 1-sigma(y*yhat) is neg
            bit mask is 1 if 1-sigma(y*yhat) is +ve
         */
        BooleanIndexing.replaceWhere(bitMaskRowCol, 0.0, Conditions.lessThan(0.0));
        BooleanIndexing.replaceWhere(bitMaskRowCol, 1.0, Conditions.greaterThan(0.0));

        INDArray dLda = labels.neg().muli(bitMaskRowCol);

        if(mask != null && LossUtil.isPerOutputMasking(dLda, mask)){
            //For *most* activation functions: we don't actually need to mask dL/da in addition to masking dL/dz later
            //but: some, like softmax, require both (due to dL/dz_i being a function of dL/da_j, for i != j)
            //We could add a special case for softmax (activationFn instanceof ActivationSoftmax) but that would be
            // error prone - though buy us a tiny bit of performance
            LossUtil.applyMask(dLda, mask);
        }

        INDArray gradients = activationFn.backprop(preOutput, dLda).getFirst(); //TODO activation functions with parameters

        if (mask != null) {
            LossUtil.applyMask(gradients, mask);
        }

        return gradients;
    }

    @Override
    public org.apache.commons.math3.util.Pair<Double, INDArray> computeGradientAndScore(INDArray labels,
                    INDArray preOutput, IActivation activationFn, INDArray mask, boolean average) {
        //TODO: probably a more efficient way to do this...
        return new Pair<>(computeScore(labels, preOutput, activationFn, mask, average),
                        computeGradient(labels, preOutput, activationFn, mask));
    }

    @Override
    public String toString() {
        return "LossHinge()";
    }

}
