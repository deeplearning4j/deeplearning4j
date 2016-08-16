package org.nd4j.linalg.lossfunctions.impl;

import org.apache.commons.math3.util.Pair;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.ILossFunction;

/**
 * Created by susaneraly on 8/15/16.
 */
public class LossHinge implements ILossFunction {

    public INDArray scoreArray(INDArray labels, INDArray preOutput, String activationFn, INDArray mask) {
        /* y_hat is -1 or 1
        hinge loss is max(0,1-y_hat*y)
        since it's not differentiable
        use the Rennie and Srebro's version
         */
        INDArray scoreArr;
        INDArray postOutput = Nd4j.getExecutioner().execAndReturn(Nd4j.getOpFactory().createTransform(activationFn, preOutput.dup()));

        INDArray yhaty = labels.mul(postOutput);
        INDArray zeroMin = Nd4j.getExecutioner().execAndReturn(Nd4j.getOpFactory().createTransform("max",Nd4j.zeros(labels.shape()),yhaty));
        INDArray oneMax = Nd4j.getExecutioner().execAndReturn(Nd4j.getOpFactory().createTransform("max",Nd4j.ones(labels.shape()),yhaty));
        INDArray zeroMinOneMax = zeroMin.mul(oneMax);
        zeroMinOneMax = Nd4j.getExecutioner().execAndReturn(Nd4j.getOpFactory().createTransform("min",Nd4j.ones(labels.shape()),zeroMinOneMax));

        INDArray fullMask = zeroMinOneMax.sub(oneMax);
        fullMask.muli(zeroMin);
        fullMask.divi(yhaty);

        scoreArr = fullMask.mul(yhaty);

        if (mask != null) scoreArr.muliColumnVector(mask);
        return scoreArr;
    }

    @Override
    public double computeScore(INDArray labels, INDArray preOutput, String activationFn, INDArray mask, boolean average) {
        INDArray scoreArr = scoreArray(labels, preOutput, activationFn, mask);

        double score = scoreArr.sumNumber().doubleValue();

        if(average) score /= scoreArr.size(0);

        return score;
    }

    @Override
    public INDArray computeScoreArray(INDArray labels, INDArray preOutput, String activationFn, INDArray mask) {
        INDArray scoreArr = scoreArray(labels, preOutput, activationFn, mask);
        return scoreArr.sum(1);
    }

    @Override
    public INDArray computeGradient(INDArray labels, INDArray preOutput, String activationFn, INDArray mask) {
        INDArray postOutput = Nd4j.getExecutioner().execAndReturn(Nd4j.getOpFactory().createTransform(activationFn, preOutput.dup()));
        INDArray postOutDer = Nd4j.getExecutioner().execAndReturn(Nd4j.getOpFactory().createTransform(activationFn,preOutput.dup()).derivative());

        INDArray yhaty = labels.mul(postOutput);
        INDArray zeroMin = Nd4j.getExecutioner().execAndReturn(Nd4j.getOpFactory().createTransform("max",Nd4j.zeros(labels.shape()),yhaty));
        INDArray oneMax = Nd4j.getExecutioner().execAndReturn(Nd4j.getOpFactory().createTransform("max",Nd4j.ones(labels.shape()),yhaty));
        INDArray zeroMinOneMax = zeroMin.mul(oneMax);

        INDArray fullMask = zeroMinOneMax.sub(oneMax);

        INDArray gradients = labels.mul(-1).mul(postOutDer);
        //use Rennie and Srebro's smoothed out version


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
    public String toString(){
        return "LossHinge()";
    }
}
