package org.nd4j.linalg.lossfunctions.impl;

import lombok.EqualsAndHashCode;
import org.apache.commons.math3.util.Pair;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.ILossFunction;

/**
 * Created by susaneraly on 9/9/16.
 */
@EqualsAndHashCode
public class LossCosineProximity implements ILossFunction {

    public INDArray scoreArray(INDArray labels, INDArray preOutput, String activationFn, INDArray mask) {
        /*
         mean of -(y.dot(yhat)/||y||*||yhat||)
         */
        INDArray postOutput = Nd4j.getExecutioner().execAndReturn(Nd4j.getOpFactory().createTransform(activationFn, preOutput.dup()));

        INDArray yhatmag = postOutput.norm2(1);
        INDArray ymag = labels.norm2(1);

        INDArray scoreArr = postOutput.mul(labels);
        scoreArr.diviColumnVector(yhatmag);
        scoreArr.diviColumnVector(ymag);

        if (mask != null) scoreArr.muliColumnVector(mask);
        return scoreArr.muli(-1);
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

        INDArray yhat = Nd4j.getExecutioner().execAndReturn(Nd4j.getOpFactory().createTransform(activationFn, preOutput.dup()));
        INDArray postOutDer = Nd4j.getExecutioner().execAndReturn(Nd4j.getOpFactory().createTransform(activationFn,preOutput.dup()).derivative());
        /*

        */
        INDArray yL2norm = labels.norm2(1);
        INDArray yL2normSq = yL2norm.mul(yL2norm);

        INDArray yhatL2norm = yhat.norm2(1);
        INDArray yhatL2normSq = yhatL2norm.mul(yhatL2norm);

        //Note: This is not really the L1 norm since I am not taking abs values
        INDArray yhatDotyL1norm = labels.mul(yhat).sum(1);

        INDArray gradients = labels.mulColumnVector(yhatL2normSq);
        gradients.subi(yhat.mulColumnVector(yhatDotyL1norm));
        gradients.diviColumnVector(yL2norm);
        gradients.diviColumnVector(yhatL2norm.mul(yhatL2normSq));
        gradients.muli(postOutDer);
        gradients.muli(-1);

        if(mask != null){
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
    public String toString(){
        return "LossCosineProximity()";
    }
}
