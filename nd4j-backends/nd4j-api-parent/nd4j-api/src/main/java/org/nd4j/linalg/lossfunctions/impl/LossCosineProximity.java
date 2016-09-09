package org.nd4j.linalg.lossfunctions.impl;

import lombok.EqualsAndHashCode;
import org.apache.commons.math3.util.Pair;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.Accumulation;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;

/**
 * Created by susaneraly on 9/9/16.
 */
@EqualsAndHashCode
public class LossCosineProximity {

    public INDArray scoreArray(INDArray labels, INDArray preOutput, String activationFn, INDArray mask) {
        /*
         mean of -(y.dot(yhat)/||y||*||yhat||)
         */
        INDArray postOutput = Nd4j.getExecutioner().execAndReturn(Nd4j.getOpFactory().createTransform(activationFn, preOutput.dup()));

        double yhatmag = postOutput.norm2Number().doubleValue();
        double ymag = labels.norm2Number().doubleValue();

        INDArray scoreArr = postOutput.mul(labels);
        scoreArr.divi(yhatmag);
        scoreArr.divi(ymag);

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
        INDArray postOutDer = Nd4j.getExecutioner().execAndReturn(Nd4j.getOpFactory().createTransform(activationFn,preOutput.dup()).derivative());
        INDArray yhat = Nd4j.getExecutioner().execAndReturn(Nd4j.getOpFactory().createTransform(activationFn, preOutput.dup()));
        /*

        */
        double yL2norm = labels.norm2Number().doubleValue();
        double yL2normSq = yL2norm * yL2norm;

        double yhatL2norm = yhat.norm2Number().doubleValue();
        double yhatL2normSq = yhatL2norm * yhatL2norm;

        double yhatDotyL1norm = labels.mul(yhat).norm1Number().doubleValue();

        INDArray gradients = labels.mul(yhatL2normSq);
        gradients.subi(yhat.mul(yhatDotyL1norm));
        gradients.divi(yL2normSq);
        gradients.divi(yhatL2norm*yhatL2normSq);
        gradients.muli(postOutDer);
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
