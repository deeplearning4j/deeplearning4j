package org.nd4j.linalg.lossfunctions.impl;

import org.apache.commons.math3.util.Pair;
import org.nd4j.linalg.activations.IActivation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.ILossFunction;
import org.nd4j.linalg.ops.transforms.Transforms;

/**
 * Created by Alex on 15/06/2017.
 */
public class LossFMeasure implements ILossFunction {

    /**
     * Cases:
     * - Binary: single output
     * - Binary: 2 outputs (softmax style)
     * - Multiple independent binary
     * - Multiclass (via micro or macro averaging) -> not supported; do this in a separate loss function
     *
     *
     */
    public enum OutputType {BINARY, BINARY_MULTILABEL};

    public static final double DEFAULT_BETA = 1.0;

    private final OutputType outputType;
    private final double beta;

    public LossFMeasure(OutputType outputType){
        this(outputType, DEFAULT_BETA);
    }

    public LossFMeasure(OutputType outputType, double beta){
        this.outputType = outputType;
        this.beta = beta;
    }


    @Override
    public double computeScore(INDArray labels, INDArray preOutput, IActivation activationFn, INDArray mask, boolean average) {

        double[] d = computeScoreNumDenom(labels, preOutput, activationFn, mask, average);
        double numerator = d[0];
        double denominator = d[1];

        if(numerator == 0.0 && denominator == 0.0){
            return 0.0;
        }

        return numerator / denominator;
    }

    private double[] computeScoreNumDenom(INDArray labels, INDArray preOutput, IActivation activationFn, INDArray mask, boolean average){

        INDArray output = activationFn.getActivation(preOutput.dup(), true);

        int n = labels.size(1);
        if(outputType == OutputType.BINARY && n != 1 && n != 2){
            throw new UnsupportedOperationException("For binary classification: expect output size of 1 or 2");
        } else if(outputType == OutputType.BINARY_MULTILABEL){
            throw new UnsupportedOperationException("binary multi-label not yet supported");
        }

        //First: determine positives and negatives
        INDArray isPositiveLabel;
        INDArray isNegativeLabel;
        INDArray pClass0;
        INDArray pClass1;
        if(n == 1){
            isPositiveLabel = labels;
            isNegativeLabel = Transforms.not(isPositiveLabel);
            pClass0 = output.rsub(1.0);
            pClass1 = output;
        } else {
            isPositiveLabel = labels.getColumn(1);
            isNegativeLabel = labels.getColumn(0);
            pClass0 = output.getColumn(0);
            pClass1 = output.getColumn(1);
        }

        if(mask != null){
            isPositiveLabel = isPositiveLabel.mulColumnVector(mask);
            isNegativeLabel = isNegativeLabel.mulColumnVector(mask);
        }

        double tp = isPositiveLabel.mul(pClass1).sumNumber().doubleValue();
        double fp = isNegativeLabel.mul(pClass1).sumNumber().doubleValue();
        double fn = isPositiveLabel.mul(pClass0).sumNumber().doubleValue();

        double numerator = (1.0 + beta * beta) * tp;
        double denominator = (1.0 + beta * beta) * tp + beta * beta * fn + fp;

        return new double[]{numerator, denominator};
    }

    @Override
    public INDArray computeScoreArray(INDArray labels, INDArray preOutput, IActivation activationFn, INDArray mask) {
        throw new UnsupportedOperationException("Cannot compute score array for FMeasure loss function: loss is only " +
                "defined for minibatches");
    }

    @Override
    public INDArray computeGradient(INDArray labels, INDArray preOutput, IActivation activationFn, INDArray mask) {
        double[] d = computeScoreNumDenom(labels, preOutput, activationFn, mask, false);
        double numerator = d[0];
        double denominator = d[1];

        if(numerator == 0.0 && denominator == 0.0){
            //Zero score -> zero gradient
            return Nd4j.create(preOutput.shape());
        }

        double secondTerm = numerator / (denominator * denominator);

        INDArray grad = labels.mul(1+beta*beta).divi(denominator).subi(secondTerm);

        if(mask != null){
            grad.muliColumnVector(mask);
        }

        return grad;
    }

    @Override
    public Pair<Double, INDArray> computeGradientAndScore(INDArray labels, INDArray preOutput, IActivation activationFn, INDArray mask, boolean average) {
        //TODO optimize to reduce 
        return new Pair<>(
                computeScore(labels, preOutput, activationFn, mask, average),
                computeGradient(labels, preOutput, activationFn, mask));
    }
}
