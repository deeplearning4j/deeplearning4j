package org.nd4j.linalg.lossfunctions.impl;

import org.apache.commons.math3.util.Pair;
import org.nd4j.linalg.activations.IActivation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.ILossFunction;
import org.nd4j.linalg.ops.transforms.Transforms;
import org.nd4j.shade.jackson.annotation.JsonProperty;

/**
 * F–measure loss function is a loss function design for training on imbalanced datasets.
 * Essentially, this loss function is a continuous approximation of the F_Beta evaluation measure, of which F_1 is
 * a special case.<br>
 *
 * This implementation supports 2 types of operation:<br>
 * - Binary: single output/label (Typically sigmoid activation function)<br>
 * - Binary: 2-output/label (softmax activation function + 1-hot labels)<br>
 * Note that the beta value can be configured
 * <br>
 * The following situations are not currently supported, may be added in the future:
 * - Multi-label (multiple independent binary outputs)<br>
 * - Multiclass (via micro or macro averaging)<br>
 *
 * <br>
 * Reference: Pastor-Pellicer et al. (2013), F-Measure as the Error Function to Train Neural Networks,
 * <a href="https://link.springer.com/chapter/10.1007/978-3-642-38679-4_37">
 *     https://link.springer.com/chapter/10.1007/978-3-642-38679-4_37</a>
 *
 * @author Alex Black
 */
public class LossFMeasure implements ILossFunction {

    /**
     * Multiple modes are supported in this implementation of the F-Measure loss function.<br>
     * - BINARY: single output (sigmoid style) OR 2 outputs (softmax style)<br>
     * - BINARY_MULTILABEL: Multiple independent binary classes<br>
     */
    public enum OutputType {BINARY, BINARY_MULTILABEL};

    public static final double DEFAULT_BETA = 1.0;

    private final OutputType outputType;
    private final double beta;

    public LossFMeasure() {
        this(DEFAULT_BETA);
    }

    public LossFMeasure(double beta){
        this(OutputType.BINARY, beta);
    }

    public LossFMeasure(OutputType outputType){
        this(outputType, DEFAULT_BETA);
    }

    public LossFMeasure(@JsonProperty("outputType") OutputType outputType, @JsonProperty("beta") double beta){
        if(beta <= 0){
            throw new UnsupportedOperationException("Invalid value: beta must be >= 0. Got: " + beta);
        }
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
            throw new UnsupportedOperationException("For binary classification: expect output size of 1 or 2. Got: " + n);
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

        INDArray dLdOut = labels.mul(1+beta*beta).divi(denominator).subi(secondTerm);

        INDArray dLdPreOut = activationFn.backprop(preOutput, dLdOut).getFirst();

        if(mask != null){
            dLdPreOut.muliColumnVector(mask);
        }

        return dLdPreOut;
    }

    @Override
    public Pair<Double, INDArray> computeGradientAndScore(INDArray labels, INDArray preOutput, IActivation activationFn, INDArray mask, boolean average) {
        //TODO optimize
        return new Pair<>(
                computeScore(labels, preOutput, activationFn, mask, average),
                computeGradient(labels, preOutput, activationFn, mask));
    }

    @Override
    public String toString(){
        return "LossFMeasure(outputType=" + outputType + ")";
    }
}
