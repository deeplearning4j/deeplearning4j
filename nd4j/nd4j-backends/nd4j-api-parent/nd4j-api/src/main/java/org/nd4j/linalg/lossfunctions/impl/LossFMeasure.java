package org.nd4j.linalg.lossfunctions.impl;

import lombok.EqualsAndHashCode;
import lombok.Getter;
import onnx.OnnxProto3;
import org.nd4j.autodiff.functions.DifferentialFunction;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.imports.NoOpNameFoundException;
import org.nd4j.linalg.api.ops.Op;
import org.nd4j.linalg.primitives.Pair;
import org.nd4j.linalg.activations.IActivation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.ILossFunction;
import org.nd4j.linalg.ops.transforms.Transforms;
import org.nd4j.shade.jackson.annotation.JsonProperty;
import org.tensorflow.framework.AttrValue;
import org.tensorflow.framework.GraphDef;
import org.tensorflow.framework.NodeDef;

import java.util.List;
import java.util.Map;

/**
 * F–measure loss function is a loss function design for training on imbalanced datasets.
 * Essentially, this loss function is a continuous approximation of the F_Beta evaluation measure, of which F_1 is
 * a special case.<br>
 * <br>
 * <b>Note</b>: this implementation differs from that described in the original paper by Pastor-Pellicer et al. in
 * one important way: instead of maximizing the F-measure loss function as presented there (equations 2/3), we minimize
 * 1.0 - FMeasure. Consequently, a score of 0 is the minimum possible value (optimal predictions) and a score of 1.0 is
 * the maximum possible value.<br>
 * <br>
 * This implementation supports 2 types of operation:<br>
 * - Binary: single output/label (Typically sigmoid activation function)<br>
 * - Binary: 2-output/label (softmax activation function + 1-hot labels)<br>
 * Note that the beta value can be configured via the constructor.<br>
 * <br>
 * The following situations are NOT currently supported, may be added in the future:<br>
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
@Getter
@EqualsAndHashCode
public class LossFMeasure extends DifferentialFunction implements ILossFunction {

    public static final double DEFAULT_BETA = 1.0;

    private final double beta;

    public LossFMeasure() {
        this(DEFAULT_BETA);
    }

    public LossFMeasure(@JsonProperty("beta") double beta) {
        if (beta <= 0) {
            throw new UnsupportedOperationException("Invalid value: beta must be > 0. Got: " + beta);
        }
        this.beta = beta;
    }


    @Override
    public double computeScore(INDArray labels, INDArray preOutput, IActivation activationFn, INDArray mask,
                    boolean average) {

        double[] d = computeScoreNumDenom(labels, preOutput, activationFn, mask, average);
        double numerator = d[0];
        double denominator = d[1];

        if (numerator == 0.0 && denominator == 0.0) {
            return 0.0;
        }

        return 1.0 - numerator / denominator;
    }

    private double[] computeScoreNumDenom(INDArray labels, INDArray preOutput, IActivation activationFn, INDArray mask,
                    boolean average) {
        INDArray output = activationFn.getActivation(preOutput.dup(), true);

        long n = labels.size(1);
        if (n != 1 && n != 2) {
            throw new UnsupportedOperationException(
                            "For binary classification: expect output size of 1 or 2. Got: " + n);
        }

        //First: determine positives and negatives
        INDArray isPositiveLabel;
        INDArray isNegativeLabel;
        INDArray pClass0;
        INDArray pClass1;
        if (n == 1) {
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

        if (mask != null) {
            isPositiveLabel = isPositiveLabel.mulColumnVector(mask);
            isNegativeLabel = isNegativeLabel.mulColumnVector(mask);
        }

        double tp = isPositiveLabel.mul(pClass1).sumNumber().doubleValue();
        double fp = isNegativeLabel.mul(pClass1).sumNumber().doubleValue();
        double fn = isPositiveLabel.mul(pClass0).sumNumber().doubleValue();

        double numerator = (1.0 + beta * beta) * tp;
        double denominator = (1.0 + beta * beta) * tp + beta * beta * fn + fp;

        return new double[] {numerator, denominator};
    }

    @Override
    public INDArray computeScoreArray(INDArray labels, INDArray preOutput, IActivation activationFn, INDArray mask) {
        throw new UnsupportedOperationException("Cannot compute score array for FMeasure loss function: loss is only "
                        + "defined for minibatches");
    }

    @Override
    public INDArray computeGradient(INDArray labels, INDArray preOutput, IActivation activationFn, INDArray mask) {
        double[] d = computeScoreNumDenom(labels, preOutput, activationFn, mask, false);
        double numerator = d[0];
        double denominator = d[1];

        if (numerator == 0.0 && denominator == 0.0) {
            //Zero score -> zero gradient
            return Nd4j.create(preOutput.shape());
        }

        double secondTerm = numerator / (denominator * denominator);

        INDArray dLdOut;
        if (labels.size(1) == 1) {
            //Single binary output case
            dLdOut = labels.mul(1 + beta * beta).divi(denominator).subi(secondTerm);
        } else {
            //Softmax case: the getColumn(1) here is to account for the fact that we're using prob(class1)
            // only in the score function; column(1) is equivalent to output for the single output case
            dLdOut = Nd4j.create(labels.shape());
            dLdOut.getColumn(1).assign(labels.getColumn(1).mul(1 + beta * beta).divi(denominator).subi(secondTerm));
        }

        //Negate relative to description in paper, as we want to *minimize* 1.0-fMeasure, which is equivalent to
        // maximizing fMeasure
        dLdOut.negi();

        INDArray dLdPreOut = activationFn.backprop(preOutput, dLdOut).getFirst();

        if (mask != null) {
            dLdPreOut.muliColumnVector(mask);
        }

        return dLdPreOut;
    }

    @Override
    public Pair<Double, INDArray> computeGradientAndScore(INDArray labels, INDArray preOutput, IActivation activationFn,
                    INDArray mask, boolean average) {
        //TODO optimize
        return new Pair<>(computeScore(labels, preOutput, activationFn, mask, average),
                        computeGradient(labels, preOutput, activationFn, mask));
    }

    /**
     * The opName of this function
     *
     * @return
     */
    @Override
    public String name() {
        return "floss";
    }

    @Override
    public String toString() {
        return "LossFMeasure(beta=" + beta + ")";
    }


    @Override
    public SDVariable[] outputVariables() {
        return new SDVariable[0];
    }

    @Override
    public SDVariable[] outputVariables(String baseName) {
        return new SDVariable[0];
    }

    @Override
    public List<SDVariable> doDiff(List<SDVariable> f1) {
        return null;
    }



    @Override
    public String opName() {
        return name();
    }

    @Override
    public Op.Type opType() {
        return Op.Type.CUSTOM;
    }

    @Override
    public void initFromTensorFlow(NodeDef nodeDef, SameDiff initWith, Map<String, AttrValue> attributesForNode, GraphDef graph) {

    }

    @Override
    public void initFromOnnx(OnnxProto3.NodeProto node, SameDiff initWith, Map<String, OnnxProto3.AttributeProto> attributesForNode, OnnxProto3.GraphProto graph) {

    }

    @Override
    public String onnxName() {
        throw new NoOpNameFoundException("No onnx op name found for " + opName());
    }

    @Override
    public String tensorflowName() {
        throw new NoOpNameFoundException("No tensorflow op name found for " + opName());
    }
}
