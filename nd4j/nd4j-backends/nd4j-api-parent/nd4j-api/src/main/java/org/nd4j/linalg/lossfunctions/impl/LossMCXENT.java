package org.nd4j.linalg.lossfunctions.impl;


import lombok.EqualsAndHashCode;
import lombok.Getter;
import lombok.Setter;
import onnx.OnnxProto3;
import org.nd4j.autodiff.functions.DifferentialFunction;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.activations.IActivation;
import org.nd4j.linalg.activations.impl.ActivationSoftmax;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.Op;
import org.nd4j.linalg.indexing.BooleanIndexing;
import org.nd4j.linalg.indexing.conditions.Conditions;
import org.nd4j.linalg.lossfunctions.ILossFunction;
import org.nd4j.linalg.lossfunctions.LossUtil;
import org.nd4j.linalg.lossfunctions.serde.RowVectorDeserializer;
import org.nd4j.linalg.lossfunctions.serde.RowVectorSerializer;
import org.nd4j.linalg.ops.transforms.Transforms;
import org.nd4j.linalg.primitives.Pair;
import org.nd4j.shade.jackson.annotation.JsonInclude;
import org.nd4j.shade.jackson.annotation.JsonProperty;
import org.nd4j.shade.jackson.databind.annotation.JsonDeserialize;
import org.nd4j.shade.jackson.databind.annotation.JsonSerialize;
import org.tensorflow.framework.AttrValue;
import org.tensorflow.framework.GraphDef;
import org.tensorflow.framework.NodeDef;

import java.util.List;
import java.util.Map;

/**
 *
 * Multi-Class Cross Entropy loss function:<br>
 * L = sum_i actual_i * log( predicted_i )
 *
 * @author Alex Black, Susan Eraly
 * @see LossNegativeLogLikelihood
 */
@EqualsAndHashCode
@JsonInclude(JsonInclude.Include.NON_NULL)
@Getter @Setter
public class LossMCXENT extends DifferentialFunction implements ILossFunction {
    private static final double DEFAULT_SOFTMAX_CLIPPING_EPSILON = 1e-10;

    @JsonSerialize(using = RowVectorSerializer.class)
    @JsonDeserialize(using = RowVectorDeserializer.class)
    private INDArray weights;

    private double softmaxClipEps;

    public LossMCXENT() {
        this(null);
    }

    /**
     * Multi-Class Cross Entropy loss function where each the output is (optionally) weighted/scaled by a flags scalar value.
     * Note that the weights array must be a row vector, of length equal to the labels/output dimension 1 size.
     * A weight vector of 1s should give identical results to no weight vector.
     *
     * @param weights Weights array (row vector). May be null.
     */
    public LossMCXENT(INDArray weights) {
        this(DEFAULT_SOFTMAX_CLIPPING_EPSILON, weights);
    }

    /**
     * Multi-Class Cross Entropy loss function where each the output is (optionally) weighted/scaled by a fixed scalar value.
     * Note that the weights array must be a row vector, of length equal to the labels/output dimension 1 size.
     * A weight vector of 1s should give identical results to no weight vector.
     *
     * @param weights Weights array (row vector). May be null.
     */
    public LossMCXENT(@JsonProperty("softmaxClipEps") double softmaxClipEps, @JsonProperty("weights") INDArray weights) {
        if (weights != null && !weights.isRowVector()) {
            throw new IllegalArgumentException("Weights array must be a row vector");
        }
        if(softmaxClipEps < 0 || softmaxClipEps > 0.5){
            throw new IllegalArgumentException("Invalid clipping epsilon: epsilon should be >= 0 (but near zero). Got: "
                    + softmaxClipEps);
        }
        this.weights = weights;
        this.softmaxClipEps = softmaxClipEps;
    }

    private INDArray scoreArray(INDArray labels, INDArray preOutput, IActivation activationFn, INDArray mask) {
        if (labels.size(1) != preOutput.size(1)) {
            throw new IllegalArgumentException(
                            "Labels array numColumns (size(1) = " + labels.size(1) + ") does not match output layer"
                                            + " number of outputs (nOut = " + preOutput.size(1) + ") ");

        }

        INDArray output = activationFn.getActivation(preOutput.dup(), true);
        if(activationFn instanceof ActivationSoftmax && softmaxClipEps > 0.0){
            BooleanIndexing.replaceWhere(output, softmaxClipEps, Conditions.lessThan(softmaxClipEps));
            BooleanIndexing.replaceWhere(output, 1.0-softmaxClipEps, Conditions.greaterThan(1.0-softmaxClipEps));
        }
        INDArray scoreArr = Transforms.log(output, false).muli(labels);

        //Weighted loss function
        if (weights != null) {
            if (weights.length() != scoreArr.size(1)) {
                throw new IllegalStateException("Weights vector (length " + weights.length()
                                + ") does not match output.size(1)=" + preOutput.size(1));
            }
            scoreArr.muliRowVector(weights);
        }

        if (mask != null) {
            LossUtil.applyMask(scoreArr, mask);
        }
        return scoreArr;
    }

    @Override
    public double computeScore(INDArray labels, INDArray preOutput, IActivation activationFn, INDArray mask,
                    boolean average) {
        INDArray scoreArr = scoreArray(labels, preOutput, activationFn, mask);

        double score = -scoreArr.sumNumber().doubleValue();

        if (average) {
            score /= scoreArr.size(0);
        }

        return score;
    }

    @Override
    public INDArray computeScoreArray(INDArray labels, INDArray preOutput, IActivation activationFn, INDArray mask) {
        INDArray scoreArr = scoreArray(labels, preOutput, activationFn, mask);
        return scoreArr.sum(1).muli(-1);
    }

    @Override
    public INDArray computeGradient(INDArray labels, INDArray preOutput, IActivation activationFn, INDArray mask) {
        if (labels.size(1) != preOutput.size(1)) {
            throw new IllegalArgumentException(
                            "Labels array numColumns (size(1) = " + labels.size(1) + ") does not match output layer"
                                            + " number of outputs (nOut = " + preOutput.size(1) + ") ");

        }
        INDArray grad;
        //INDArray output = Nd4j.getExecutioner().execAndReturn(Nd4j.getOpFactory().createTransform(activationFn, preOutput.dup()));
        INDArray output = activationFn.getActivation(preOutput.dup(), true);

        if (activationFn instanceof ActivationSoftmax) {

            if (mask != null && LossUtil.isPerOutputMasking(output, mask)) {
                throw new UnsupportedOperationException("Per output masking for MCXENT + softmax: not supported");
            }

            //Weighted loss function
            if (weights != null) {
                if (weights.length() != output.size(1)) {
                    throw new IllegalStateException("Weights vector (length " + weights.length()
                                    + ") does not match output.size(1)=" + output.size(1));
                }
                INDArray temp = labels.mulRowVector(weights);
                INDArray col = temp.sum(1);
                grad = output.mulColumnVector(col).sub(temp);
            } else {
                grad = output.subi(labels);
            }
        } else {
            INDArray dLda = output.rdivi(labels).negi();

            grad = activationFn.backprop(preOutput, dLda).getFirst(); //TODO activation function with weights

            //Weighted loss function
            if (weights != null) {
                if (weights.length() != output.size(1)) {
                    throw new IllegalStateException("Weights vector (length " + weights.length()
                                    + ") does not match output.size(1)=" + output.size(1));
                }
                grad.muliRowVector(weights);
            }
        }

        //Loss function with masking
        if (mask != null) {
            LossUtil.applyMask(grad, mask);
        }

        return grad;
    }

    @Override
    public Pair<Double, INDArray> computeGradientAndScore(INDArray labels, INDArray preOutput, IActivation activationFn,
                    INDArray mask, boolean average) {
        //TODO: probably a more efficient way to do this...

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
        return toString();
    }


    @Override
    public String toString() {
        if (weights == null)
            return "LossMCXENT()";
        return "LossMCXENT(weights=" + weights + ")";
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
        return "lossmcxent";
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
        return "SoftmaxCrossEntropyWithLogits";
    }

    @Override
    public String tensorflowName() {
        return "SoftmaxCrossEntropyWithLogits";
    }
}
