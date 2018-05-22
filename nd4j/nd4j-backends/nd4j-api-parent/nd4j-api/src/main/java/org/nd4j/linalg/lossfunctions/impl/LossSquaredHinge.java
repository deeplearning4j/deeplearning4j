package org.nd4j.linalg.lossfunctions.impl;

import lombok.EqualsAndHashCode;
import onnx.OnnxProto3;
import org.nd4j.autodiff.functions.DifferentialFunction;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.imports.NoOpNameFoundException;
import org.nd4j.linalg.api.ops.Op;
import org.nd4j.linalg.primitives.Pair;
import org.nd4j.linalg.activations.IActivation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.indexing.BooleanIndexing;
import org.nd4j.linalg.indexing.conditions.Conditions;
import org.nd4j.linalg.lossfunctions.ILossFunction;
import org.nd4j.linalg.lossfunctions.LossUtil;
import org.tensorflow.framework.AttrValue;
import org.tensorflow.framework.GraphDef;
import org.tensorflow.framework.NodeDef;

import java.util.List;
import java.util.Map;

/**
 * Created by susaneraly on 9/9/16.
 */
@EqualsAndHashCode
public class LossSquaredHinge extends DifferentialFunction implements ILossFunction {

    public INDArray scoreArray(INDArray labels, INDArray preOutput, IActivation activationFn, INDArray mask) {
        if (labels.size(1) != preOutput.size(1)) {
            throw new IllegalArgumentException(
                            "Labels array numColumns (size(1) = " + labels.size(1) + ") does not match output layer"
                                            + " number of outputs (nOut = " + preOutput.size(1) + ") ");

        }
        /* y_hat is -1 or 1
        hinge loss is max(0,1-y_hat*y)
         */
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
        scoreArr.muli(scoreArr);
        return scoreArr.sum(1);
    }

    @Override
    public INDArray computeGradient(INDArray labels, INDArray preOutput, IActivation activationFn, INDArray mask) {
        if (labels.size(1) != preOutput.size(1)) {
            throw new IllegalArgumentException(
                            "Labels array numColumns (size(1) = " + labels.size(1) + ") does not match output layer"
                                            + " number of outputs (nOut = " + preOutput.size(1) + ") ");

        }
        INDArray scoreArr = scoreArray(labels, preOutput, activationFn, mask);

        INDArray bitMaskRowCol = scoreArr.dup();
        /*
            bit mask is 0 if 1-sigma(y*yhat) is neg, bit mask is 1 if 1-sigma(y*yhat) is +ve
         */
        BooleanIndexing.replaceWhere(bitMaskRowCol, 0.0, Conditions.lessThan(0.0));
        BooleanIndexing.replaceWhere(bitMaskRowCol, 1.0, Conditions.greaterThan(0.0));

        INDArray dLda = scoreArr.muli(2).muli(labels.neg());
        dLda.muli(bitMaskRowCol);

        if (mask != null && LossUtil.isPerOutputMasking(dLda, mask)) {
            //For *most* activation functions: we don't actually need to mask dL/da in addition to masking dL/dz later
            //but: some, like softmax, require both (due to dL/dz_i being a function of dL/da_j, for i != j)
            //We could add a special case for softmax (activationFn instanceof ActivationSoftmax) but that would be
            // error prone - though buy us a tiny bit of performance
            LossUtil.applyMask(dLda, mask);
        }

        INDArray gradients = activationFn.backprop(preOutput, dLda).getFirst(); //TODO activation functions with params

        if (mask != null) {
            LossUtil.applyMask(gradients, mask);
        }

        return gradients;
    }

    @Override
    public Pair<Double, INDArray> computeGradientAndScore(INDArray labels,
                    INDArray preOutput, IActivation activationFn, INDArray mask, boolean average) {
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
        return "LossSquaredHinge()";
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
