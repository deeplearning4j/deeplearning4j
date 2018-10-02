/*******************************************************************************
 * Copyright (c) 2015-2018 Skymind, Inc.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

package org.nd4j.linalg.lossfunctions.impl;

import lombok.EqualsAndHashCode;
import lombok.Getter;
import lombok.Setter;
import onnx.OnnxProto3;
import org.nd4j.autodiff.functions.DifferentialFunction;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.base.Preconditions;
import org.nd4j.linalg.api.ops.CustomOp;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.nd4j.linalg.api.ops.Op;
import org.nd4j.linalg.primitives.Pair;
import org.nd4j.linalg.activations.IActivation;
import org.nd4j.linalg.activations.impl.ActivationSoftmax;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.transforms.strict.LogSoftMax;
import org.nd4j.linalg.api.ops.impl.transforms.strict.TimesOneMinus;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.ILossFunction;
import org.nd4j.linalg.lossfunctions.LossUtil;
import org.nd4j.linalg.lossfunctions.serde.RowVectorDeserializer;
import org.nd4j.linalg.lossfunctions.serde.RowVectorSerializer;
import org.nd4j.linalg.ops.transforms.Transforms;
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
 * Binary cross entropy loss function
 * <a href="https://en.wikipedia.org/wiki/Cross_entropy#Cross-entropy_error_function_and_logistic_regression">
 * https://en.wikipedia.org/wiki/Cross_entropy#Cross-entropy_error_function_and_logistic_regression</a>
 * Labels are assumed to take values 0 or 1
 *
 * @author Susan Eraly
 */
@EqualsAndHashCode
@JsonInclude(JsonInclude.Include.NON_NULL)
@Getter @Setter
public class LossBinaryXENT extends DifferentialFunction implements ILossFunction {
    public static final double DEFAULT_CLIPPING_EPSILON = 1e-5;

    @JsonSerialize(using = RowVectorSerializer.class)
    @JsonDeserialize(using = RowVectorDeserializer.class)
    private final INDArray weights;

    private double clipEps;

    public LossBinaryXENT() {
        this(null);
    }

    /**
     * Binary cross entropy where each the output is
     * (optionally) weighted/scaled by a fixed scalar value.
     * Note that the weights array must be a row vector, of length equal to
     * the labels/output dimension 1 size.
     * A weight vector of 1s should give identical results to no weight vector.
     *
     * @param weights Weights array (row vector). May be null.
     */
    public LossBinaryXENT(INDArray weights) {
        this(DEFAULT_CLIPPING_EPSILON, weights);
    }

    /**
     * Binary cross entropy where each the output is
     * (optionally) weighted/scaled by a fixed scalar value.
     * Note that the weights array must be a row vector, of length equal to
     * the labels/output dimension 1 size.
     * A weight vector of 1s should give identical results to no weight vector.
     *
     * @param clipEps Epsilon value for clipping. Probabilities are clipped in range of [eps, 1-eps]. Default eps: 1e-5
     */
    public LossBinaryXENT(double clipEps){
        this(clipEps, null);
    }

    /**
     * Binary cross entropy where each the output is
     * (optionally) weighted/scaled by a fixed scalar value.
     * Note that the weights array must be a row vector, of length equal to
     * the labels/output dimension 1 size.
     * A weight vector of 1s should give identical results to no weight vector.
     *
     * @param clipEps Epsilon value for clipping. Probabilities are clipped in range of [eps, 1-eps]. Default eps: 1e-5
     * @param weights Weights array (row vector). May be null.
     */
    public LossBinaryXENT(@JsonProperty("clipEps") double clipEps, @JsonProperty("weights") INDArray weights){
        if (weights != null && !weights.isRowVector()) {
            throw new IllegalArgumentException("Weights array must be a row vector");
        }
        if(clipEps < 0 || clipEps > 0.5){
            throw new IllegalArgumentException("Invalid clipping epsilon value: epsilon should be >= 0 (but near zero)."
                    + "Got: " + clipEps);
        }

        this.clipEps = clipEps;
        this.weights = weights;
    }

    private INDArray scoreArray(INDArray labels, INDArray preOutput, IActivation activationFn, INDArray mask) {
        if(!labels.equalShapes(preOutput)){
            Preconditions.throwEx("Labels and preOutput must have equal shapes: got shapes %s vs %s", labels.shape(), preOutput.shape());
        }

        INDArray scoreArr;
        if (activationFn instanceof ActivationSoftmax) {
            //Use LogSoftMax op to avoid numerical issues when calculating score
            INDArray logsoftmax = Nd4j.getExecutioner().execAndReturn(new LogSoftMax(preOutput.dup()));
            scoreArr = logsoftmax.muli(labels);

        } else {
            //INDArray output = Nd4j.getExecutioner().execAndReturn(Nd4j.getOpFactory().createTransform(activationFn, preOutput.dup()));
            INDArray output = activationFn.getActivation(preOutput.dup(), true);
            if (clipEps > 0.0) {
                CustomOp op = DynamicCustomOp.builder("clipbyvalue")
                        .addInputs(output)
                        .callInplace(true)
                        .addFloatingPointArguments(clipEps, 1.0-clipEps)
                        .build();
                Nd4j.getExecutioner().exec(op);
            }
            scoreArr = Transforms.log(output, true).muli(labels);
            INDArray secondTerm = output.rsubi(1);
            Transforms.log(secondTerm, false);
            secondTerm.muli(labels.rsub(1));
            scoreArr.addi(secondTerm);
        }

        //Weighted loss function
        if (weights != null) {
            if (weights.length() != preOutput.size(1)) {
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
        if(!labels.equalShapes(preOutput)){
            Preconditions.throwEx("Labels and preOutput must have equal shapes: got shapes %s vs %s", labels.shape(), preOutput.shape());
        }

        INDArray output = activationFn.getActivation(preOutput.dup(), true);
        if (clipEps > 0.0) {
            CustomOp op = DynamicCustomOp.builder("clipbyvalue")
                    .addInputs(output)
                    .callInplace(true)
                    .addFloatingPointArguments(clipEps, 1.0-clipEps)
                    .build();
            Nd4j.getExecutioner().exec(op);
        }

        INDArray numerator = output.sub(labels);
        INDArray denominator = Nd4j.getExecutioner().execAndReturn(new TimesOneMinus(output)); // output * (1-output)
        INDArray dLda = numerator.divi(denominator);

        if (mask != null && LossUtil.isPerOutputMasking(dLda, mask)) {
            //For *most* activation functions: we don't actually need to mask dL/da in addition to masking dL/dz later
            //but: some, like softmax, require both (due to dL/dz_i being a function of dL/da_j, for i != j)
            //We could add a special case for softmax (activationFn instanceof ActivationSoftmax) but that would be
            // error prone - but buy us a tiny bit of performance
            LossUtil.applyMask(dLda, mask);
        }

        INDArray grad = activationFn.backprop(preOutput, dLda).getFirst(); //TODO activation functions with weights

        //Weighted loss function
        if (weights != null) {
            if (weights.length() != output.size(1)) {
                throw new IllegalStateException("Weights vector (length " + weights.length()
                                + ") does not match output.size(1)=" + output.size(1));
            }

            grad.muliRowVector(weights);
        }

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
    public String toString() {
        if (weights == null)
            return "LossBinaryXENT()";
        return "LossBinaryXENT(weights=" + weights + ")";
    }

    @Override
    public void initFromTensorFlow(NodeDef nodeDef, SameDiff initWith, Map<String, AttrValue> attributesForNode, GraphDef graph) {

    }

    @Override
    public void initFromOnnx(OnnxProto3.NodeProto node, SameDiff initWith, Map<String, OnnxProto3.AttributeProto> attributesForNode, OnnxProto3.GraphProto graph) {

    }



    @Override
    public String opName() {
        return "lossbinaryxent";
    }

    @Override
    public Op.Type opType() {
        return Op.Type.CUSTOM;
    }

    @Override
    public String onnxName() {
        return "SigmoidCrossEntropy";
    }

    @Override
    public String tensorflowName() {
        return "SigmoidCrossEntropy";
    }
}
