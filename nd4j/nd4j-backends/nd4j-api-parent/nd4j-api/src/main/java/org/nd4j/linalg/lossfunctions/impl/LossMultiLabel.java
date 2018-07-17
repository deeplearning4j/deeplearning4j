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
import onnx.OnnxProto3;
import org.nd4j.autodiff.functions.DifferentialFunction;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.imports.NoOpNameFoundException;
import org.nd4j.linalg.activations.IActivation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.Op;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.ILossFunction;
import org.nd4j.linalg.lossfunctions.LossUtil;
import org.nd4j.linalg.ops.transforms.Transforms;
import org.nd4j.linalg.primitives.Pair;
import org.nd4j.shade.jackson.annotation.JsonInclude;
import org.tensorflow.framework.AttrValue;
import org.tensorflow.framework.GraphDef;
import org.tensorflow.framework.NodeDef;

import javax.naming.OperationNotSupportedException;
import java.util.List;
import java.util.Map;

/**
 * Multi-Label-Loss Function, maybe more commonly known as BPMLL
 * <p>
 * This Loss function requires that the Labels are given as a multi-hot encoded vector. It doesn't require any special
 * Activation method, i.e. the network output doesn't have to be in any specific range.
 * <p>
 * The loss is calculated based on the classification difference on labels that the examples has, and those that it
 * doesn't have. Assume that each example has a set of labels, these labels are the positive set, the labels that do not
 * belong to the example are in the negative set. This loss function trains the network to produce a higher value for
 * labels that are in the positive set than those that are in the negative set.
 * <p>
 * For a more detailed explanation and the actual formulas, read the original paper by Zhang and Zhou. The
 * implementation on scoreArray is based on equation 3, while computeGradient is based on equation 11. The main
 * difference being that -(c_k - c_l) = (c_l - c_k) was used to simplify the calculations.
 * <p>
 * Min-Ling Zhang and Zhi-Hua Zhou, "Multilabel Neural Networks with Applications to Functional Genomics and Text
 * Categorization," in IEEE Transactions on Knowledge and Data Engineering, vol. 18, no. 10, pp. 1338-1351, Oct. 2006.
 * doi: 10.1109/TKDE.2006.162
 * *
 *
 * @author Paul Dubs
 */
@EqualsAndHashCode
@JsonInclude(JsonInclude.Include.NON_NULL)
@Getter
public class LossMultiLabel extends DifferentialFunction implements ILossFunction {


    public LossMultiLabel() {
    }

    private void calculate(INDArray labels, INDArray preOutput, IActivation activationFn, INDArray mask, INDArray scoreOutput, INDArray gradientOutput) {
        if (scoreOutput == null && gradientOutput == null) {
            throw new IllegalArgumentException("You have to provide at least one of scoreOutput or gradientOutput!");
        }
        if (labels.size(1) != preOutput.size(1)) {
            throw new IllegalArgumentException(
                    "Labels array numColumns (size(1) = " + labels.size(1) + ") does not match output layer"
                            + " number of outputs (nOut = " + preOutput.size(1) + ") ");

        }
        final INDArray postOutput = activationFn.getActivation(preOutput.dup(), true);

        final INDArray positive = labels;
        final INDArray negative = labels.eq(0.0);
        final INDArray normFactor = negative.sum(1).muli(positive.sum(1));


        long examples = positive.size(0);
        for (int i = 0; i < examples; i++) {
            final INDArray locCfn = postOutput.getRow(i);
            final long[] shape = locCfn.shape();

            final INDArray locPositive = positive.getRow(i);
            final INDArray locNegative = negative.getRow(i);
            final Double locNormFactor = normFactor.getDouble(i);

            final INDArray operandA = Nd4j.ones(shape[1], shape[0]).mmul(locCfn);
            final INDArray operandB = operandA.transpose();

            final INDArray pairwiseSub = Transforms.exp(operandA.sub(operandB));

            final INDArray selection = locPositive.transpose().mmul(locNegative);

            final INDArray classificationDifferences = pairwiseSub.muli(selection).divi(locNormFactor);

            if (scoreOutput != null) {
                if (mask != null) {
                    final INDArray perLabel = classificationDifferences.sum(0);
                    LossUtil.applyMask(perLabel, mask.getRow(i));
                    perLabel.sum(scoreOutput.getRow(i), 0);
                } else {
                    classificationDifferences.sum(scoreOutput.getRow(i), 0, 1);
                }
            }

            if (gradientOutput != null) {
                gradientOutput.getRow(i).assign(classificationDifferences.sum(0).addi(classificationDifferences.sum(1).transposei().negi()));
            }
        }

        if (gradientOutput != null) {
            gradientOutput.assign(activationFn.backprop(preOutput.dup(), gradientOutput).getFirst());
            //multiply with masks, always
            if (mask != null) {
                LossUtil.applyMask(gradientOutput, mask);
            }
        }
    }

    public INDArray scoreArray(INDArray labels, INDArray preOutput, IActivation activationFn, INDArray mask) {
        final INDArray scoreArr = Nd4j.create(labels.size(0), 1);
        calculate(labels, preOutput, activationFn, mask, scoreArr, null);
        return scoreArr;
    }

    @Override
    public double computeScore(INDArray labels, INDArray preOutput, IActivation activationFn, INDArray mask,
                               boolean average) {
        INDArray scoreArr = scoreArray(labels, preOutput, activationFn, mask);

        double score = scoreArr.sumNumber().doubleValue();

        if (average)
            score /= scoreArr.size(0);

        return score;
    }

    @Override
    public INDArray computeScoreArray(INDArray labels, INDArray preOutput, IActivation activationFn, INDArray mask) {
        INDArray scoreArr = scoreArray(labels, preOutput, activationFn, mask);
        return scoreArr.sum(1);
    }

    @Override
    public INDArray computeGradient(INDArray labels, INDArray preOutput, IActivation activationFn, INDArray mask) {
        if (labels.size(1) != preOutput.size(1)) {
            throw new IllegalArgumentException(
                    "Labels array numColumns (size(1) = " + labels.size(1) + ") does not match output layer"
                            + " number of outputs (nOut = " + preOutput.size(1) + ") ");

        }
        final INDArray grad = Nd4j.ones(labels.shape());
        calculate(labels, preOutput, activationFn, mask, null, grad);
        return grad;
    }

    @Override
    public Pair<Double, INDArray> computeGradientAndScore(INDArray labels,
                                                          INDArray preOutput, IActivation activationFn, INDArray mask, boolean average) {
        final INDArray scoreArr = Nd4j.create(labels.size(0), 1);
        final INDArray grad = Nd4j.ones(labels.shape());

        calculate(labels, preOutput, activationFn, mask, scoreArr, grad);

        double score = scoreArr.sumNumber().doubleValue();

        if (average)
            score /= scoreArr.size(0);

        return new Pair<>(score, grad);
    }

    @Override
    public String name() {
        return toString();
    }


    @Override
    public String toString() {
        return "LossMultiLabel";
    }


    @Override
    public SDVariable[] outputVariables() {
        throw new UnsupportedOperationException();
    }

    @Override
    public SDVariable[] outputVariables(String baseName) {
        throw new UnsupportedOperationException();
    }

    @Override
    public List<SDVariable> doDiff(List<SDVariable> f1) {
        throw new UnsupportedOperationException();
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
