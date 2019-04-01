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

import java.util.List;
import java.util.Map;
import lombok.EqualsAndHashCode;
import onnx.OnnxProto3;
import org.nd4j.autodiff.functions.DifferentialFunction;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.base.Preconditions;
import org.nd4j.imports.NoOpNameFoundException;
import org.nd4j.linalg.activations.IActivation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.Op;
import org.nd4j.linalg.api.ops.impl.transforms.same.Sign;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.ILossFunction;
import org.nd4j.linalg.lossfunctions.LossUtil;
import org.nd4j.linalg.primitives.Pair;
import org.tensorflow.framework.AttrValue;
import org.tensorflow.framework.GraphDef;
import org.tensorflow.framework.NodeDef;

/**
 * Wasserstein loss function, which calculates the Wasserstein distance, also known as earthmover's distance.
 *
 * This is not necessarily a general purpose loss function, and is intended for use as a discriminator loss.
 *
 * When using in a discriminator, use a label of 1 for real and -1 for generated
 * instead of the 1 and 0 used in normal GANs.
 *
 * As described in <a href="https://papers.nips.cc/paper/5679-learning-with-a-wasserstein-loss.pdf">Learning with a Wasserstein Loss</a>
 *
 * @author Ryan Nett
 */
@EqualsAndHashCode(callSuper = true)
public class LossWasserstein extends DifferentialFunction implements ILossFunction {

    private INDArray scoreArray(INDArray labels, INDArray preOutput, IActivation activationFn, INDArray mask){
        if(!labels.equalShapes(preOutput)){
            Preconditions.throwEx("Labels and preOutput must have equal shapes: got shapes %s vs %s", labels.shape(), preOutput.shape());
        }

        INDArray output = activationFn.getActivation(preOutput.dup(), true);

        INDArray scoreArr = labels.mul(output);
        if (mask != null) {
            LossUtil.applyMask(scoreArr, mask);
        }
        return scoreArr;
    }

    @Override
    public double computeScore(INDArray labels, INDArray preOutput, IActivation activationFn, INDArray mask,
            boolean average) {
        INDArray scoreArr = scoreArray(labels, preOutput, activationFn, mask);

        double score = scoreArr.mean(1).sumNumber().doubleValue();

        if (average) {
            score /= scoreArr.size(0);
        }

        return score;
    }

    @Override
    public INDArray computeScoreArray(INDArray labels, INDArray preOutput, IActivation activationFn, INDArray mask) {
        INDArray scoreArr = scoreArray(labels, preOutput, activationFn, mask);
        return Nd4j.expandDims(scoreArr.mean(1), 1);
    }

    @Override
    public INDArray computeGradient(INDArray labels, INDArray preOutput, IActivation activationFn, INDArray mask) {
        if(!labels.equalShapes(preOutput)){
            Preconditions.throwEx("Labels and preOutput must have equal shapes: got shapes %s vs %s", labels.shape(), preOutput.shape());
        }

        INDArray dLda = labels.div(labels.size(1));

        if (mask != null && LossUtil.isPerOutputMasking(dLda, mask)) {
            LossUtil.applyMask(labels, mask);
        }

        INDArray grad = activationFn.backprop(preOutput, dLda).getFirst();

        if (mask != null) {
            LossUtil.applyMask(grad, mask);
        }

        return grad;
    }

    @Override
    public Pair<Double, INDArray> computeGradientAndScore(INDArray labels, INDArray preOutput, IActivation activationFn,
            INDArray mask, boolean average) {
        return new Pair<>(computeScore(labels, preOutput, activationFn, mask, average),
                computeGradient(labels, preOutput, activationFn, mask));
    }
    @Override
    public String name() {
        return toString();
    }

    @Override
    public String toString() {
        return "LossWasserstein()";
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
