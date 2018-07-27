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
import onnx.OnnxProto3;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.imports.NoOpNameFoundException;
import org.nd4j.linalg.activations.IActivation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.Op;
import org.tensorflow.framework.AttrValue;
import org.tensorflow.framework.GraphDef;
import org.tensorflow.framework.NodeDef;

import java.util.List;
import java.util.Map;

/**
 * Mean absolute error loss function: L = 1/N sum_i abs(predicted_i - actual_i)
 * See also {@link LossL1} for a mathematically similar loss function (LossL1 does not have division by N, where N is output size)
 *
 * @author Susan Eraly
 */
@EqualsAndHashCode(callSuper = true)
public class LossMAE extends LossL1 {

    public LossMAE() {

    }

    /**
     * Mean Absolute Error loss function where each the output is (optionally) weighted/scaled by a flags scalar value.
     * Note that the weights array must be a row vector, of length equal to the labels/output dimension 1 size.
     * A weight vector of 1s should give identical results to no weight vector.
     *
     * @param weights Weights array (row vector). May be null.
     */
    public LossMAE(INDArray weights) {
        super(weights);
    }

    @Override
    public double computeScore(INDArray labels, INDArray preOutput, IActivation activationFn, INDArray mask,
                    boolean average) {

        double score = super.computeScore(labels, preOutput, activationFn, mask, average);
        score /= (labels.size(1));
        return score;
    }

    @Override
    public INDArray computeScoreArray(INDArray labels, INDArray preOutput, IActivation activationFn, INDArray mask) {
        INDArray scoreArr = super.computeScoreArray(labels, preOutput, activationFn, mask);
        scoreArr.divi(scoreArr.size(1));
        return scoreArr;
    }

    @Override
    public INDArray computeGradient(INDArray labels, INDArray preOutput, IActivation activationFn, INDArray mask) {
        INDArray gradients = super.computeGradient(labels, preOutput, activationFn, mask);
        gradients.divi(labels.size(1));
        return gradients;
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
            return "LossMAE()";
        return "LossMAE(weights=" + weights + ")";
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
