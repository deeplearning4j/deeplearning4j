/* ******************************************************************************
 * Copyright (c) 2019 Konduit K.K.
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
import org.nd4j.common.base.Preconditions;
import org.nd4j.linalg.activations.IActivation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.shape.OneHot;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.common.primitives.Pair;
import org.nd4j.shade.jackson.annotation.JsonInclude;
import org.nd4j.shade.jackson.annotation.JsonProperty;
import org.nd4j.shade.jackson.databind.annotation.JsonDeserialize;
import org.nd4j.shade.jackson.databind.annotation.JsonSerialize;

/**
 *
 * Sparse Multi-Class Cross Entropy loss function:<br>
 * L = sum_i actual_i * log( predicted_i )<br>
 * Note: this is the same loss function as {@link LossMCXENT}, the only difference being the format for the labels -
 * this loss function uses integer indices (zero indexed) for the loss array, whereas LossMCXENT uses the equivalent
 * one-hot representation
 *
 * @author Alex Black
 * @see LossNegativeLogLikelihood
 * @see LossMCXENT
 */
@EqualsAndHashCode(callSuper = true)
@JsonInclude(JsonInclude.Include.NON_NULL)
@Getter @Setter
public class LossSparseMCXENT extends LossMCXENT {
    private static final double DEFAULT_SOFTMAX_CLIPPING_EPSILON = 1e-10;

    public LossSparseMCXENT() {
        this(null);
    }

    /**
     * Multi-Class Cross Entropy loss function where each the output is (optionally) weighted/scaled by a flags scalar value.
     * Note that the weights array must be a row vector, of length equal to the labels/output dimension 1 size.
     * A weight vector of 1s should give identical results to no weight vector.
     *
     * @param weights Weights array (row vector). May be null.
     */
    public LossSparseMCXENT(INDArray weights) {
        this(DEFAULT_SOFTMAX_CLIPPING_EPSILON, weights);
    }

    /**
     * Multi-Class Cross Entropy loss function where each the output is (optionally) weighted/scaled by a fixed scalar value.
     * Note that the weights array must be a row vector, of length equal to the labels/output dimension 1 size.
     * A weight vector of 1s should give identical results to no weight vector.
     *
     * @param weights Weights array (row vector). May be null.
     */
    public LossSparseMCXENT(@JsonProperty("softmaxClipEps") double softmaxClipEps, @JsonProperty("weights") INDArray weights) {
        super(softmaxClipEps, weights);
    }

    protected INDArray sparseScoreArray(INDArray labels, INDArray preOutput, IActivation activationFn, INDArray mask) {
        INDArray oneHotLabels = toOneHot(labels, preOutput);
        return super.scoreArray(oneHotLabels, preOutput, activationFn, mask);
    }

    @Override
    public double computeScore(INDArray labels, INDArray preOutput, IActivation activationFn, INDArray mask,
                               boolean average) {
        INDArray oneHotLabels = toOneHot(labels, preOutput);
        return super.computeScore(oneHotLabels, preOutput, activationFn, mask, average);
    }

    @Override
    public INDArray computeScoreArray(INDArray labels, INDArray preOutput, IActivation activationFn, INDArray mask) {
        INDArray scoreArr = sparseScoreArray(labels, preOutput, activationFn, mask);
        return scoreArr.sum(true,1).muli(-1);
    }

    @Override
    public INDArray computeGradient(INDArray labels, INDArray preOutput, IActivation activationFn, INDArray mask) {
        INDArray oneHotLabels = toOneHot(labels, preOutput);
        return super.computeGradient(oneHotLabels, preOutput, activationFn, mask);
    }

    @Override
    public Pair<Double, INDArray> computeGradientAndScore(INDArray labels, INDArray preOutput, IActivation activationFn,
                                                          INDArray mask, boolean average) {
        INDArray oneHotLabels = toOneHot(labels, preOutput);
        return new Pair<>(super.computeScore(oneHotLabels, preOutput, activationFn, mask, average),
                super.computeGradient(oneHotLabels, preOutput, activationFn, mask));
    }

    private INDArray toOneHot(INDArray labels, INDArray preOutput){
        Preconditions.checkState(labels.size(-1) == 1, "Labels for LossSparseMCXENT should be an array of integers " +
                "with first dimension equal to minibatch size, and last dimension having size 1. Got labels array with shape %ndShape", labels);
        INDArray oneHotLabels = preOutput.ulike();
        Nd4j.exec(new OneHot(labels.reshape(labels.length()), oneHotLabels, (int)preOutput.size(-1)));
        return oneHotLabels;
    }


    @Override
    public String toString() {
        if (weights == null)
            return "LossSparseMCXENT()";
        return "LossSparseMCXENT(weights=" + weights + ")";
    }
}
