/*
 *  ******************************************************************************
 *  * Copyright (c) 2021 Deeplearning4j Contributors
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

package org.nd4j.linalg.lossfunctions.impl;

import lombok.EqualsAndHashCode;
import org.nd4j.linalg.activations.IActivation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.shade.jackson.annotation.JsonProperty;

/**
 * Mean Squared Error loss function: L = 1/N sum_i (actual_i - predicted)^2
 * See also {@link LossL2} for a mathematically similar loss function (LossL2 does not have division by N, where N is output size)
 *
 * @author Susan Eraly
 */
@EqualsAndHashCode(callSuper = true)
public class LossMSE extends LossL2 {

    public LossMSE() {}

    /**
     * Mean Squared Error loss function where each the output is (optionally) weighted/scaled by a flags scalar value.
     * Note that the weights array must be a row vector, of length equal to the labels/output dimension 1 size.
     * A weight vector of 1s should give identical results to no weight vector.
     *
     * @param weights Weights array (row vector). May be null.
     */
    public LossMSE(@JsonProperty("weights") INDArray weights) {
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
        return scoreArr.divi(labels.size(1));
    }

    @Override
    public INDArray computeGradient(INDArray labels, INDArray preOutput, IActivation activationFn, INDArray mask) {
        INDArray gradients = super.computeGradient(labels, preOutput, activationFn, mask);
        return gradients.divi(labels.size(1));
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
            return "LossMSE()";
        return "LossMSE(weights=" + weights + ")";
    }
}
