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
import org.nd4j.base.Preconditions;
import org.nd4j.linalg.activations.IActivation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.lossfunctions.ILossFunction;
import org.nd4j.linalg.lossfunctions.LossUtil;
import org.nd4j.linalg.ops.transforms.Transforms;
import org.nd4j.linalg.primitives.Pair;
import org.nd4j.serde.jackson.shaded.NDArrayTextDeSerializer;
import org.nd4j.serde.jackson.shaded.NDArrayTextSerializer;
import org.nd4j.shade.jackson.annotation.JsonInclude;
import org.nd4j.shade.jackson.databind.annotation.JsonDeserialize;
import org.nd4j.shade.jackson.databind.annotation.JsonSerialize;

/**
 * Mean Squared Logarithmic Error loss function: L = 1/N sum_i (log(1+predicted_i) - log(1+actual_i))^2
 *
 * @author Susan Eraly
 */
@EqualsAndHashCode
@JsonInclude(JsonInclude.Include.NON_NULL)
@Getter
public class LossMSLE implements ILossFunction {

    @JsonSerialize(using = NDArrayTextSerializer.class)
    @JsonDeserialize(using = NDArrayTextDeSerializer.class)
    private final INDArray weights;

    public LossMSLE() {
        this(null);
    }

    /**
     * Mean Squared Logarithmic Error loss function where each the output is (optionally) weighted/scaled by a flags scalar value.
     * Note that the weights array must be a row vector, of length equal to the labels/output dimension 1 size.
     * A weight vector of 1s should give identical results to no weight vector.
     *
     * @param weights Weights array (row vector). May be null.
     */
    public LossMSLE(INDArray weights) {
        if (weights != null && !weights.isRowVector()) {
            throw new IllegalArgumentException("Weights array must be a row vector");
        }
        this.weights = weights;
    }

    public INDArray scoreArray(INDArray labels, INDArray preOutput, IActivation activationFn, INDArray mask) {
        if(!labels.equalShapes(preOutput)){
            Preconditions.throwEx("Labels and preOutput must have equal shapes: got shapes %s vs %s", labels.shape(), preOutput.shape());
        }
        labels = labels.castTo(preOutput.dataType());   //No-op if already correct dtype
        INDArray scoreArr;
        INDArray output = activationFn.getActivation(preOutput.dup(), true);
        scoreArr = Transforms.log(output.addi(1.0).divi(labels.add(1.0)), false);
        scoreArr = scoreArr.muli(scoreArr).divi(labels.size(1));

        //Weighted loss function
        if (weights != null) {
            if (weights.length() != output.size(1)) {
                throw new IllegalStateException("Weights vector (length " + weights.length()
                                + ") does not match output.size(1)=" + output.size(1));
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

        double score = scoreArr.sumNumber().doubleValue();

        if (average)
            score /= scoreArr.size(0);

        return score;
    }

    @Override
    public INDArray computeScoreArray(INDArray labels, INDArray preOutput, IActivation activationFn, INDArray mask) {
        INDArray scoreArr = scoreArray(labels, preOutput, activationFn, mask);
        return scoreArr.sum(true,1);
    }

    @Override
    public INDArray computeGradient(INDArray labels, INDArray preOutput, IActivation activationFn, INDArray mask) {
        if(!labels.equalShapes(preOutput)){
            Preconditions.throwEx("Labels and preOutput must have equal shapes: got shapes %s vs %s", labels.shape(), preOutput.shape());
        }
        labels = labels.castTo(preOutput.dataType());   //No-op if already correct dtype
        INDArray output = activationFn.getActivation(preOutput.dup(), true);

        INDArray p1 = output.add(1.0);
        INDArray dlda = p1.rdiv(2.0 / labels.size(1));
        INDArray logRatio = Transforms.log(p1.divi(labels.add(1.0)), false);
        dlda.muli(logRatio);

        if (weights != null) {
            dlda.muliRowVector(weights);
        }

        if (mask != null && LossUtil.isPerOutputMasking(dlda, mask)) {
            //For *most* activation functions: we don't actually need to mask dL/da in addition to masking dL/dz later
            //but: some, like softmax, require both (due to dL/dz_i being a function of dL/da_j, for i != j)
            //We could add a special case for softmax (activationFn instanceof ActivationSoftmax) but that would be
            // error prone - though buy us a tiny bit of performance
            LossUtil.applyMask(dlda, mask);
        }

        //dL/dz
        INDArray gradients = activationFn.backprop(preOutput, dlda).getFirst(); //TODO activation functions with weights

        if (mask != null) {
            LossUtil.applyMask(gradients, mask);
        }

        return gradients;
    }

    @Override
    public Pair<Double, INDArray> computeGradientAndScore(INDArray labels,
                    INDArray preOutput, IActivation activationFn, INDArray mask, boolean average) {
        //TODO: probably a more efficient way to do this...
        //Yes - will implement in round two. Just want to get done now.

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
            return "LossMSLE()";
        return "LossMSLE(weights=" + weights + ")";
    }
}
