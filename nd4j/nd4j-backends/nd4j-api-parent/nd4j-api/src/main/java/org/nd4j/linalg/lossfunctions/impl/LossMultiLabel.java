/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  *  See the NOTICE file distributed with this work for additional
 *  *  information regarding copyright ownership.
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
import lombok.Getter;
import org.nd4j.linalg.activations.IActivation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.ILossFunction;
import org.nd4j.linalg.lossfunctions.LossUtil;
import org.nd4j.linalg.ops.transforms.Transforms;
import org.nd4j.common.primitives.Pair;
import org.nd4j.shade.jackson.annotation.JsonInclude;

@EqualsAndHashCode
@JsonInclude(JsonInclude.Include.NON_NULL)
@Getter
public class LossMultiLabel implements ILossFunction {


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
        labels = labels.castTo(preOutput.dataType());   //No-op if already correct dtype
        final INDArray postOutput = activationFn.getActivation(preOutput.dup(), true);

        final INDArray positive = labels;
        final INDArray negative = labels.eq(0.0).castTo(Nd4j.defaultFloatingPointType());
        final INDArray normFactor = negative.sum(true,1).castTo(Nd4j.defaultFloatingPointType()).muli(positive.sum(true,1));


        long examples = positive.size(0);
        for (int i = 0; i < examples; i++) {
            final INDArray locCfn = postOutput.getRow(i, true);
            final long[] shape = locCfn.shape();

            final INDArray locPositive = positive.getRow(i, true);
            final INDArray locNegative = negative.getRow(i, true);
            final Double locNormFactor = normFactor.getDouble(i);

            final int outSetSize = locNegative.sumNumber().intValue();
            if(outSetSize == 0 || outSetSize == locNegative.columns()){
                if (scoreOutput != null) {
                    scoreOutput.getRow(i, true).assign(0);
                }

                if (gradientOutput != null) {
                    gradientOutput.getRow(i, true).assign(0);
                }
            }else {
                final INDArray operandA = Nd4j.ones(shape[1], shape[0]).mmul(locCfn);
                final INDArray operandB = operandA.transpose();

                final INDArray pairwiseSub = Transforms.exp(operandA.sub(operandB));

                final INDArray selection = locPositive.transpose().mmul(locNegative);

                final INDArray classificationDifferences = pairwiseSub.muli(selection).divi(locNormFactor);

                if (scoreOutput != null) {
                    if (mask != null) {
                        final INDArray perLabel = classificationDifferences.sum(0);
                        LossUtil.applyMask(perLabel, mask.getRow(i, true));
                        perLabel.sum(scoreOutput.getRow(i, true), 0);
                    } else {
                        classificationDifferences.sum(scoreOutput.getRow(i, true), 0, 1);
                    }
                }

                if (gradientOutput != null) {
                    gradientOutput.getRow(i, true).assign(classificationDifferences.sum(true, 0).addi(classificationDifferences.sum(true,1).transposei().negi()));
                }
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
        return scoreArr.sum(true,1);
    }

    @Override
    public INDArray computeGradient(INDArray labels, INDArray preOutput, IActivation activationFn, INDArray mask) {
        labels = labels.castTo(preOutput.dataType());   //No-op if already correct dtype
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
}
