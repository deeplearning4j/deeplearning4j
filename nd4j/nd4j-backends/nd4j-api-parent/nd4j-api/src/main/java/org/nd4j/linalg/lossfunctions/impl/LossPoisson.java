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
import org.nd4j.common.base.Preconditions;
import org.nd4j.linalg.activations.IActivation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.lossfunctions.ILossFunction;
import org.nd4j.linalg.lossfunctions.LossUtil;
import org.nd4j.linalg.ops.transforms.Transforms;
import org.nd4j.common.primitives.Pair;

/**
 * Created by susaneraly on 9/9/16.
 */
@EqualsAndHashCode
public class LossPoisson implements ILossFunction {

    public INDArray scoreArray(INDArray labels, INDArray preOutput, IActivation activationFn, INDArray mask) {
        if(!labels.equalShapes(preOutput)){
            Preconditions.throwEx("Labels and preOutput must have equal shapes: got shapes %s vs %s", labels.shape(), preOutput.shape());
        }
        labels = labels.castTo(preOutput.dataType());   //No-op if already correct dtype
        /*
         mean of (yhat - y * log(yhat))
         */
        INDArray postOutput = activationFn.getActivation(preOutput.dup(), true);

        INDArray scoreArr = Transforms.log(postOutput);
        scoreArr.muli(labels);
        scoreArr = postOutput.sub(scoreArr);

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
        INDArray yHat = activationFn.getActivation(preOutput.dup(), true);
        INDArray yDivyhat = labels.div(yHat);
        INDArray dLda = yDivyhat.rsubi(1);

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
        return "LossPoisson()";
    }
}
