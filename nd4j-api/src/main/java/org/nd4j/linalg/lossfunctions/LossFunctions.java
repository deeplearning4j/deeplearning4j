/*
 * Copyright 2015 Skymind,Inc.
 *
 *    Licensed under the Apache License, Version 2.0 (the "License");
 *    you may not use this file except in compliance with the License.
 *    You may obtain a copy of the License at
 *
 *        http://www.apache.org/licenses/LICENSE-2.0
 *
 *    Unless required by applicable law or agreed to in writing, software
 *    distributed under the License is distributed on an "AS IS" BASIS,
 *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *    See the License for the specific language governing permissions and
 *    limitations under the License.
 */

package org.nd4j.linalg.lossfunctions;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.Arrays;

import static org.nd4j.linalg.ops.transforms.Transforms.*;


/**
 * Central class for loss functions
 */
public class LossFunctions {


    /**
     * Generic scoring function
     *
     * @param labels            the labels to score
     * @param lossFunction      the loss function to use
     * @param z                 the output function
     * @param l2                the l2 coefficient
     * @param useRegularization whether to use regularization
     * @return the score for the given parameters
     */
    public static double score(INDArray labels, LossFunction lossFunction, INDArray z, double l2, boolean useRegularization) {
        assert !Nd4j.hasInvalidNumber(z) : "Invalid output on labels. Must not contain nan or infinite numbers.";

        double ret = 0.0f;
        double reg = 0.5 * l2;
        if (!Arrays.equals(labels.shape(), z.shape()))
            throw new IllegalArgumentException("Output and labels must be same length");
        switch (lossFunction) {
            case RECONSTRUCTION_CROSSENTROPY:
                INDArray xEntLogZ2 = log(z);
                INDArray xEntOneMinusLabelsOut2 = labels.rsub(1);
                INDArray xEntOneMinusLogOneMinusZ2 = log(z).rsubi(1);
                ret = labels.mul(xEntLogZ2).add(xEntOneMinusLabelsOut2).muli(xEntOneMinusLogOneMinusZ2).sum(1).mean(Integer.MAX_VALUE).getDouble(0);
                break;
            case MCXENT:
                INDArray columnSums = labels.mul(log(z));
                ret = columnSums.mean(1).mean(Integer.MAX_VALUE).getDouble(0);
                break;
            case XENT:
                INDArray xEntLogZ = log(z);
                INDArray xEntOneMinusLabelsOut = labels.rsub(1);
                INDArray xEntOneMinusLogOneMinusZ = log(z).rsubi(1);
                ret = labels.mul(xEntLogZ).add(xEntOneMinusLabelsOut).muli(xEntOneMinusLogOneMinusZ).sum(1).sum(Integer.MAX_VALUE).getDouble(0);
                break;
            case RMSE_XENT:
                INDArray rmseXentDiff = labels.sub(z);
                INDArray squaredrmseXentDiff = pow(rmseXentDiff, 2.0);
                INDArray sqrt = sqrt(squaredrmseXentDiff);
                ret = sqrt.sum(1).mean(Integer.MAX_VALUE).getDouble(0);
                break;
            case MSE:
                INDArray mseDelta = labels.sub(z);
                ret = 0.5 * pow(mseDelta, 2).sum(1).sum(Integer.MAX_VALUE).getDouble(0);
                break;
            case EXPLL:
                INDArray expLLLogZ = log(z);
                ret = z.sub(labels.mul(expLLLogZ)).sum(1).sum(Integer.MAX_VALUE).getDouble(0);
                break;
            case SQUARED_LOSS:
                ret = pow(labels.sub(z), 2).sum(1).sum(Integer.MAX_VALUE).getDouble(0);
                break;
            case NEGATIVELOGLIKELIHOOD:
                ret = -Nd4j.mean(Nd4j.sum(
                        labels.mul(log(z))
                                .addi(labels.rsub(1).muli(log(z.rsub(1))))
                        , 1)).getDouble(0);
                break;


        }

        if (useRegularization)
            ret += reg;

        ret /= (double) labels.rows();
        return ret;

    }


    /**
     * MSE: Mean Squared Error: Linear Regression
     * EXPLL: Exponential log likelihood: Poisson Regression
     * XENT: Cross Entropy: Binary Classification
     * SOFTMAX: Softmax Regression
     * RMSE_XENT: RMSE Cross Entropy
     */
    public static enum LossFunction {
        MSE,
        EXPLL,
        XENT,
        MCXENT,
        RMSE_XENT,
        SQUARED_LOSS,
        RECONSTRUCTION_CROSSENTROPY,
        NEGATIVELOGLIKELIHOOD
    }


}
