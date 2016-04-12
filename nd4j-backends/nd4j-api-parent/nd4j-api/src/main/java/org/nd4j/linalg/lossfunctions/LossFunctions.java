/*
 *
 *  * Copyright 2015 Skymind,Inc.
 *  *
 *  *    Licensed under the Apache License, Version 2.0 (the "License");
 *  *    you may not use this file except in compliance with the License.
 *  *    You may obtain a copy of the License at
 *  *
 *  *        http://www.apache.org/licenses/LICENSE-2.0
 *  *
 *  *    Unless required by applicable law or agreed to in writing, software
 *  *    distributed under the License is distributed on an "AS IS" BASIS,
 *  *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  *    See the License for the specific language governing permissions and
 *  *    limitations under the License.
 *
 *
 */

package org.nd4j.linalg.lossfunctions;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.Arrays;

import static org.nd4j.linalg.ops.transforms.Transforms.*;


/**
 * Central class for loss functions
 * @author Adam Gibson
 */
public class LossFunctions {

    /**
     * Generic scoring function.
     * Note that an IllegalArgumentException is thrown if the given
     * loss function is custom. An alternative mechanism for scoring
     * (preferrably with a function name and the op factory) should be used instead.
     *
     * @param labels            the labels to score
     * @param lossFunction      the loss function to use
     * @param z                 the output function
     * @param l2                the l2 regularization term (0.5 * l2Coeff * sum w^2)
     * @param l1                the l1 regularization term (l1Coeff * sum |w|)
     * @param useRegularization whether to use regularization
     * @return the score for the given parameters
     */
    public static double score(INDArray labels, LossFunction lossFunction, INDArray z, double l2, double l1,boolean useRegularization) {
        return LossCalculation.builder()
                .l1(l1).lossFunction(lossFunction)
                .l2(l2).labels(labels)
                .z(z)
                .useRegularization(useRegularization)
                .build().score();
    }

    /**
     * Generic scoring function.
     * Note that an IllegalArgumentException is thrown if the given
     * loss function is custom. An alternative mechanism for scoring
     * (preferrably with a function name and the op factory) should be used instead.
     *
     * @param labels            the labels to score
     * @param lossFunction      the loss function to use
     * @param z                 the output function
     * @param l2                the l2 coefficient
     * @param useRegularization whether to use regularization
     * @return the score for the given parameters
     */
    public static double score(INDArray labels, LossFunction lossFunction, INDArray z, double l2, boolean useRegularization) {
        double ret = 0.0;
        double reg = 0.5 * l2;
        if (!Arrays.equals(labels.shape(), z.shape()))
            throw new IllegalArgumentException("Output and labels must be same length");
        boolean oldEnforce = Nd4j.ENFORCE_NUMERICAL_STABILITY;
        Nd4j.ENFORCE_NUMERICAL_STABILITY = true;
        switch (lossFunction) {
            case CUSTOM: throw new IllegalStateException("Unable to score custom operation. Please define an alternative mechanism");
            case RECONSTRUCTION_CROSSENTROPY:
                INDArray xEntLogZ2 = log(z);
                INDArray xEntOneMinusLabelsOut2 = labels.rsub(1);
                INDArray xEntOneMinusLogOneMinusZ2 = log(z).rsubi(1);
                ret = labels.mul(xEntLogZ2).add(xEntOneMinusLabelsOut2).muli(xEntOneMinusLogOneMinusZ2).sum(1).meanNumber().doubleValue();
                break;
            case MCXENT:
                INDArray sums = log(z);
                INDArray columnSums = labels.mul(sums);
                ret = -columnSums.sumNumber().doubleValue();
                break;
            case XENT:
                INDArray xEntLogZ = log(z);
                INDArray xEntOneMinusLabelsOut = labels.rsub(1);
                INDArray xEntOneMinusLogOneMinusZ = log(z).rsubi(1);
                ret = labels.mul(xEntLogZ).add(xEntOneMinusLabelsOut).muli(xEntOneMinusLogOneMinusZ).sum(1).sumNumber().doubleValue();
                break;
            case RMSE_XENT:
                INDArray rmseXentDiff = labels.sub(z);
                INDArray squaredrmseXentDiff = pow(rmseXentDiff, 2.0);
                INDArray sqrt = sqrt(squaredrmseXentDiff);
                ret = sqrt.sum(1).sumNumber().doubleValue();
                break;
            case MSE:
                INDArray mseDelta = labels.sub(z);
                ret = 0.5 * pow(mseDelta, 2).sum(1).sumNumber().doubleValue();
                break;
            case EXPLL:
                INDArray expLLLogZ = log(z);
                ret = z.sub(labels.mul(expLLLogZ)).sum(1).sumNumber().doubleValue();
                break;
            case SQUARED_LOSS:
                ret = pow(labels.sub(z), 2).sum(1).sumNumber().doubleValue();
                break;
            case NEGATIVELOGLIKELIHOOD:
                INDArray sums2 = log(z);
                INDArray columnSums2 = labels.mul(sums2);
                ret = -columnSums2.sumNumber().doubleValue();
                break;


        }

        if (useRegularization)
            ret += reg;


        ret /= (double) labels.size(0);
        Nd4j.ENFORCE_NUMERICAL_STABILITY = oldEnforce;
        return ret;

    }


    /**
     * MSE: Mean Squared Error: Linear Regression
     * EXPLL: Exponential log likelihood: Poisson Regression
     * XENT: Cross Entropy: Binary Classification
     * MCXENT: Multiclass Cross Entropy
     * RMSE_XENT: RMSE Cross Entropy
     * SQUARED_LOSS: Squared Loss
     * RECONSTRUCTION_CROSSENTROPY: Reconstruction Cross Entropy
     * NEGATIVELOGLIKELIHOOD: Negative Log Likelihood
     * CUSTOM: Define your own loss function
     */
    public  enum LossFunction {
        MSE,
        EXPLL,
        XENT,
        MCXENT,
        RMSE_XENT,
        SQUARED_LOSS,
        RECONSTRUCTION_CROSSENTROPY,
        NEGATIVELOGLIKELIHOOD,
        CUSTOM
    }


}
