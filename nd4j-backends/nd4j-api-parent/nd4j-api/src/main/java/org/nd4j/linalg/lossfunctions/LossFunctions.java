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
