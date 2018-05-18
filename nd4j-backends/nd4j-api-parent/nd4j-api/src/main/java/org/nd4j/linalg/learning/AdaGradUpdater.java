/*-
 *
 *  * Copyright 2017 Skymind,Inc.
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

package org.nd4j.linalg.learning;


import lombok.Data;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.shape.Shape;
import org.nd4j.linalg.learning.config.AdaGrad;

import static org.nd4j.linalg.ops.transforms.Transforms.sqrt;


/**
 * Vectorized Learning Rate used per Connection Weight
 * <p/>
 * Adapted from: http://xcorr.net/2014/01/23/adagrad-eliminating-learning-rates-in-stochastic-gradient-descent/
 * See also http://cs231n.github.io/neural-networks-3/#ada
 *
 * @author Adam Gibson
 */
@Data
public class AdaGradUpdater implements GradientUpdater<AdaGrad> {
    public INDArray historicalGradient;
    public int[] shape;
    protected double learningRate = 1e-1; // learning rate
    protected int numIterations = 0;
    private double epsilon = AdaGrad.DEFAULT_ADAGRAD_EPSILON;

    private char gradientReshapeOrder;

    private AdaGrad config;

    public AdaGradUpdater(AdaGrad config) {
        this.config = config;
    }

    @Override
    public void setStateViewArray(INDArray viewArray, long[] gradientShape, char gradientOrder, boolean initialize) {
        if (!viewArray.isRowVector())
            throw new IllegalArgumentException("Invalid input: expect row vector input");
        if (initialize)
            viewArray.assign(epsilon);
        this.historicalGradient = viewArray;
        //Reshape to match the expected shape of the input gradient arrays
        this.historicalGradient = Shape.newShapeNoCopy(this.historicalGradient, gradientShape, gradientOrder == 'f');
        if (historicalGradient == null)
            throw new IllegalStateException("Could not correctly reshape gradient view array");

        this.gradientReshapeOrder = gradientOrder;
    }

    /**
     * Gets feature specific learning rates
     * Adagrad keeps a history of gradients being passed in.
     * Note that each gradient passed in becomes adapted over time, hence the opName adagrad
     *
     * @param gradient  the gradient to get learning rates for
     * @param iteration
     */
    @Override
    public void applyUpdater(INDArray gradient, int iteration, int epoch) {
        if (historicalGradient == null)
            throw new IllegalStateException("Updater has not been initialized with view state");

        double learningRate = config.getLearningRate(iteration, epoch);
        double epsilon = config.getEpsilon();

        historicalGradient.addi(gradient.mul(gradient));

        INDArray sqrtHistory = sqrt(historicalGradient.dup(gradientReshapeOrder), false).addi(epsilon);
        // lr * gradient / (sqrt(sumSquaredGradients) + epsilon)
        gradient.muli(sqrtHistory.rdivi(learningRate));
    }
}
