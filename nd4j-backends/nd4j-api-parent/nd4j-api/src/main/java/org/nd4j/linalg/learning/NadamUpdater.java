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
import org.apache.commons.math3.util.FastMath;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.shape.Shape;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.learning.config.Nadam;
import org.nd4j.linalg.ops.transforms.Transforms;

/**
 * The Nadam updater.
 * https://arxiv.org/pdf/1609.04747.pdf
 *
 * @author Andrey Spiridonov
 */
@Data
public class NadamUpdater implements GradientUpdater<Nadam> {

    private Nadam config;
    private INDArray m, v; // moving avg & sqrd gradients

    private char gradientReshapeOrder;

    public NadamUpdater(Nadam config) {
        this.config = config;
    }

    @Override
    public void setStateViewArray(INDArray viewArray, long[] gradientShape, char gradientOrder, boolean initialize) {
        if (!viewArray.isRowVector())
            throw new IllegalArgumentException("Invalid input: expect row vector input");
        if (initialize)
            viewArray.assign(0);
        long length = viewArray.length();
        this.m = viewArray.get(NDArrayIndex.point(0), NDArrayIndex.interval(0, length / 2));
        this.v = viewArray.get(NDArrayIndex.point(0), NDArrayIndex.interval(length / 2, length));

        //Reshape to match the expected shape of the input gradient arrays
        this.m = Shape.newShapeNoCopy(this.m, gradientShape, gradientOrder == 'f');
        this.v = Shape.newShapeNoCopy(this.v, gradientShape, gradientOrder == 'f');
        if (m == null || v == null)
            throw new IllegalStateException("Could not correctly reshape gradient view arrays");

        this.gradientReshapeOrder = gradientOrder;
    }

    /**
     * Calculate the update based on the given gradient
     *
     * @param gradient  the gradient to get the update for
     * @param iteration
     * @return the gradient
     */
    @Override
    public void applyUpdater(INDArray gradient, int iteration, int epoch) {
        if (m == null || v == null)
            throw new IllegalStateException("Updater has not been initialized with view state");

        double beta1 = config.getBeta1();
        double beta2 = config.getBeta2();
        double learningRate = config.getLearningRate(iteration, epoch);
        double epsilon = config.getEpsilon();

        INDArray oneMinusBeta1Grad = gradient.mul(1.0 - beta1);
        m.muli(beta1).addi(oneMinusBeta1Grad);

        INDArray oneMinusBeta2GradSquared = gradient.mul(gradient).muli(1.0 - beta2);
        v.muli(beta2).addi(oneMinusBeta2GradSquared);

        double beta1t = FastMath.pow(beta1, iteration + 1);

        INDArray biasCorrectedEstimateOfMomentum = m.mul(beta1).divi(1.0 - beta1t);
        INDArray secondTerm = oneMinusBeta1Grad.divi(1 - beta1t);

        INDArray alphat = biasCorrectedEstimateOfMomentum.add(secondTerm).muli(learningRate);

        INDArray sqrtV = Transforms.sqrt(v.dup(gradientReshapeOrder), false).addi(epsilon);

        gradient.assign(alphat).divi(sqrtV);
    }
}
