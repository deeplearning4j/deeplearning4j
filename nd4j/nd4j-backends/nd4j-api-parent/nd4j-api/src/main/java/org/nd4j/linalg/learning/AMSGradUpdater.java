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

package org.nd4j.linalg.learning;

import lombok.Data;
import lombok.val;
import org.apache.commons.math3.util.FastMath;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.transforms.Sqrt;
import org.nd4j.linalg.api.shape.Shape;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.learning.config.AMSGrad;
import org.nd4j.linalg.ops.transforms.Transforms;

/**
 * The AMSGrad updater<br>
 * Reference: On the Convergence of Adam and Beyond - https://openreview.net/forum?id=ryQu7f-RZ
 *
 * @author Alex Black
 */
@Data
public class AMSGradUpdater implements GradientUpdater<AMSGrad> {

    private AMSGrad config;
    private INDArray m, v, vHat; // moving avg, sqrd gradients, max

    private char gradientReshapeOrder;

    public AMSGradUpdater(AMSGrad config) {
        this.config = config;
    }

    @Override
    public void setStateViewArray(INDArray viewArray, long[] gradientShape, char gradientOrder, boolean initialize) {
        if (!viewArray.isRowVector())
            throw new IllegalArgumentException("Invalid input: expect row vector input");
        if (initialize)
            viewArray.assign(0);
        val n = viewArray.length() / 3;
        this.m = viewArray.get(NDArrayIndex.point(0), NDArrayIndex.interval(0, n));
        this.v = viewArray.get(NDArrayIndex.point(0), NDArrayIndex.interval(n, 2*n));
        this.vHat = viewArray.get(NDArrayIndex.point(0), NDArrayIndex.interval(2*n, 3*n));

        //Reshape to match the expected shape of the input gradient arrays
        this.m = Shape.newShapeNoCopy(this.m, gradientShape, gradientOrder == 'f');
        this.v = Shape.newShapeNoCopy(this.v, gradientShape, gradientOrder == 'f');
        this.vHat = Shape.newShapeNoCopy(this.vHat, gradientShape, gradientOrder == 'f');
        if (m == null || v == null || vHat == null)
            throw new IllegalStateException("Could not correctly reshape gradient view arrays");

        this.gradientReshapeOrder = gradientOrder;
    }

    @Override
    public void applyUpdater(INDArray gradient, int iteration, int epoch) {
        if (m == null || v == null || vHat == null)
            throw new IllegalStateException("Updater has not been initialized with view state");

        double beta1 = config.getBeta1();
        double beta2 = config.getBeta2();
        double learningRate = config.getLearningRate(iteration, epoch);
        double epsilon = config.getEpsilon();

        //m_t = b_1 * m_{t-1} + (1-b_1) * g_t       eq 1 pg 3
        INDArray oneMinusBeta1Grad = gradient.mul(1.0 - beta1);
        m.muli(beta1).addi(oneMinusBeta1Grad);

        //v_t = b_2 * v_{t-1} + (1-b_2) * (g_t)^2   eq 1 pg 3
        INDArray oneMinusBeta2GradSquared = gradient.mul(gradient).muli(1 - beta2);
        v.muli(beta2).addi(oneMinusBeta2GradSquared);

        double beta1t = FastMath.pow(beta1, iteration + 1);
        double beta2t = FastMath.pow(beta2, iteration + 1);

        //vHat_t = max(vHat_{t-1}, v_t)
        Transforms.max(vHat, v, false);

        double alphat = learningRate * FastMath.sqrt(1 - beta2t) / (1 - beta1t);
        if (Double.isNaN(alphat) || alphat == 0.0)
            alphat = epsilon;

        //gradient array contains: sqrt(vHat) + eps
        Nd4j.getExecutioner().execAndReturn(new Sqrt(vHat, gradient)).addi(epsilon);

        //gradient = alphat * m_t / (sqrt(vHat) + eps)
        gradient.rdivi(m).muli(alphat);
    }
}
