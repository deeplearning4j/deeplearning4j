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
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.learning.config.AdaDelta;
import org.nd4j.linalg.ops.transforms.Transforms;

/**
 * http://www.matthewzeiler.com/pubs/googleTR2012/googleTR2012.pdf
 * https://arxiv.org/pdf/1212.5701v1.pdf
 * <p>
 * Ada delta updater. More robust adagrad that keeps track of a moving window
 * average of the gradient rather than the every decaying learning rates of adagrad
 *
 * @author Adam Gibson
 */
@Data
public class AdaDeltaUpdater implements GradientUpdater<AdaDelta> {

    private final AdaDelta config;

    private INDArray msg; //E[g^2]_t by arxiv paper, algorithm 1
    private INDArray msdx; //E[delta x^2]_t by arxiv paper, algorithm 1



    public AdaDeltaUpdater(AdaDelta config) {
        this.config = config;
    }

    @Override
    public void setStateViewArray(INDArray viewArray, long[] gradientShape, char gradientOrder, boolean initialize) {
        if (!viewArray.isRowVector())
            throw new IllegalArgumentException("Invalid input: expect row vector input");
        if (initialize)
            viewArray.assign(0);
        long length = viewArray.length();
        this.msg = viewArray.get(NDArrayIndex.point(0), NDArrayIndex.interval(0, length / 2));
        this.msdx = viewArray.get(NDArrayIndex.point(0), NDArrayIndex.interval(length / 2, length));

        //Reshape to match the expected shape of the input gradient arrays
        this.msg = Shape.newShapeNoCopy(this.msg, gradientShape, gradientOrder == 'f');
        this.msdx = Shape.newShapeNoCopy(this.msdx, gradientShape, gradientOrder == 'f');
        if (msg == null || msdx == null)
            throw new IllegalStateException("Could not correctly reshape gradient view arrays");
    }

    /**
     * Get the updated gradient for the given gradient
     * and also update the state of ada delta.
     *
     * @param gradient  the gradient to get the
     *                  updated gradient for
     * @param iteration
     * @return the update gradient
     */
    @Override
    public void applyUpdater(INDArray gradient, int iteration, int epoch) {
        if (msg == null || msdx == null)
            throw new IllegalStateException("Updater has not been initialized with view state");

        double rho = config.getRho();
        double epsilon = config.getEpsilon();

        //Line 4 of Algorithm 1: https://arxiv.org/pdf/1212.5701v1.pdf
        //E[g^2]_t = rho * E[g^2]_{t−1} + (1-rho)*g^2_t
        msg.muli(rho).addi(gradient.mul(gradient).muli(1 - rho));

        //Calculate update:
        //dX = - g * RMS[delta x]_{t-1} / RMS[g]_t
        //Note: negative is applied in the DL4J step function: params -= update rather than params += update
        INDArray rmsdx_t1 = Transforms.sqrt(msdx.add(epsilon), false);
        INDArray rmsg_t = Transforms.sqrt(msg.add(epsilon), false);
        INDArray update = gradient.muli(rmsdx_t1.divi(rmsg_t));

        //Accumulate gradients: E[delta x^2]_t = rho * E[delta x^2]_{t-1} + (1-rho)* (delta x_t)^2
        msdx.muli(rho).addi(update.mul(update).muli(1 - rho));
    }
}
