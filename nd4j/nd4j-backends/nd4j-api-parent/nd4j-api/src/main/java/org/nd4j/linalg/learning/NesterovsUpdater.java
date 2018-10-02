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
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.transforms.pairwise.arithmetic.OldAddOp;
import org.nd4j.linalg.api.shape.Shape;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Nesterovs;

/**
 * Nesterov's momentum.
 * Keep track of the previous layer's gradient
 * and use it as a way of updating the gradient.
 *
 * @author Adam Gibson
 */
@Data
public class NesterovsUpdater implements GradientUpdater<Nesterovs> {

    private final Nesterovs config;

    private INDArray v;
    private char gradientReshapeOrder;

    public NesterovsUpdater(Nesterovs config) {
        this.config = config;
    }

    @Override
    public void setStateViewArray(INDArray viewArray, long[] gradientShape, char gradientOrder, boolean initialize) {
        if (!viewArray.isRowVector())
            throw new IllegalArgumentException("Invalid input: expect row vector input");
        if (initialize)
            viewArray.assign(0);

        this.v = viewArray;

        //Reshape to match the expected shape of the input gradient arrays
        this.v = Shape.newShapeNoCopy(this.v, gradientShape, gradientOrder == 'f');
        if (v == null)
            throw new IllegalStateException("Could not correctly reshape gradient view array");
        this.gradientReshapeOrder = gradientOrder;
    }

    /**
     * Get the nesterov update
     *
     * @param gradient  the gradient to get the update for
     * @param iteration
     * @return
     */
    @Override
    public void applyUpdater(INDArray gradient, int iteration, int epoch) {
        if (v == null)
            throw new IllegalStateException("Updater has not been initialized with view state");

        double momentum = config.currentMomentum(iteration, epoch);
        double learningRate = config.getLearningRate(iteration, epoch);

        //reference https://cs231n.github.io/neural-networks-3/#sgd 2nd equation
        //DL4J default is negative step function thus we flipped the signs:
        // x += mu * v_prev + (-1 - mu) * v
        //i.e., we do params -= updatedGradient, not params += updatedGradient

        //v = mu * v - lr * gradient
        INDArray vPrev = v.dup(gradientReshapeOrder);
        v.muli(momentum).subi(gradient.dup(gradientReshapeOrder).muli(learningRate)); //Modify state array in-place

        /*
        Next line is equivalent to:
        INDArray ret = vPrev.muli(momentum).addi(v.mul(-momentum - 1));
        gradient.assign(ret);
        */
        Nd4j.getExecutioner().exec(new OldAddOp(vPrev.muli(momentum), v.mul(-momentum - 1), gradient));
    }
}
