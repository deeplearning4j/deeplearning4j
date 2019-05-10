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

package org.nd4j.linalg.learning.legacy;

import lombok.Data;
import lombok.NoArgsConstructor;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.shape.Shape;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.util.ArrayUtil;

import java.io.Serializable;

import static org.nd4j.linalg.ops.transforms.Transforms.sqrt;


/**
 * Legacy AdaGrad implementation for use in NLP etc applications
 */
@Data
@NoArgsConstructor
public class AdaGrad implements Serializable {
    public static final double DEFAULT_ADAGRAD_EPSILON = 1e-6;

    public INDArray historicalGradient;
    public int[] shape;
    protected double learningRate = 1e-1; // learning rate
    protected int numIterations = 0;
    private double epsilon = DEFAULT_ADAGRAD_EPSILON;

    private char gradientReshapeOrder;


    public int stateSizeForInputSize(int inputSize) {
        return inputSize;
    }

    public void setStateViewArray(INDArray viewArray, int[] gradientShape, char gradientOrder, boolean initialize) {
        setStateViewArray(viewArray, ArrayUtil.toLongArray(gradientShape), gradientOrder, initialize);
    }

    public void setStateViewArray(INDArray viewArray, long[] gradientShape, char gradientOrder, boolean initialize) {
        if (!viewArray.isRowVector() && !(viewArray.rank() == 2 && viewArray.columns() == 1 && viewArray.rows() == 1))
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
     * @param rows
     * @param cols
     * @param learningRate
     */
    public AdaGrad(int rows, int cols, double learningRate) {
        this.shape = new int[] {rows, cols};
        this.learningRate = learningRate;
    }

    public AdaGrad(int rows, int cols) {
        this(rows, cols, 0.1);
    }

    public AdaGrad(int[] shape, double learningRate) {
        this.shape = shape;
        this.learningRate = learningRate;
    }

    public AdaGrad(double learningRate) {
        this.learningRate = learningRate;
    }

    public AdaGrad(double learningRate, double epsilon) {
        this.learningRate = learningRate;
        this.epsilon = epsilon;
    }

    public void update(Object... args) {
        if (args.length > 0) {
            learningRate = (Double) args[0];
        }
    }

    /**
     * Gets feature specific learning rates
     * Adagrad keeps a history of gradients being passed in.
     * Note that each gradient passed in becomes adapted over time, hence
     * the opName adagrad
     *
     * @param gradient  the gradient to get learning rates for
     * @param iteration
     * @return the feature specific learning rates
     */
    public INDArray getGradient(INDArray gradient, int iteration) {
        if (historicalGradient == null)
            throw new IllegalStateException("Updater has not been initialized with view state");

        historicalGradient.addi(gradient.mul(gradient));

        INDArray sqrtHistory = sqrt(historicalGradient.dup(gradientReshapeOrder), false).addi(epsilon);
        // lr * gradient / (sqrt(sumSquaredGradients) + epsilon)
        INDArray ret = gradient.muli(sqrtHistory.rdivi(learningRate));
        numIterations++;
        return ret;
    }

    public double getGradient(double gradient, int column, int[] shape) {
        boolean historicalInitialized = false;
        if (this.historicalGradient == null) {
            this.historicalGradient = Nd4j.ones(shape);
            historicalInitialized = true;
        }

        double sqrtHistory = !historicalInitialized ? Math.sqrt(historicalGradient.getDouble(column))
                        : historicalGradient.getDouble(column);
        double learningRates = learningRate / (sqrtHistory + epsilon);
        double adjustedGradient = gradient * (learningRates);

        historicalGradient.putScalar(column, historicalGradient.getDouble(column) + gradient * gradient);
        numIterations++;

        //ensure no zeros
        return adjustedGradient;
    }

    public INDArray getGradient(INDArray gradient, int slice, int[] shape) {
        boolean historicalInitialized = false;
        INDArray sqrtHistory;

        if (this.historicalGradient == null) {
            this.historicalGradient = Nd4j.zeros(shape).add(epsilon);
            historicalInitialized = true;
        } else if (!this.historicalGradient.isVector()
                        && this.historicalGradient.slice(slice).length() != gradient.length())
            throw new IllegalArgumentException("Illegal gradient");

        if (historicalGradient.isVector())
            sqrtHistory = sqrt(historicalGradient);
        else
            sqrtHistory = !historicalInitialized ? sqrt(historicalGradient.slice(slice)) : historicalGradient;
        INDArray learningRates;
        try {
            learningRates = sqrtHistory.rdivi(learningRate);
        } catch (ArithmeticException ae) {
            learningRates = sqrtHistory.rdivi(learningRate + epsilon);
        }
        if (gradient.length() != learningRates.length())
            gradient.muli(learningRates.slice(slice));
        else
            gradient.muli(learningRates);

        this.historicalGradient.slice(slice).addi(gradient.mul(gradient));
        numIterations++;

        //ensure no zeros
        return gradient;
    }

    public AdaGrad createSubset(int index) {
        if (historicalGradient == null)
            this.historicalGradient = Nd4j.ones(shape);

        if (Shape.isMatrix(shape)) {
            AdaGrad a = new AdaGrad(1, historicalGradient.columns());
            //grab only the needed elements
            INDArray slice = historicalGradient.slice(index).dup();
            a.historicalGradient = slice;
            a.setLearningRate(learningRate);
            return a;
        } else {
            AdaGrad a = new AdaGrad(1, 1);
            //grab only the needed elements
            INDArray slice = Nd4j.scalar(historicalGradient.getDouble(index));
            a.historicalGradient = slice;
            a.setLearningRate(learningRate);
            return a;
        }
    }
}
