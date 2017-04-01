/*-
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

package org.nd4j.linalg.learning;


import lombok.Data;
import lombok.NoArgsConstructor;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.shape.Shape;
import org.nd4j.linalg.factory.Nd4j;

import java.io.Serializable;

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
@NoArgsConstructor
public class AdaGrad implements Serializable, GradientUpdater {
    public static final double DEFAULT_ADAGRAD_EPSILON = 1e-6;

    //protected double squaredGradientSum = 0;
    public INDArray historicalGradient;
    public int[] shape;
    protected double learningRate = 1e-1; // learning rate
    protected int numIterations = 0;
    private double epsilon = DEFAULT_ADAGRAD_EPSILON;

    @Override
    public int stateSizeForInputSize(int inputSize) {
        return inputSize;
    }

    @Override
    public void setStateViewArray(INDArray viewArray, int[] gradientShape, char gradientOrder, boolean initialize) {
        if (!viewArray.isRowVector())
            throw new IllegalArgumentException("Invalid input: expect row vector input");
        if (initialize)
            viewArray.assign(epsilon);
        this.historicalGradient = viewArray;
        //Reshape to match the expected shape of the input gradient arrays
        this.historicalGradient = Shape.newShapeNoCopy(this.historicalGradient, gradientShape, gradientOrder == 'f');
        if (historicalGradient == null)
            throw new IllegalStateException("Could not correctly reshape gradient view array");
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

    @Override
    public void update(Object... args) {
        if (args.length > 0) {
            learningRate = (Double) args[0];
        }
    }

    /**
     * Gets feature specific learning rates
     * Adagrad keeps a history of gradients being passed in.
     * Note that each gradient passed in becomes adapted over time, hence
     * the name adagrad
     *
     * @param gradient  the gradient to get learning rates for
     * @param iteration
     * @return the feature specific learning rates
     */
    @Override
    public INDArray getGradient(INDArray gradient, int iteration) {
        if (historicalGradient == null)
            throw new IllegalStateException("Updater has not been initialized with view state");

        historicalGradient.addi(gradient.mul(gradient));

        INDArray sqrtHistory = sqrt(historicalGradient, true).addi(epsilon);
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

    @Override
    public GradientUpdaterAggregator getAggregator(boolean addThis) {
        AdaGradAggregator ag = new AdaGradAggregator();
        if (addThis)
            ag.aggregate(this);
        return ag;
    }

    public static class AdaGradAggregator implements GradientUpdaterAggregator {
        private INDArray historicalGradientSum;
        private double lrSum;
        private long numIterationsSum = 0;
        private int count = 0;

        @Override
        public GradientUpdater getUpdater() {
            AdaGrad adaGrad = new AdaGrad(lrSum / count);
            adaGrad.setHistoricalGradient(historicalGradientSum.div(count));
            adaGrad.setNumIterations((int) (numIterationsSum / count));
            return adaGrad;
        }

        @Override
        public void aggregate(GradientUpdater updater) {
            if (!(updater instanceof AdaGrad))
                throw new UnsupportedOperationException("Cannot aggregate AdaGrad with updater: " + updater);
            AdaGrad adagrad = (AdaGrad) updater;
            if (historicalGradientSum == null) {
                historicalGradientSum = adagrad.historicalGradient.dup();
                lrSum = adagrad.learningRate;
                numIterationsSum = adagrad.numIterations;
            } else {
                historicalGradientSum.addi(adagrad.historicalGradient);
                lrSum += adagrad.learningRate;
                numIterationsSum += adagrad.numIterations;
            }
            count++;
        }

        @Override
        public GradientUpdaterAggregator combine(GradientUpdaterAggregator other) {
            if (!(other instanceof AdaGradAggregator))
                throw new IllegalArgumentException("Cannot combine AdaGradAggregator with aggregator: " + other);
            AdaGradAggregator aggregator = (AdaGradAggregator) other;
            historicalGradientSum.addi(aggregator.historicalGradientSum);
            lrSum += aggregator.lrSum;
            numIterationsSum += aggregator.numIterationsSum;
            count += aggregator.count;
            return this;
        }
    }
}
