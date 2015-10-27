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
 *
 * @author Adam Gibson
 */
@Data
@NoArgsConstructor
public class AdaGrad implements Serializable,GradientUpdater {

    //protected double squaredGradientSum = 0;
    public INDArray historicalGradient;
    public int[] shape;
    protected double learningRate = 1e-1; // learning rate
    protected int numIterations = 0;
    private double epsilon = 1e-8;
    private double momentum = 0.5;

    /**
     *
     * @param rows
     * @param cols
     * @param learningRate
     */
    public AdaGrad(int rows, int cols, double learningRate) {
        this.shape = new int[]{rows, cols};
        this.learningRate = learningRate;
    }

    public AdaGrad(int rows, int cols){
        this(rows, cols, 0.1);
    }

    public AdaGrad(int[] shape, double learningRate) {
        this.shape = shape;
        this.learningRate = learningRate;
    }

    public AdaGrad(double learningRate) {
        this.learningRate = learningRate;
    }

    /**
     * Gets feature specific learning rates
     * Adagrad keeps a history of gradients being passed in.
     * Note that each gradient passed in becomes adapted over time, hence
     * the name adagrad
     *
     * @param gradient the gradient to get learning rates for
     * @param iteration
     * @return the feature specific learning rates
     */
    @Override
    public INDArray getGradient(INDArray gradient, int iteration) {
        if(historicalGradient == null) historicalGradient = gradient.mul(gradient);
        else historicalGradient.addi(gradient.mul(gradient));

        INDArray sqrtHistory = sqrt(historicalGradient.add(epsilon));
        // lr * gradient / sqrt(sumSquaredGradients + 1e-8)
        INDArray ret = sqrtHistory.rdivi(learningRate).muli(gradient);
        numIterations++;
        return ret;
    }

    public double getGradient(double gradient, int column, int[] shape) {
        boolean historicalInitialized = false;
        if (this.historicalGradient == null) {
                this.historicalGradient = Nd4j.ones(shape);
                historicalInitialized = true;
            }

        double sqrtHistory = !historicalInitialized ? Math.sqrt(historicalGradient.getDouble(column)) : historicalGradient.getDouble(column);
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
            this.historicalGradient = Nd4j.ones(shape);
            historicalInitialized = true;
        } else if (!this.historicalGradient.isVector() && this.historicalGradient.slice(slice).length() != gradient.length())
            throw new IllegalArgumentException("Illegal gradient");

        if (historicalGradient.isVector())
            sqrtHistory = sqrt(historicalGradient);
        else
            sqrtHistory = !historicalInitialized ? sqrt(historicalGradient.slice(slice)) : historicalGradient;
        INDArray learningRates = sqrtHistory.add(epsilon).rdivi(learningRate);
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