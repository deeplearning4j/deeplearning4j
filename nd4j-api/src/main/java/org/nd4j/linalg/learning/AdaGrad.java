/*
 * Copyright 2015 Skymind,Inc.
 *
 *    Licensed under the Apache License, Version 2.0 (the "License");
 *    you may not use this file except in compliance with the License.
 *    You may obtain a copy of the License at
 *
 *        http://www.apache.org/licenses/LICENSE-2.0
 *
 *    Unless required by applicable law or agreed to in writing, software
 *    distributed under the License is distributed on an "AS IS" BASIS,
 *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *    See the License for the specific language governing permissions and
 *    limitations under the License.
 */

package org.nd4j.linalg.learning;


import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.util.Shape;

import java.io.Serializable;

import static org.nd4j.linalg.ops.transforms.Transforms.pow;
import static org.nd4j.linalg.ops.transforms.Transforms.sqrt;


/**
 * Vectorized Learning Rate used per Connection Weight
 * <p/>
 * Adapted from: http://xcorr.net/2014/01/23/adagrad-eliminating-learning-rates-in-stochastic-gradient-descent/
 *
 * @author Adam Gibson
 */
public class AdaGrad implements Serializable {

    /**
     *
     */
    protected static final long serialVersionUID = -4754127927704099888L;
    //protected double squaredGradientSum = 0;
    public INDArray historicalGradient;
    public int[] shape;
    protected double masterStepSize = 1e-1; // default for masterStepSize (this is the numerator)
    protected int numIterations = 0;
    protected boolean decayLr;

    public AdaGrad(int rows, int cols, double gamma) {
        this.shape = new int[]{rows, cols};
        this.masterStepSize = gamma;
        this.decayLr = false;


    }


    /**
     * Create adagrad with the specified shape
     *
     * @param shape
     */
    public AdaGrad(int[] shape) {
        this.shape = shape;
        this.masterStepSize = 1e-1;
        this.decayLr = false;


    }

    /**
     * Initializes adagrad with a gamma of 1e-2
     *
     * @param rows the rows for the gradients
     * @param cols the number of columns for the gradient
     */
    public AdaGrad(int rows, int cols) {
        this(rows, cols, 0.1);

    }


    /**
     * Gets feature specific learning rates
     * Adagrad keeps a history of gradients being passed in.
     * Note that each gradient passed in becomes adapted over time, hence
     * the name adagrad
     *
     * @param gradient the gradient to get learning rates for
     * @param column   the slice of the gradient history to use
     * @param shape    the shape of the nd array for the historical gradient
     * @return the feature specific learning rates
     */
    public double getGradient(double gradient, int column, int[] shape) {
        boolean historicalInitialized = false;
        if (this.historicalGradient == null) {
            this.historicalGradient = Nd4j.ones(shape);
            historicalInitialized = true;
        }

        double sqrtHistory = !historicalInitialized ? Math.sqrt(historicalGradient.getDouble(column)) : historicalGradient.getDouble(column);
        double learningRates = (masterStepSize) / sqrtHistory;
        double adjustedGradient = gradient * (learningRates);

        historicalGradient.putScalar(column, historicalGradient.getDouble(column) + Math.pow(gradient, 2));
        numIterations++;

        //ensure no zeros
        return adjustedGradient;
    }


    /**
     * @param index
     * @return
     */
    public AdaGrad createSubset(int index) {
        if (historicalGradient == null)
            this.historicalGradient = Nd4j.ones(shape);

        if (Shape.isMatrix(shape)) {
            AdaGrad a = new AdaGrad(1, historicalGradient.columns());
            //grab only the needed elements
            INDArray slice = historicalGradient.slice(index).dup();
            a.historicalGradient = slice;
            a.setMasterStepSize(masterStepSize);
            a.setDecayLr(decayLr);
            return a;
        } else {
            AdaGrad a = new AdaGrad(1, 1);
            //grab only the needed elements
            INDArray slice = Nd4j.scalar(historicalGradient.getDouble(index));
            a.historicalGradient = slice;
            a.setMasterStepSize(masterStepSize);
            a.setDecayLr(decayLr);
            return a;
        }

    }

    /**
     * Gets feature specific learning rates
     * Adagrad keeps a history of gradients being passed in.
     * Note that each gradient passed in becomes adapted over time, hence
     * the name adagrad
     *
     * @param gradient the gradient to get learning rates for
     * @param slice    the slice of the gradient history to use
     * @param shape    the shape of the nd array for the historical gradient
     * @return the feature specific learning rates
     */
    public INDArray getGradient(INDArray gradient, int slice, int[] shape) {
        boolean historicalInitialized = false;
        if (this.historicalGradient == null) {
            this.historicalGradient = Nd4j.ones(shape);
            historicalInitialized = true;
        } else if (!this.historicalGradient.isVector() && this.historicalGradient.slice(slice).length() != gradient.length())
            throw new IllegalArgumentException("Illegal gradient");

        INDArray sqrtHistory = null;
        if (historicalGradient.isVector())
            sqrtHistory = sqrt(historicalGradient);

        else
            sqrtHistory = !historicalInitialized ? sqrt(historicalGradient.slice(slice)) : historicalGradient;
        INDArray learningRates = sqrtHistory.rdivi(masterStepSize);
        gradient.muli(learningRates);

        this.historicalGradient.slice(slice).addi(pow(gradient, 2));
        numIterations++;

        //ensure no zeros
        return gradient;
    }


    /**
     * Gets feature specific learning rates
     * Adagrad keeps a history of gradients being passed in.
     * Note that each gradient passed in becomes adapted over time, hence
     * the name adagrad
     *
     * @param gradient the gradient to get learning rates for
     * @return the feature specific learning rates
     */
    public INDArray getGradient(INDArray gradient) {
        boolean historicalInitialized = false;
        if (this.historicalGradient == null) {
            this.historicalGradient = Nd4j.ones(gradient.rows(), gradient.columns());
            historicalInitialized = true;
        } else if (this.historicalGradient.length() != gradient.length())
            throw new IllegalArgumentException("Illegal gradient");


        INDArray sqrtHistory = !historicalInitialized ? sqrt(historicalGradient) : historicalGradient;
        INDArray learningRates = sqrtHistory.rdivi(masterStepSize);
        gradient.muli(learningRates);

        this.historicalGradient.addi(pow(gradient, 2));
        numIterations++;

        //ensure no zeros
        return gradient;
    }

    public double getMasterStepSize() {
        return masterStepSize;
    }

    public void setMasterStepSize(double masterStepSize) {
        this.masterStepSize = masterStepSize;
    }

    public boolean isDecayLr() {
        return decayLr;
    }

    public void setDecayLr(boolean decayLr) {
        this.decayLr = decayLr;
    }


}