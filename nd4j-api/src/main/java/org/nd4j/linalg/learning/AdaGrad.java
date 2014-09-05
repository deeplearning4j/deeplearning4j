package org.nd4j.linalg.learning;


import static org.nd4j.linalg.ops.transforms.Transforms.*;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.io.Serializable;




/**
 *
 * Vectorized Learning Rate used per Connection Weight
 *
 * Adapted from: http://xcorr.net/2014/01/23/adagrad-eliminating-learning-rates-in-stochastic-gradient-descent/
 *
 * @author Adam Gibson
 *
 */
public class AdaGrad implements Serializable {

    /**
     *
     */
    protected static final long serialVersionUID = -4754127927704099888L;
    protected double masterStepSize = 1e-1; // default for masterStepSize (this is the numerator)
    //protected double squaredGradientSum = 0;
    public INDArray historicalGradient;
    public INDArray adjustedGradient;
    public double fudgeFactor = 1e-6;
    public INDArray gradient;
    public int[] shape;
    protected int numIterations = 0;
    protected double lrDecay = 0.95;
    protected boolean decayLr;
    protected double minLearningRate = 1e-4;

    public AdaGrad( int rows, int cols, double gamma) {
        this.shape = new int[]{rows,cols};
        createHistoricalGradient();
        createAdjustedGradient();
        this.masterStepSize = gamma;
        this.decayLr = false;


    }


    /**
     * Create adagrad with the specified shape
     * @param shape
     */
    public AdaGrad(int[] shape) {
        this.shape = shape;
        createHistoricalGradient();
        createAdjustedGradient();
        this.masterStepSize = 1e-1;
        this.decayLr = false;


    }

    /**
     * Initializes adagrad with a gamma of 1e-2
     * @param rows the rows for the gradients
     * @param cols the number of columns for the gradient
     */
    public AdaGrad( int rows, int cols) {
        this(rows,cols,0.1);

    }

    protected void createHistoricalGradient() {
        this.historicalGradient = Nd4j.create(shape);

    }
    protected void createAdjustedGradient() {
        this.adjustedGradient = Nd4j.create(shape);
    }






    /**
     * Gets feature specific learning rates
     * Adagrad keeps a history of gradients being passed in.
     * Note that each gradient passed in becomes adapted over time, hence
     * the name adagrad
     * @param gradient the gradient to getFromOrigin learning rates for
     * @return the feature specific learning rates
     */
    public INDArray getLearningRates(INDArray gradient) {
        this.gradient = gradient;
        INDArray squaredGradient = pow(this.gradient,2);
        if(this.historicalGradient == null || this.historicalGradient.length() != this.gradient.length())
            this.historicalGradient = Nd4j.zeros(this.gradient.rows(), this.gradient.columns());
        this.historicalGradient.addi(squaredGradient);
        numIterations++;
        INDArray sqrtGradient = sqrt(historicalGradient).addi(fudgeFactor);
        INDArray div = abs(gradient).divi(sqrtGradient);
        this.adjustedGradient = div.muli(masterStepSize);
        //ensure no zeros
        return adjustedGradient;
    }

    public  double getMasterStepSize() {
        return masterStepSize;
    }

    public  void setMasterStepSize(double masterStepSize) {
        this.masterStepSize = masterStepSize;
    }

    public synchronized boolean isDecayLr() {
        return decayLr;
    }

    public synchronized void setDecayLr(boolean decayLr) {
        this.decayLr = decayLr;
    }




}