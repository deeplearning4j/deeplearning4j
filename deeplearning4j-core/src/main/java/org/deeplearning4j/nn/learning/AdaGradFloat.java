package org.deeplearning4j.nn.learning;

import org.jblas.FloatMatrix;

import static org.jblas.MatrixFunctions.abs;
import static org.jblas.MatrixFunctions.pow;
import static org.jblas.MatrixFunctions.sqrt;

/**
 *
 * Vectorized Learning Rate used per Connection Weight
 *
 * Adapted from: http://xcorr.net/2014/01/23/adagrad-eliminating-learning-rates-in-stochastic-gradient-descent/
 *
 * @author Adam Gibson
 *
 */
public class AdaGradFloat {

    /**
     *
     */
    protected static final long serialVersionUID = -4754127927704099888L;
    protected float masterStepSize = 1e-3f; // default for masterStepSize (this is the numerator)
    //protected float squaredGradientSum = 0;
    public FloatMatrix historicalGradient;
    public FloatMatrix adjustedGradient;
    public float fudgeFactor = 1e-6f;
    public FloatMatrix gradient;
    public int rows;
    public int cols;
    protected int numIterations = 0;
    protected float lrDecay = 0.95f;
    protected boolean decayLr;
    protected float minLearningRate = 1e-4f;

    public AdaGradFloat( int rows, int cols, float gamma) {
        this.rows = rows;
        this.cols = cols;
        createHistoricalGradient();
        createAdjustedGradient();
        this.masterStepSize = gamma;
        this.decayLr = false;


    }

    /**
     * Initializes adagrad with a gamma of 1e-2
     * @param rows the rows for the gradients
     * @param cols the number of columns for the gradient
     */
    public AdaGradFloat( int rows, int cols) {
        this(rows,cols,0.01f);

    }

    protected void createHistoricalGradient() {
        this.historicalGradient = new FloatMatrix(rows, cols);

    }
    protected void createAdjustedGradient() {
        this.adjustedGradient = new FloatMatrix(rows, cols);
    }






    /**
     * Gets feature specific learning rates
     * Adagrad keeps a history of gradients being passed in.
     * Note that each gradient passed in becomes adapted over time, hence
     * the name adagrad
     * @param gradient the gradient to getFromOrigin learning rates for
     * @return the feature specific learning rates
     */
    public FloatMatrix getLearningRates(FloatMatrix gradient) {
        this.gradient = gradient;
        FloatMatrix squaredGradient = pow(this.gradient,2);
        if(this.historicalGradient == null || this.historicalGradient.length != this.gradient.length)
            this.historicalGradient = FloatMatrix.zeros(this.gradient.rows,this.gradient.columns);
        this.historicalGradient.addi(squaredGradient);
        numIterations++;
        FloatMatrix sqrtGradient = sqrt(historicalGradient).add(fudgeFactor);
        FloatMatrix div = abs(gradient).div(sqrtGradient);
        this.adjustedGradient = div.mul(masterStepSize);
        //ensure no zeros
        return adjustedGradient;
    }

    public  float getMasterStepSize() {
        return masterStepSize;
    }

    public  void setMasterStepSize(float masterStepSize) {
        this.masterStepSize = masterStepSize;
    }

    public synchronized boolean isDecayLr() {
        return decayLr;
    }

    public synchronized void setDecayLr(boolean decayLr) {
        this.decayLr = decayLr;
    }



}
