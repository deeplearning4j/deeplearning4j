package org.deeplearning4j.nn.learning;

import java.io.Serializable;

import static org.jblas.MatrixFunctions.sqrt;
import static org.jblas.MatrixFunctions.pow;
import static org.jblas.MatrixFunctions.abs;


import org.jblas.DoubleMatrix;
import org.jblas.MatrixFunctions;

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
	private static final long serialVersionUID = -4754127927704099888L;
	private double masterStepSize = 1e-3; // default for masterStepSize (this is the numerator)
	//private double squaredGradientSum = 0;
	public DoubleMatrix historicalGradient;
	public DoubleMatrix adjustedGradient;
	public double fudgeFactor = 1e-6;
	public DoubleMatrix gradient;
	public int rows;
	public int cols;
	private int numIterations = 0;
	private double lrDecay = 0.95;
	private boolean decayLr;
	private double minLearningRate = 1e-4;

	public AdaGrad( int rows, int cols, double gamma) {
		this.rows = rows;
		this.cols = cols;
		this.adjustedGradient = new DoubleMatrix(rows, cols);
		this.historicalGradient = new DoubleMatrix(rows, cols);
		this.masterStepSize = gamma;
		this.decayLr = false;


	}

	public AdaGrad( int rows, int cols) {
		this(rows,cols,0.01);

	}





	/**
	 * Gets feature specific learning rates
	 * Adagrad keeps a history of gradients being passed in.
	 * Note that each gradient passed in becomes adapted over time, hence
	 * the name adagrad
	 * @param gradient the gradient to get learning rates for
	 * @return the feature specific learning rates
	 */
	public DoubleMatrix getLearningRates(DoubleMatrix gradient) {
		this.gradient = gradient.dup();
        DoubleMatrix squaredGradient = pow(this.gradient,2);
        if(this.historicalGradient.length != this.gradient.length)
            this.historicalGradient = DoubleMatrix.zeros(this.gradient.rows,this.gradient.columns);
        this.historicalGradient.addi(squaredGradient);
        numIterations++;
        DoubleMatrix sqrtGradient = sqrt(squaredGradient).add(fudgeFactor);
        DoubleMatrix div = abs(gradient).div(sqrtGradient);
		this.adjustedGradient = div.mul(masterStepSize);
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