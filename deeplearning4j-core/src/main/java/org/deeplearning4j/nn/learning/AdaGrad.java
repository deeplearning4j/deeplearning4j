package org.deeplearning4j.nn.learning;

import java.io.Serializable;

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
	public double fudgeFactor = 0.000001;
	public DoubleMatrix gradient;
	public int rows;
	public int cols;
	

	public AdaGrad( int rows, int cols, double gamma) {
		this.rows = rows;
		this.cols = cols;
		this.adjustedGradient = new DoubleMatrix(rows, cols);
		this.historicalGradient = new DoubleMatrix(rows, cols);
		this.masterStepSize = gamma;



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
		//lr annealing
		double currentLearningRate = this.masterStepSize;
		this.adjustedGradient = MatrixFunctions.sqrt(MatrixFunctions.pow(this.gradient,2)).mul(currentLearningRate);
		return adjustedGradient;
	}

	public  double getMasterStepSize() {
		return masterStepSize;
	}

	public  void setMasterStepSize(double masterStepSize) {
		this.masterStepSize = masterStepSize;
	}




}