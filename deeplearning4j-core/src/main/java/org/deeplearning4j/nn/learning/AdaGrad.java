package org.deeplearning4j.nn.learning;

import java.io.Serializable;

import org.deeplearning4j.util.MatrixUtil;
import org.jblas.DoubleMatrix;
import org.jblas.MatrixFunctions;

/**
 * 
 * Vectorized Learning Rate used per Connection Weight
 * 
 * Adapted from: https://github.com/jpatanooga/Metronome/blob/master/src/main/java/tv/floe/metronome/deeplearning/neuralnetwork/core/learning/AdagradLearningRate.java
 * 
 * @author Adam Gibson
 *
 */
public class AdaGrad implements Serializable {

	/**
	 * 
	 */
	private static final long serialVersionUID = -4754127927704099888L;
	private double masterStepSize = 0.01; // default for masterStepSize (this is the numerator)
	//private double squaredGradientSum = 0;
	public DoubleMatrix historicalGradient;
	public DoubleMatrix adjustedGradient;
	public double fudgeFactor = 0.000001;
	public DoubleMatrix gradient;
	public int rows;
	public int cols;
	private boolean first = true;

	public double autoCorrect = 0.95;

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
	 * square gradient, then add to ongoing sum
	 * 
	 * @param gradient
	 */
	private void addLastIterationGradient() {
		//this.squaredGradientSum += gradient * gradient;

		DoubleMatrix gradientsSquared = MatrixFunctions.pow(gradient, 2);
		if(first) 
			this.historicalGradient = gradientsSquared;

		else 
			this.historicalGradient = this.historicalGradient.mul(this.autoCorrect).add(MatrixUtil.oneMinus(historicalGradient)).mul(gradientsSquared);




	}




	public DoubleMatrix getLearningRates(DoubleMatrix gradient) {
		this.gradient = gradient;
		addLastIterationGradient();
		DoubleMatrix gAdd = MatrixFunctions.sqrt(historicalGradient).add(fudgeFactor);
		this.adjustedGradient = gradient.div(gAdd);
		this.adjustedGradient.mul(masterStepSize);
		return adjustedGradient;
	}


}