package com.ccc.sendalyzeit.textanalytics.algorithms.deeplearning.nn.matrix.jblas;

import org.apache.commons.math3.distribution.UniformRealDistribution;
import org.apache.commons.math3.random.MersenneTwister;
import org.apache.commons.math3.random.RandomGenerator;
import org.jblas.DoubleMatrix;

public abstract class BaseNeuralNetwork implements NeuralNetwork {

	public int n_visible;
	public int n_hidden;
	public DoubleMatrix W;
	public DoubleMatrix hBias;
	public DoubleMatrix vBias;
	public RandomGenerator rng;
	public DoubleMatrix input;
	
	/**
	 * 
	 * @param N the number of training examples
	 * @param n_visible the number of outbound nodes
	 * @param n_hidden the number of nodes in the hidden layer
	 * @param W the weights for this vector, maybe null, if so this will
	 * create a matrix with n_hidden x n_visible dimensions.
	 * @param hBias the hidden bias
	 * @param vBias the visible bias (usually b for the output layer)
	 * @param rng the rng, if not a seed of 1234 is used.
	 */
	public BaseNeuralNetwork(DoubleMatrix input, int n_visible, int n_hidden, 
			DoubleMatrix W, DoubleMatrix hbias, DoubleMatrix vbias, RandomGenerator rng) {
		this.input = input;
		this.n_visible = n_visible;
		this.n_hidden = n_hidden;

		if(rng == null)	
			this.rng = new MersenneTwister(1234);
		
		else 
			this.rng = rng;

		if(W == null) {
			double a = 1.0 / (double) n_visible;
			UniformRealDistribution u = new UniformRealDistribution(rng,-a,a,UniformRealDistribution.DEFAULT_INVERSE_ABSOLUTE_ACCURACY);

			this.W = DoubleMatrix.zeros(n_visible,n_hidden);

			for(int i = 0; i < this.W.rows; i++) {
				for(int j = 0; j < this.W.columns; j++) 
					this.W.put(i,j,u.sample());
				
			}


		}
		else	
			this.W = W;


		if(hbias == null) 
			this.hBias = DoubleMatrix.zeros(n_hidden);

		else if(hbias.length != n_hidden)
			throw new IllegalArgumentException("Hidden bias must have a length of " + n_hidden + " length was " + hbias.length);

		else
			this.hBias = hbias;

		if(vbias == null) 
			this.vBias = DoubleMatrix.zeros(n_visible);

		else if(vbias.length != n_visible) 
			throw new IllegalArgumentException("Visible bias must have a length of " + n_visible + " but length was " + vbias.length);

		else 
			this.vBias = vbias;
	}

	
}
