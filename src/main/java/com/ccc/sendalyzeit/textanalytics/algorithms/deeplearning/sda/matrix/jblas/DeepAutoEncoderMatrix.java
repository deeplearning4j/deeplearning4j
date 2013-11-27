package com.ccc.sendalyzeit.textanalytics.algorithms.deeplearning.sda.matrix.jblas;

import java.io.Serializable;
import java.util.Random;

import org.jblas.DoubleMatrix;

public class DeepAutoEncoderMatrix implements Serializable  {
	/**
	 * 
	 */
	private static final long serialVersionUID = -6445530486350763837L;
	public int N;
	public int n_visible;
	public int n_hidden;
	public DoubleMatrix W;
	public DoubleMatrix hbias;
	public DoubleMatrix vbias;
	public Random rng;


	public double uniform(double min, double max) {
		return rng.nextDouble() * (max - min) + min;
	}

	public int binomial(int n, double p) {
		if(p < 0 || p > 1) return 0;

		int c = 0;
		double r;

		for(int i = 0; i<n; i++) {
			r = rng.nextDouble();
			if (r < p) c++;
		}

		return c;
	}

	public static double sigmoid(double x) {
		return 1.0 / (1.0 + Math.pow(Math.E, -x));
	}
	/**
	 * 
	 * @param N the number of training examples
	 * @param n_visible the number of outbound nodes
	 * @param n_hidden the number of nodes in the hidden layer
	 * @param W the weights for this vector, maybe null, if so this will
	 * create a matrix with n_hidden x n_visible dimensions.
	 * @param hbias the hidden bias
	 * @param vbias the visible bias (usually b for the output layer)
	 * @param rng the rng, if not a seed of 1234 is used.
	 */
	public DeepAutoEncoderMatrix(int N, int n_visible, int n_hidden, 
			DoubleMatrix W, DoubleMatrix hbias, DoubleMatrix vbias, Random rng) {
		this.N = N;
		this.n_visible = n_visible;
		this.n_hidden = n_hidden;

		if(rng == null)	
			this.rng = new Random(1234);
		else 
			this.rng = rng;

		if(W == null) 
			this.W = DoubleMatrix.randn(n_hidden,n_visible);


		else if(W.rows != n_hidden && W.columns != n_visible)
			throw new IllegalArgumentException("Illegal weight vector; The number of rows are not equal to " + n_hidden + " or the number of columns aren't equal to " +  n_visible + " dimensions were " + W.rows + " x " + W.columns);
		else	
			this.W = W;


		if(hbias == null) 
			this.hbias = DoubleMatrix.zeros(n_hidden);

		else if(hbias.length != n_hidden)
			throw new IllegalArgumentException("Hidden bias must have a length of " + n_hidden + " length was " + hbias.length);

		else
			this.hbias = hbias;

		if(vbias == null) 
			this.vbias = DoubleMatrix.zeros(n_visible);

		else if(vbias.length != n_visible) 
			throw new IllegalArgumentException("Visible bias must have a length of " + n_visible + " but length was " + vbias.length);

		else 
			this.vbias = vbias;
	}

	public DoubleMatrix get_corrupted_input(DoubleMatrix x, final double p) {
		if(x.length != n_visible)
			throw new IllegalArgumentException("x must have a length of " + n_visible);


		final DoubleMatrix tilde_x = DoubleMatrix.zeros(n_visible);
		for(int i = 0; i < n_visible; i++)
			tilde_x.put(i,binomial(1,p));

		return tilde_x;

	}


	public static DoubleMatrix sigmoid(DoubleMatrix x) {
		DoubleMatrix matrix = new DoubleMatrix(x.rows,x.columns);
		for(int i = 0; i < matrix.length; i++)
			matrix.put(i, 1f / (1f + Math.pow(Math.E, -x.get(i))));

		return matrix;
	}

	// Encode
	public DoubleMatrix get_hidden_values(DoubleMatrix x) {
		if(!x.isColumnVector())
			x = x.transpose();

		return sigmoid(W.mmul(x).add(hbias));
	}

	// Decode
	public DoubleMatrix get_reconstructed_input(DoubleMatrix y) {
		DoubleMatrix z = W.transpose().mmul(y);
		z = z.add(vbias);
		for(int i = 0; i < z.length; i++)
			z.put(i,sigmoid(z.get(i)));
		return z;
	}

	public void train(DoubleMatrix x, final double lr, double corruption_level) {

		if(!x.isColumnVector()) {
			if(x.isRowVector())
				x = x.transpose();
			else
				throw new IllegalStateException("Unable to transpose row matrix as input");
		}
		double p = 1 - corruption_level;

		DoubleMatrix tilde_x = get_corrupted_input(x, p);
		DoubleMatrix y = get_hidden_values(tilde_x);
		DoubleMatrix z = get_reconstructed_input(y);
		//vbias
		DoubleMatrix L_vbias = x.mini(z);
		for(int i = 0; i < n_visible; i++) 
			vbias.put(i,vbias.get(i) + lr * L_vbias.get(i) / N);

		DoubleMatrix L_hbias = W.mmul(L_vbias);

		// hbias
		for(int i=0; i < n_hidden; i++) {
			for(int j=0; j < n_visible; j++) {
				L_hbias.put(i, W.get(i,j) * L_vbias.get(j));
			}
			L_hbias.put(i, L_hbias.get(i) * y.get(i) * (1 - y.get(i)));
			hbias.put(i,hbias.get(i) + lr * L_hbias.get(i) / N);
		}

		for(int i = 0; i < W.rows; i++)
			for(int j = 0; j < W.columns; j++)
				W.put(i,j, W.get(i,j) + (lr * (L_hbias.get(i) * tilde_x.get(j) + L_vbias.get(j) * y.get(i)) / N));


	}

	public DoubleMatrix reconstruct(DoubleMatrix x) {
		DoubleMatrix y = get_hidden_values(x);
		return get_reconstructed_input(y);
	}	
}
