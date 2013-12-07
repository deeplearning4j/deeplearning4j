package com.ccc.sendalyzeit.textanalytics.algorithms.deeplearning.sda.matrix.jblas;

import java.io.Serializable;

import org.apache.commons.math3.distribution.UniformRealDistribution;
import org.apache.commons.math3.random.JDKRandomGenerator;
import org.jblas.DoubleMatrix;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.ccc.sendalyzeit.textanalytics.util.MathUtils;
import com.ccc.sendalyzeit.textanalytics.util.MatrixUtil;
/**
 * Invidiual denoising autoencoder.
 * 
 * Basic idea: add noise and reconstruct input
 * to approximate probabilities.
 * @author Adam Gibson
 *
 */
public class DenoisingAutoEncoderMatrix implements Serializable  {
	
	private static final long serialVersionUID = -6445530486350763837L;
	public int N;
	public int n_visible;
	public int n_hidden;
	//weight vector
	public DoubleMatrix W;
	//hidden or output bias
	public DoubleMatrix hbias;
	//visible or input bias
	public DoubleMatrix vbias;
	public JDKRandomGenerator rng;
	private static Logger log = LoggerFactory.getLogger(DenoisingAutoEncoderMatrix.class);





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
	public DenoisingAutoEncoderMatrix(int N, int n_visible, int n_hidden, 
			DoubleMatrix W, DoubleMatrix hbias, DoubleMatrix vbias, JDKRandomGenerator rng) {
		this.N = N;
		this.n_visible = n_visible;
		this.n_hidden = n_hidden;

		if(rng == null)	{
			this.rng = new JDKRandomGenerator();
			this.rng.setSeed(1);
		}
		else 
			this.rng = rng;
		//generate a scaled down uniform distribution
		//normal weights between 0 and 1 tend to 
		//get kind of large and cause some weird results
		if(W == null) {
			double a = 1.0 / (double) n_visible;
			UniformRealDistribution u = new UniformRealDistribution(-a,a);

			this.W = DoubleMatrix.zeros(n_visible,n_hidden);

			for(int i = 0; i < this.W.rows; i++) {
				for(int j = 0; j < this.W.columns; j++) 
					this.W.put(i,j,u.sample());
				
			}


		}
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
	/**
	 * Corrupt input using a binomial sampling distribution
	 * @param x
	 * @param p
	 * @return
	 */
	public DoubleMatrix get_corrupted_input(DoubleMatrix x, double p) {
		DoubleMatrix tilde_x = DoubleMatrix.zeros(x.rows,x.columns);
		//binomial sampling to add noise
		for(int i = 0; i < x.rows; i++)
			for(int j = 0; j < x.columns; j++)
				tilde_x.put(i,j,MathUtils.binomial(rng,1,p));
		//corrupted version of the input
		DoubleMatrix  ret = x.mul(tilde_x);
		return ret;
	}




	/**
	 * Get the output for this autoencoder.
	 * @param x
	 * @return
	 */
	public DoubleMatrix get_hidden_values(DoubleMatrix x) {
		DoubleMatrix mul = x.mmul(W);
		//note here that we are adding the bias to each
		//row of the vector
		return MatrixUtil.sigmoid(mul.addRowVector(hbias));
	}

	/**
	 * Reconstruct the input
	 * @param y the corrupted input
	 * @return
	 */
	public DoubleMatrix get_reconstructed_input(DoubleMatrix y) {
		DoubleMatrix z = y.mmul(W.transpose());
		//add visual bias to each row in the matrix
		z = z.addRowVector(vbias);
		z = MatrixUtil.sigmoid(z);
		return z;
	}

	public void train(DoubleMatrix x, double lr, double corruption_level) {


		double p = 1 - corruption_level;

		DoubleMatrix tilde_x = get_corrupted_input(x, p);
		DoubleMatrix y = get_hidden_values(tilde_x);
		DoubleMatrix z = get_reconstructed_input(y);

		DoubleMatrix L_h2 = x.sub(z);

		DoubleMatrix L_h1 = L_h2.mmul(W).mul(y).mul(DoubleMatrix.ones(y.length).sub(y));

		DoubleMatrix L_vbias = L_h2;
		DoubleMatrix L_hbias = L_h1;
		//L_W =  numpy.dot(tilde_x.T, L_h1) + numpy.dot(L_h2.T, y)
		//
		DoubleMatrix L_W = tilde_x.transpose().mmul(L_h1).add(L_h2.transpose().mmul(y));
		DoubleMatrix learnMulL_W = L_W.mul(lr);
		this.W = W.add(learnMulL_W);
		DoubleMatrix L_hbias_mean = MatrixUtil.columnWiseMean(L_hbias, 0);
		DoubleMatrix L_vbias_mean = MatrixUtil.columnWiseMean(L_vbias, 0);
		L_hbias_mean = L_hbias_mean.mul(lr);
		L_vbias_mean = L_vbias_mean.mul(lr);
		this.hbias = hbias.add(L_hbias_mean);
		this.vbias = vbias.add(L_vbias_mean);

	}
	/**
	 * Predict the input by reconstructing it
	 * @param x the input
	 * @return the reconstructed input
	 */
	public DoubleMatrix reconstruct(DoubleMatrix x) {
		DoubleMatrix y = get_hidden_values(x);
		return get_reconstructed_input(y);
	}	
}
