package com.ccc.sendalyzeit.textanalytics.algorithms.deeplearning.sda.matrix.jblas;

import java.io.Serializable;

import org.apache.commons.math3.random.RandomGenerator;
import org.jblas.DoubleMatrix;

import com.ccc.sendalyzeit.textanalytics.algorithms.deeplearning.nn.matrix.jblas.BaseNeuralNetwork;
import com.ccc.sendalyzeit.textanalytics.util.MathUtils;
import com.ccc.sendalyzeit.textanalytics.util.MatrixUtil;
/**
 * Denoising autoencoder
 * @author Adam Gibson
 *
 */
public class DenoisingAutoEncoderMatrix extends BaseNeuralNetwork implements Serializable  {
	

	private static final long serialVersionUID = -6445530486350763837L;

	
	public DenoisingAutoEncoderMatrix() {}

	
	
	
	public DenoisingAutoEncoderMatrix(int nVisible, int nHidden,
			DoubleMatrix W, DoubleMatrix hbias, DoubleMatrix vbias,
			RandomGenerator rng) {
		super(nVisible, nHidden, W, hbias, vbias, rng);
	}




	public DenoisingAutoEncoderMatrix(DoubleMatrix input, int n_visible, int n_hidden,
			DoubleMatrix W, DoubleMatrix hbias, DoubleMatrix vbias,
			RandomGenerator rng) {
		super(input, n_visible, n_hidden, W, hbias, vbias, rng);
	}


	public DoubleMatrix get_corrupted_input(DoubleMatrix x, double p) {
		DoubleMatrix tilde_x = DoubleMatrix.zeros(x.rows,x.columns);
		for(int i = 0; i < x.rows; i++)
			for(int j = 0; j < x.columns; j++)
				tilde_x.put(i,j,MathUtils.binomial(rng,1,p));
		DoubleMatrix  ret = x.mul(tilde_x);
		return ret;
	}




	// Encode
	public DoubleMatrix get_hidden_values(DoubleMatrix x) {
		DoubleMatrix mul = x.mmul(W);
		return MatrixUtil.sigmoid(mul.addRowVector(hBias));
	}

	// Decode
	public DoubleMatrix get_reconstructed_input(DoubleMatrix y) {
		DoubleMatrix z = y.mmul(W.transpose());
		z = z.addRowVector(vBias);
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
		DoubleMatrix L_hbias_mean = L_hbias.columnMeans();
		DoubleMatrix L_vbias_mean = L_vbias.columnMeans();
		L_hbias_mean = L_hbias_mean.mul(lr);
		L_vbias_mean = L_vbias_mean.mul(lr);
		this.hBias = hBias.add(L_hbias_mean);
		this.vBias = vBias.add(L_vbias_mean);

	}

	public DoubleMatrix reconstruct(DoubleMatrix x) {
		DoubleMatrix y = get_hidden_values(x);
		return get_reconstructed_input(y);
	}	
	
	
	
	public static class Builder extends BaseNeuralNetwork.Builder<DenoisingAutoEncoderMatrix> {
		public Builder()  {
			this.clazz = DenoisingAutoEncoderMatrix.class;
		}
	}
	
}
