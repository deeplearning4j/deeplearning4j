package com.ccc.sendalyzeit.textanalytics.algorithms.deeplearning.sda.matrix.jblas;

import java.io.Serializable;

import org.apache.commons.math3.random.RandomGenerator;
import org.jblas.DoubleMatrix;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

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
	private static Logger log = LoggerFactory.getLogger(DenoisingAutoEncoderMatrix.class);
	
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

	/**
	 * Corrupts the given input by doing a binomial sampling
	 * given the corruption level
	 * @param x the input to corrupt
	 * @param corruptionLevel the corruption value
	 * @return the binomial sampled corrupted input
	 */
	public DoubleMatrix getCorruptedInput(DoubleMatrix x, double corruptionLevel) {
		DoubleMatrix tilde_x = DoubleMatrix.zeros(x.rows,x.columns);
		for(int i = 0; i < x.rows; i++)
			for(int j = 0; j < x.columns; j++)
				tilde_x.put(i,j,MathUtils.binomial(rng,1,corruptionLevel));
		DoubleMatrix  ret = x.mul(tilde_x);
		return ret;
	}


	


	/**
	 * Negative log likelihood of the current input given
	 * the corruption level
	 * @param corruptionLevel the corruption level to use
	 * @return the negative log likelihood of the auto encoder
	 * given the corruption level
	 */
	public double negativeLoglikelihood(double corruptionLevel) {
		DoubleMatrix corrupted = this.getCorruptedInput(input, corruptionLevel);
		DoubleMatrix y = this.getHiddenValues(corrupted);
		DoubleMatrix z = this.getReconstructedInput(y);
		DoubleMatrix inside = input.mul(MatrixUtil.log(z)).add(MatrixUtil.oneMinus(input).mul(MatrixUtil.log(MatrixUtil.oneMinus(z))));
		return - inside.columnSums().mean();
	}



	// Encode
	public DoubleMatrix getHiddenValues(DoubleMatrix x) {
		DoubleMatrix mul = x.mmul(W);
		return MatrixUtil.sigmoid(mul.addRowVector(hBias));
	}

	// Decode
	public DoubleMatrix getReconstructedInput(DoubleMatrix y) {
		DoubleMatrix z = y.mmul(W.transpose());
		z = z.addRowVector(vBias);
		z = MatrixUtil.sigmoid(z);
		return z;
	}

	public void train(DoubleMatrix x, double lr, double corruption_level) {

		this.input = x;
		
		double p = 1 - corruption_level;

		DoubleMatrix tilde_x = getCorruptedInput(x, p);
		DoubleMatrix y = getHiddenValues(tilde_x);
		DoubleMatrix z = getReconstructedInput(y);

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
		log.info("Training negative log likelihood " + this.negativeLoglikelihood(corruption_level));

	}

	public DoubleMatrix reconstruct(DoubleMatrix x) {
		DoubleMatrix y = getHiddenValues(x);
		return getReconstructedInput(y);
	}	
	
	
	
	public static class Builder extends BaseNeuralNetwork.Builder<DenoisingAutoEncoderMatrix> {
		public Builder()  {
			this.clazz = DenoisingAutoEncoderMatrix.class;
		}
	}
	
}
