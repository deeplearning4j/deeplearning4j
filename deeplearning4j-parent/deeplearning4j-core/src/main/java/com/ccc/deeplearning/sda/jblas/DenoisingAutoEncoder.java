package com.ccc.deeplearning.sda.jblas;

import static com.ccc.deeplearning.util.MatrixUtil.log;
import static com.ccc.deeplearning.util.MatrixUtil.oneMinus;
import static com.ccc.deeplearning.util.MatrixUtil.sigmoid;

import java.io.Serializable;

import org.apache.commons.math3.random.RandomGenerator;
import org.jblas.DoubleMatrix;

import com.ccc.deeplearning.nn.matrix.jblas.BaseNeuralNetwork;
import com.ccc.deeplearning.optimize.NeuralNetworkOptimizer;
import static com.ccc.deeplearning.util.MathUtils.*;

/**
 * Denoising Autoencoder.
 * Add Gaussian noise to input and learn
 * a reconstruction function.
 * 
 * @author Adam Gibson
 *
 */
public class DenoisingAutoEncoder extends BaseNeuralNetwork implements Serializable  {


	private static final long serialVersionUID = -6445530486350763837L;

	public DenoisingAutoEncoder() {}




	public DenoisingAutoEncoder(int nVisible, int nHidden,
			DoubleMatrix W, DoubleMatrix hbias, DoubleMatrix vbias,
			RandomGenerator rng) {
		super(nVisible, nHidden, W, hbias, vbias, rng);
	}




	public DenoisingAutoEncoder(DoubleMatrix input, int n_visible, int n_hidden,
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
				tilde_x.put(i,j,binomial(rng,1,1 - corruptionLevel));
		DoubleMatrix  ret = tilde_x.mul(x);
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
		DoubleMatrix corrupted = getCorruptedInput(input, corruptionLevel);
		DoubleMatrix y = getHiddenValues(corrupted);
		DoubleMatrix z = getReconstructedInput(y);
		return - input.mul(log(z)).add(
				oneMinus(input).mul(log(oneMinus(z)))).
	    columnSums().mean();
	}



	// Encode
	public DoubleMatrix getHiddenValues(DoubleMatrix x) {
		return sigmoid(x.mmul(W).addRowVector(hBias));
	}

	// Decode
	public DoubleMatrix getReconstructedInput(DoubleMatrix y) {
		return sigmoid(y.mmul(W.transpose()).addRowVector(vBias));
	}


	/**
	 * Run a network optimizer
	 * @param x the input
	 * @param lr the learning rate
	 * @param corruptionLevel the corruption level
	 */
	public void trainTillConverge(DoubleMatrix x, double lr,double corruptionLevel) {
		optimizer = new DenoisingAutoEncoderOptimizer(this, lr, new Object[]{corruptionLevel});
		optimizer.train(x);
	}
	
	/**
	 * Perform one iteration of training
	 * @param x the input
	 * @param lr the learning rate
	 * @param corruptionLevel the corruption level to train with
	 */
	public void train(DoubleMatrix x, double lr, double corruptionLevel) {

		this.input = x;


		DoubleMatrix tildeX = getCorruptedInput(x, corruptionLevel);
		DoubleMatrix y = getHiddenValues(tildeX);
		DoubleMatrix z = getReconstructedInput(y);

		DoubleMatrix L_h2 = x.sub(z);

		DoubleMatrix L_h1 = L_h2.mmul(W).mul(y).mul(oneMinus(y));

		DoubleMatrix L_vbias = L_h2;
		DoubleMatrix L_hbias = L_h1;
		
		DoubleMatrix L_W = tildeX.transpose().mmul(L_h1).add(L_h2.transpose().mmul(y)).mul(lr);
		
		this.W = W.add(L_W).mul(momentum);
		//regularizeWeights(x.rows, lr);
		
		
		DoubleMatrix L_hbias_mean = L_hbias.columnMeans().mul(lr);
		DoubleMatrix L_vbias_mean = L_vbias.columnMeans().mul(lr);
	
		this.hBias = hBias.add(L_hbias_mean);
		this.vBias = vBias.add(L_vbias_mean);

	}

	@Override
	public DoubleMatrix reconstruct(DoubleMatrix x) {
		DoubleMatrix y = getHiddenValues(x);
		return getReconstructedInput(y);
	}	



	public static class Builder extends BaseNeuralNetwork.Builder<DenoisingAutoEncoder> {
		public Builder()  {
			this.clazz = DenoisingAutoEncoder.class;
		}
	}



	@Override
	public void trainTillConvergence(DoubleMatrix input, double lr,
			Object[] params) {
		double corruptionLevel = (double) params[0];
		trainTillConverge(input, lr, corruptionLevel);
	}




	@Override
	public double lossFunction(Object[] params) {
		double corruptionLevel = (double) params[0];
		return negativeLoglikelihood(corruptionLevel);
	}




	@Override
	public void train(DoubleMatrix input, double lr, Object[] params) {
		double corruptionLevel = (double) params[0];
		train(input, lr, corruptionLevel);
	}

}
