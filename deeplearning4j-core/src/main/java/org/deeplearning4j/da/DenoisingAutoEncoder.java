package org.deeplearning4j.da;

import static org.deeplearning4j.util.MathUtils.binomial;
import static org.deeplearning4j.util.MatrixUtil.oneMinus;
import static org.deeplearning4j.util.MatrixUtil.sigmoid;

import java.io.Serializable;

import org.apache.commons.math3.distribution.RealDistribution;
import org.apache.commons.math3.random.RandomGenerator;
import org.deeplearning4j.nn.BaseNeuralNetwork;
import org.deeplearning4j.nn.gradient.NeuralNetworkGradient;
import org.deeplearning4j.sda.DenoisingAutoEncoderOptimizer;
import org.jblas.DoubleMatrix;


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


	public DenoisingAutoEncoder(DoubleMatrix input, int nVisible, int nHidden,
			DoubleMatrix W, DoubleMatrix hbias, DoubleMatrix vbias,
			RandomGenerator rng,double fanIn,RealDistribution dist) {
		super(input, nVisible, nHidden, W, hbias, vbias, rng,fanIn,dist);
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
	public void trainTillConvergence(DoubleMatrix x, double lr,double corruptionLevel) {
		if(x != null)
			this.input = x;
		optimizer = new DenoisingAutoEncoderOptimizer(this, lr, new Object[]{corruptionLevel,lr});
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
		NeuralNetworkGradient gradient = getGradient(new Object[]{corruptionLevel,lr});
		vBias.addi(gradient.getvBiasGradient());
		W.addi(gradient.getwGradient());
		hBias.addi(gradient.gethBiasGradient());

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
		if(input != null)
			this.input = input;
		optimizer = new DenoisingAutoEncoderOptimizer(this, lr, params);
		optimizer.train(input);
	}




	@Override
	public double lossFunction(Object[] params) {
		return negativeLoglikelihood();
	}




	@Override
	public void train(DoubleMatrix input, double lr, Object[] params) {
		double corruptionLevel = (double) params[0];
		train(input, lr, corruptionLevel);
	}




	@Override
	public synchronized NeuralNetworkGradient getGradient(Object[] params) {

		double corruptionLevel = (double) params[0];
		double lr = (double) params[1];

		DoubleMatrix tildeX = getCorruptedInput(input, corruptionLevel);
		DoubleMatrix y = getHiddenValues(tildeX);
		DoubleMatrix z = getReconstructedInput(y);

		DoubleMatrix L_h2 =  input.sub(z) ;

		DoubleMatrix L_h1 = sparsity == 0 ? L_h2.mmul(W).mul(y).mul(oneMinus(y)) : L_h2.mmul(W).mul(y).mul(y.add(- sparsity));

		DoubleMatrix L_vbias = L_h2;
		DoubleMatrix L_hbias = L_h1;

		DoubleMatrix L_W = tildeX.transpose().mmul(L_h1).add(L_h2.transpose().mmul(y));
		
		if(useAdaGrad)
		   L_W.muli(wAdaGrad.getLearningRates(L_W));
		else 
			L_W.muli(lr);


		if(useRegularization) 
			L_W.subi(W.muli(l2));
		

		if(momentum != 0)
			L_W.muli(1 - momentum);
		L_W.divi(input.rows);



		DoubleMatrix L_hbias_mean = L_hbias.columnMeans();
		DoubleMatrix L_vbias_mean = L_vbias.columnMeans();

		NeuralNetworkGradient gradient = new NeuralNetworkGradient(L_W,L_vbias_mean,L_hbias_mean);
		this.triggerGradientEvents(gradient);
		
		return gradient;
	}





}
