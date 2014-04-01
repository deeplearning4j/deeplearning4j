package org.deeplearning4j.rbm;

import static org.jblas.MatrixFunctions.sqrt;

import static org.deeplearning4j.util.MatrixUtil.sigmoid;



import org.apache.commons.math3.distribution.RealDistribution;
import org.apache.commons.math3.random.RandomGenerator;
import org.deeplearning4j.berkeley.Pair;
import org.deeplearning4j.nn.BaseNeuralNetwork;
import org.deeplearning4j.util.MatrixUtil;
import org.jblas.DoubleMatrix;
/**
 * Visible units with 
 * gaussian noise and hidden binary activations.
 * Meant for continuous data
 * 
 * 
 * @author Adam Gibson
 *
 */
public class GaussianRectifiedLinearRBM extends RBM {

	/**
	 * 
	 */
	private static final long serialVersionUID = 5186639601076269003L;
	private DoubleMatrix sigma;


	//never instantiate without the builder
	private GaussianRectifiedLinearRBM(){}

	private GaussianRectifiedLinearRBM(DoubleMatrix input, int nVisible, int nHidden,
			DoubleMatrix W, DoubleMatrix hbias, DoubleMatrix vbias,
			RandomGenerator rng, double fanIn, RealDistribution dist) {
		super(input, nVisible, nHidden, W, hbias, vbias, rng, fanIn, dist);
		if(useAdaGrad) {
			this.wAdaGrad.setMasterStepSize(1e-2);
			this.wAdaGrad.setDecayLr(true);
		}

		sigma = DoubleMatrix.ones(nVisible);
	}



	/**
	 * Trains till global minimum is found.
	 * @param learningRate
	 * @param k
	 * @param input
	 */
	@Override
	public void trainTillConvergence(double learningRate,int k,DoubleMatrix input) {
		if(input != null)
			this.input = input;
		optimizer = new RBMOptimizer(this, learningRate, new Object[]{k,learningRate});
		optimizer.setTolerance(1e-6);
		optimizer.train(input);
	}


	/**
	 * Calculates the activation of the visible :
	 * sigmoid(v * W + hbias)
	 * 
	 *  Compute the mean activation of the hiddens given visible unit
        configurations for a set of training examples.

        Parameters
	 * @param v the visible layer
	 * @return the approximated activations of the visible layer
	 */
	public DoubleMatrix propUp(DoubleMatrix v) {
		//rectified linear
		DoubleMatrix preSig = v.divRowVector(sigma).mmul(W).addiRowVector(hBias);
		return preSig;

	}


	/**
	 * Rectified linear units for output
	 * @param v the visible values
	 * @return a binomial distribution containing the expected values and the samples
	 */
	public Pair<DoubleMatrix,DoubleMatrix> sampleHiddenGivenVisible(DoubleMatrix v) {
		DoubleMatrix h1Mean = propUp(v);
		DoubleMatrix sigH1Mean = sigmoid(h1Mean);
		/*
		 * Rectified linear part
		 */
		DoubleMatrix h1Sample = h1Mean.addi(MatrixUtil.normal(getRng(), h1Mean,1).mul(sqrt(sigH1Mean)));
		MatrixUtil.max(0.0, h1Sample);
		

	
		return new Pair<DoubleMatrix,DoubleMatrix>(h1Mean,h1Sample);

	}

	/**
	 * Calculates the activation of the hidden:
	 * h * W + vbias
	 *  Compute the mean activation of the visibles given hidden unit
        configurations for a set of training examples.

        Parameters
	 * @param h the hidden layer
	 * @return the approximated output of the hidden layer
	 */
	public DoubleMatrix propDown(DoubleMatrix h) {
		DoubleMatrix vMean = h.mmul(W.transpose()).mulRowVector(vBias.add(sigma));
		return vMean;

	}
	@Override
	public Pair<DoubleMatrix, DoubleMatrix> sampleVisibleGivenHidden(DoubleMatrix h) {
		DoubleMatrix v1Mean = propDown(h);
		DoubleMatrix v1Sample = MatrixUtil.normal(getRng(), v1Mean, 1).mulRowVector(sigma);
		return new Pair<>(v1Mean,v1Sample);



	}

	public static class Builder extends BaseNeuralNetwork.Builder<GaussianRectifiedLinearRBM> {
		public Builder() {
			this.clazz = GaussianRectifiedLinearRBM.class;
		}
	}


}
