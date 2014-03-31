package org.deeplearning4j.rbm;

import static org.deeplearning4j.util.MatrixUtil.*;

import org.apache.commons.math3.distribution.RealDistribution;
import org.apache.commons.math3.random.RandomGenerator;
import org.deeplearning4j.berkeley.Pair;
import org.deeplearning4j.nn.BaseNeuralNetwork;
import org.deeplearning4j.util.MatrixUtil;
import org.jblas.DoubleMatrix;
import org.jblas.MatrixFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
/**
 * RBM with rectified linear hidden units and linear units with gaussian noise.
 * This is meant for use with continuous data, note that training needs to be slower
 * in order to accomadate for the variance in data vs the normal binary-binary rbm.
 * http://machinelearning.wustl.edu/mlpapers/paper_files/icml2010_NairH10.pdf
 * 
 * @author Adam Gibson
 *
 */
public class RectifiedLinearRBM extends RBM {

	private static Logger log = LoggerFactory.getLogger(RectifiedLinearRBM.class);
	/**
	 * 
	 */
	private static final long serialVersionUID = -8368874372096273122L;


	//never instantiate without the builder
	private RectifiedLinearRBM(){}

	private RectifiedLinearRBM(DoubleMatrix input, int nVisible, int nHidden,
			DoubleMatrix W, DoubleMatrix hbias, DoubleMatrix vbias,
			RandomGenerator rng, double fanIn, RealDistribution dist) {
		super(input, nVisible, nHidden, W, hbias, vbias, rng, fanIn, dist);
	}

	/**
	 * Activation of visible units:
	 * Linear units with gaussian noise:
	 * max(0,x + N(0,sigmoid(x)))
	 * @param v the visible layer
	 * @return the approximated activations of the visible layer
	 */
	public DoubleMatrix propUp(DoubleMatrix v) {
		DoubleMatrix preSig = sigmoid(v.mmul(W).addiRowVector(hBias));
		double variance = MatrixUtil.variance(preSig);

		DoubleMatrix gaussian = MatrixUtil.normal(getRng(), preSig, variance).mul(variance);
		preSig.addi(gaussian);
		return preSig;

	}
	
	/**
	 * Calculates the activation of the hidden:
	 * using rectified linear units
	 * sigmoid(h * W + vbias)
	 * @param h the hidden layer
	 * @return the approximated output of the hidden layer
	 */
	public DoubleMatrix propDown(DoubleMatrix h) {
		DoubleMatrix preSig = sigmoid(h.mmul(W.transpose()).addRowVector(vBias));
		double variance = MatrixUtil.variance(preSig);

		DoubleMatrix gaussian = MatrixUtil.normal(getRng(), preSig, variance).mul(variance);
		preSig.addi(gaussian);
		for(int i = 0;i < preSig.length; i++)
			preSig.put(i,Math.max(0,preSig.get(i)));
		
		
		return preSig;

	}

	

	@Override
	public Pair<DoubleMatrix, DoubleMatrix> sampleVisibleGivenHidden(DoubleMatrix h) {



		DoubleMatrix v1Mean = propDown(h);
		double variance = MatrixFunctions.pow(input.sub(v1Mean),2).mean();
		
		/**
		 * Dynamically set the variance = to the squared 
		 * differences from the mean relative to the data.
		 * 
		 */
		DoubleMatrix gaussianNoise = normal(getRng(), v1Mean,variance).mul(variance);

		DoubleMatrix v1Sample = v1Mean.add(gaussianNoise);

		return new Pair<>(v1Mean,v1Sample);



	}





	/**
	 * Rectified linear hidden units
	 * @param v the visible values
	 * @return a the hidden samples as rectified linear units
	 */
	public Pair<DoubleMatrix,DoubleMatrix> sampleHiddenGivenVisible(DoubleMatrix v) {
		DoubleMatrix h1Mean = propUp(v);
		//variance wrt reconstruction
		double variance =  MatrixFunctions.pow(v.sub(v.mean()),2).mean();
	
		/**
		 * Dynamically set the variance = to the squared 
		 * differences from the mean relative to the data.
		 * 
		 */


		DoubleMatrix gaussianNoise = normal(getRng(), sigmoid(h1Mean),variance).mul(variance);
		//max(zero,x + noise)
		DoubleMatrix h1Sample = h1Mean.add(gaussianNoise);
		for(int i = 0;i < h1Sample.length; i++)
			h1Sample.put(i,Math.max(0,h1Sample.get(i)));

		return new Pair<>(h1Mean,h1Sample);

	}


	public static class Builder extends BaseNeuralNetwork.Builder<RectifiedLinearRBM> {
		public Builder() {
			this.clazz = RectifiedLinearRBM.class;
		}
	}


}
