package org.deeplearning4j.rbm;


import static org.deeplearning4j.util.MatrixUtil.log;
import static org.deeplearning4j.util.MatrixUtil.oneDiv;
import static org.deeplearning4j.util.MatrixUtil.oneMinus;
import static org.deeplearning4j.util.MatrixUtil.uniform;
import static org.jblas.MatrixFunctions.exp;

import org.apache.commons.math3.distribution.RealDistribution;
import org.apache.commons.math3.random.RandomGenerator;
import org.deeplearning4j.berkeley.Pair;
import org.deeplearning4j.nn.BaseNeuralNetwork;
import org.jblas.DoubleMatrix;


/**
 * Continuous Restricted Boltzmann Machine.
 * Scale input to between 0 and 1. 
 * 
 * Consider each input unit as a probability of taking the value 1
 * with a different form of sampling.
 * 
 * See: http://papers.nips.cc/paper/3048-greedy-layer-wise-training-of-deep-networks.pdf
 * 
 * 
 * @author Adam Gibson
 *
 */
public class CRBM extends RBM {

	/**
	 * 
	 */
	private static final long serialVersionUID = 598767790003731193L;


	

	//never instantiate without the builder
	private CRBM(){}

	private CRBM(DoubleMatrix input, int nVisible, int nHidden,
			DoubleMatrix W, DoubleMatrix hbias, DoubleMatrix vbias,
			RandomGenerator rng, double fanIn, RealDistribution dist) {
		super(input, nVisible, nHidden, W, hbias, vbias, rng, fanIn, dist);
		if(useAdaGrad) {
			this.wAdaGrad.setMasterStepSize(0.0001);
		}
	}

	
	
	
	@Override
	public Pair<DoubleMatrix, DoubleMatrix> sampleVisibleGivenHidden(DoubleMatrix h) {
		DoubleMatrix activationHidden = propDown(h);
		DoubleMatrix negativeEnergy = exp(activationHidden.neg());
		DoubleMatrix positiveEnergy = exp(activationHidden);


		DoubleMatrix v1Mean = oneDiv(oneMinus(negativeEnergy).sub(oneDiv(activationHidden)));
		/*
		 * Scaled sampling of a probability distribution indicating the examples from 0 to 1:
		 * log( 1 - gaussian noise * 1 - ep ) / ah
		 */
		DoubleMatrix v1Sample = log(
				oneMinus(
				uniform(rng,v1Mean.rows,v1Mean.columns)
				.mul(oneMinus(positiveEnergy)))
				).div(activationHidden);


		return new Pair<DoubleMatrix,DoubleMatrix>(v1Mean,v1Sample);



	}


	public static class Builder extends BaseNeuralNetwork.Builder<CRBM> {
		public Builder() {
			this.clazz = CRBM.class;
		}
	}



}
