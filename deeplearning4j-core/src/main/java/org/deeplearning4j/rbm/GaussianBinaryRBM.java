package org.deeplearning4j.rbm;

import org.apache.commons.math3.distribution.RealDistribution;
import org.apache.commons.math3.random.RandomGenerator;
import org.deeplearning4j.berkeley.Pair;
import org.deeplearning4j.nn.BaseNeuralNetwork;
import org.deeplearning4j.util.MatrixUtil;
import org.jblas.DoubleMatrix;
import org.jblas.MatrixFunctions;

public class GaussianBinaryRBM extends RBM {

	/**
	 * 
	 */
	private static final long serialVersionUID = 5186639601076269003L;

	

	//never instantiate without the builder
	private GaussianBinaryRBM(){}

	private GaussianBinaryRBM(DoubleMatrix input, int nVisible, int nHidden,
			DoubleMatrix W, DoubleMatrix hbias, DoubleMatrix vbias,
			RandomGenerator rng, double fanIn, RealDistribution dist) {
		super(input, nVisible, nHidden, W, hbias, vbias, rng, fanIn, dist);
		if(useAdaGrad) {
			this.wAdaGrad.setMasterStepSize(1e-2);
			this.wAdaGrad.setDecayLr(true);
		}
		
	}

	
	
	
	@Override
	public Pair<DoubleMatrix, DoubleMatrix> sampleVisibleGivenHidden(DoubleMatrix h) {
	
		
		
		DoubleMatrix v1Mean = propDown(h);
		double diffFromData = MatrixFunctions.pow(input.sub(v1Mean),2).mean() * 1e-4;
		/**
		 * Dynamically set the variance = to the squared 
		 * differences from the mean relative to the data.
		 * 
		 */
		DoubleMatrix gaussianNoise = MatrixUtil.normal(getRng(), v1Mean,diffFromData).mul(diffFromData);
		
		DoubleMatrix v1Sample = v1Mean.add(gaussianNoise);

		return new Pair<DoubleMatrix,DoubleMatrix>(v1Mean,v1Sample);



	}


	public static class Builder extends BaseNeuralNetwork.Builder<GaussianBinaryRBM> {
		public Builder() {
			this.clazz = GaussianBinaryRBM.class;
		}
	}


}
