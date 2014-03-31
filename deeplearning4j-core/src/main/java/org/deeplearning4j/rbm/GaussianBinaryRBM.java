package org.deeplearning4j.rbm;

import static org.deeplearning4j.util.MatrixUtil.sigmoid;

import org.apache.commons.math3.distribution.RealDistribution;
import org.apache.commons.math3.random.RandomGenerator;
import org.deeplearning4j.berkeley.Pair;
import org.deeplearning4j.nn.BaseNeuralNetwork;
import org.deeplearning4j.util.MatrixUtil;
import org.jblas.DoubleMatrix;
/**
 * Visible units with gaussian noise and hidden binary activations
 * @author Adam Gibson
 *
 */
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

		for(int i = 0;i < this.getvBias().length; i++) {
			this.getvBias().put(i,this.getvBias().get(i) - i);
		}

	}

	/**
	 * Activation of visible units:
	 * Linear units with gaussian noise:
	 * max(0,x + N(0,sigmoid(x)))
	 * @param v the visible layer
	 * @return the approximated activations of the visible layer
	 */
	public DoubleMatrix propUp(DoubleMatrix v) {
		DoubleMatrix activation = sigmoid(v.mmul(W).addiRowVector(hBias));
		DoubleMatrix gaussian = MatrixUtil.normal(getRng(), activation);
		activation.addi(gaussian);
		for(int i = 0; i < activation.length; i++)
			activation.put(i,Math.max(0,activation.get(i)));
		
		return activation;

	}


	@Override
	public Pair<DoubleMatrix, DoubleMatrix> sampleVisibleGivenHidden(DoubleMatrix h) {
		
		DoubleMatrix v1Mean = propDown(h);
		//prevent zeros
		v1Mean.addi(1e-4);
		
		/**
		 * Use the sigmoid bounds as the standard deviation
		 * with a mean of 0. 
		 */
		DoubleMatrix gaussianNoise = MatrixUtil.normal(getRng(), v1Mean);

		DoubleMatrix v1Sample = v1Mean.add(gaussianNoise);

		return new Pair<>(v1Mean,v1Sample);



	}


	public static class Builder extends BaseNeuralNetwork.Builder<GaussianBinaryRBM> {
		public Builder() {
			this.clazz = GaussianBinaryRBM.class;
		}
	}


}
