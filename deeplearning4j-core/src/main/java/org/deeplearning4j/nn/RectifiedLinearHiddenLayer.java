package org.deeplearning4j.nn;

import static org.jblas.MatrixFunctions.exp;
import static org.jblas.MatrixFunctions.log;

import org.apache.commons.math3.distribution.NormalDistribution;
import org.apache.commons.math3.distribution.RealDistribution;
import org.apache.commons.math3.random.MersenneTwister;
import org.apache.commons.math3.random.RandomGenerator;
import org.deeplearning4j.nn.activation.ActivationFunction;
import org.deeplearning4j.nn.activation.Sigmoid;
import org.jblas.DoubleMatrix;
/**
 * Rectified linear hidden units vs binomial sampled ones
 * @author Adam Gibson
 *
 */
public class RectifiedLinearHiddenLayer extends HiddenLayer {

	/**
	 * 
	 */
	private static final long serialVersionUID = 2266162281744170946L;


	public RectifiedLinearHiddenLayer() {}

	public RectifiedLinearHiddenLayer(int nIn, int nOut, DoubleMatrix W, DoubleMatrix b, RandomGenerator rng,DoubleMatrix input,ActivationFunction activationFunction) {
		this(nIn,nOut,W,b,rng,input,activationFunction,null);
	}


	public RectifiedLinearHiddenLayer(int nIn, int nOut, DoubleMatrix W, DoubleMatrix b, RandomGenerator rng,DoubleMatrix input) {
		this(nIn,nOut,W,b,rng,input,null,null);
	}




	public RectifiedLinearHiddenLayer(int nIn, int nOut, DoubleMatrix W, DoubleMatrix b, RandomGenerator rng,DoubleMatrix input,ActivationFunction activationFunction,RealDistribution dist) {
		this.nIn = nIn;
		this.nOut = nOut;
		this.input = input;
		if(activationFunction != null)
			this.activationFunction = activationFunction;

		if(rng == null) {
			this.rng = new MersenneTwister(1234);
		}
		else 
			this.rng = rng;

		if(dist == null)
			this.dist = new NormalDistribution(this.rng,0,.01,NormalDistribution.DEFAULT_INVERSE_ABSOLUTE_ACCURACY);
		else
			this.dist = dist;

		if(W == null) {

			this.W = DoubleMatrix.zeros(nIn,nOut);

			for(int i = 0; i < this.W.rows; i++) 
				this.W.putRow(i,new DoubleMatrix(this.dist.sample(this.W.columns)));
		}

		else 
			this.W = W;


		if(b == null) 
			this.b = DoubleMatrix.zeros(nOut);
		else 
			this.b = b;
	}


	public RectifiedLinearHiddenLayer(int nIn, int nOut, DoubleMatrix W, DoubleMatrix b, RandomGenerator rng,DoubleMatrix input,RealDistribution dist) {
		this.nIn = nIn;
		this.nOut = nOut;
		this.input = input;


		if(rng == null) 
			this.rng = new MersenneTwister(1234);

		else 
			this.rng = rng;

		if(dist == null)
			this.dist = new NormalDistribution(this.rng,0,.01,NormalDistribution.DEFAULT_INVERSE_ABSOLUTE_ACCURACY);
		else
			this.dist = dist;

		if(W == null) {

			this.W = DoubleMatrix.zeros(nIn,nOut);

			for(int i = 0; i < this.W.rows; i++) 
				this.W.putRow(i,new DoubleMatrix(this.dist.sample(this.W.columns)));
		}

		else 
			this.W = W;


		if(b == null) 
			this.b = DoubleMatrix.zeros(nOut);
		else 
			this.b = b;
	}


	public static class Builder {
		protected int nIn;
		protected int nOut;
		protected DoubleMatrix W;
		protected DoubleMatrix b;
		protected RandomGenerator rng;
		protected DoubleMatrix input;
		protected ActivationFunction activationFunction = new Sigmoid();
		protected RealDistribution dist;

		public Builder dist(RealDistribution dist) {
			this.dist = dist;
			return this;
		}

		public Builder nIn(int nIn) {
			this.nIn = nIn;
			return this;
		}

		public Builder nOut(int nOut) {
			this.nOut = nOut;
			return this;
		}

		public Builder withWeights(DoubleMatrix W) {
			this.W = W;
			return this;
		}

		public Builder withRng(RandomGenerator gen) {
			this.rng = gen;
			return this;
		}

		public Builder withActivation(ActivationFunction function) {
			this.activationFunction = function;
			return this;
		}

		public Builder withBias(DoubleMatrix b) {
			this.b = b;
			return this;
		}

		public Builder withInput(DoubleMatrix input) {
			this.input = input;
			return this;
		}

		public RectifiedLinearHiddenLayer build() {
			RectifiedLinearHiddenLayer ret =  new RectifiedLinearHiddenLayer(nIn,nOut,W,b,rng,input); 
			ret.activationFunction = activationFunction;
			ret.dist = dist;
			return ret;
		}

	}


	/**
	 * Sample this hidden layer given the input
	 * and initialize this layer with the given input
	 * @param input the input to sample
	 * @return the activation for this layer
	 * given the input
	 */
	@Override
	public DoubleMatrix sampleHGivenV(DoubleMatrix input) {
		this.input = input;
		DoubleMatrix output = activate();
		return output;
	}

	/**
	 * Sample this hidden layer given the last input.
	 * @return the activation for this layer given 
	 * the previous input
	 */
	@Override
	public DoubleMatrix sampleHiddenGivenVisible() {
		DoubleMatrix output = activate();
		return output;

	}
}
