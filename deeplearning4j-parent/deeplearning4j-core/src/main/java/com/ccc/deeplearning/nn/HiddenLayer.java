package com.ccc.deeplearning.nn;

import java.io.Serializable;

import org.apache.commons.math3.distribution.NormalDistribution;
import org.apache.commons.math3.distribution.UniformRealDistribution;
import org.apache.commons.math3.random.MersenneTwister;
import org.apache.commons.math3.random.RandomGenerator;
import org.jblas.DoubleMatrix;

import com.ccc.deeplearning.nn.activation.ActivationFunction;
import com.ccc.deeplearning.nn.activation.Sigmoid;
import com.ccc.deeplearning.util.MatrixUtil;

/**
 * Vectorized Hidden Layer
 * @author Adam Gibson
 *
 */
public class HiddenLayer implements Serializable {

	private static final long serialVersionUID = 915783367350830495L;
	public int nIn;
	public int nOut;
	public DoubleMatrix W;
	public DoubleMatrix b;
	public RandomGenerator rng;
	public DoubleMatrix input;
	public ActivationFunction activationFunction = new Sigmoid();


	private HiddenLayer() {}

	public HiddenLayer(int nIn, int nOut, DoubleMatrix W, DoubleMatrix b, RandomGenerator rng,DoubleMatrix input,ActivationFunction activationFunction) {
		this.nIn = nIn;
		this.nOut = nOut;
		this.input = input;
		this.activationFunction = activationFunction;
		
		if(rng == null) {
			this.rng = new MersenneTwister(1234);
		}
		else 
			this.rng = rng;

		if(W == null) {

			NormalDistribution u = new NormalDistribution(rng,0,.01,NormalDistribution.DEFAULT_INVERSE_ABSOLUTE_ACCURACY);

			this.W = DoubleMatrix.zeros(nIn,nOut);

			for(int i = 0; i < this.W.rows; i++) 
				this.W.putRow(i,new DoubleMatrix(u.sample(this.W.columns)));
		}

		else 
			this.W = W;


		if(b == null) 
			this.b = DoubleMatrix.zeros(nOut);
		else 
			this.b = b;
	}


	public HiddenLayer(int nIn, int nOut, DoubleMatrix W, DoubleMatrix b, RandomGenerator rng,DoubleMatrix input) {
		this.nIn = nIn;
		this.nOut = nOut;
		this.input = input;

		if(rng == null) {
			this.rng = new MersenneTwister(1234);
		}
		else 
			this.rng = rng;

		if(W == null) {
			NormalDistribution u = new NormalDistribution(rng,0,.01,NormalDistribution.DEFAULT_INVERSE_ABSOLUTE_ACCURACY);

			this.W = DoubleMatrix.zeros(nIn,nOut);

			for(int i = 0; i < this.W.rows; i++) 
				this.W.putRow(i,new DoubleMatrix(u.sample(this.W.columns)));
		}

		else 
			this.W = W;


		if(b == null) 
			this.b = DoubleMatrix.zeros(nOut);
		else 
			this.b = b;
	}
	
	@Override
	public HiddenLayer clone() {
		HiddenLayer layer = new HiddenLayer();
		layer.b = b.dup();
		layer.W = W.dup();
		layer.input = input.dup();
		layer.activationFunction = activationFunction;
		layer.nOut = nOut;
		layer.nIn = nIn;
		layer.rng = rng;
		return layer;
	}
	
	
	public HiddenLayer transpose() {
		HiddenLayer layer = new HiddenLayer();
		layer.b = b.dup();
		layer.W = W.transpose();
		layer.input = input.transpose();
		layer.activationFunction = activationFunction;
		layer.nOut = nIn;
		layer.nIn = nOut;
		layer.rng = rng;
		return layer;
	}

	/**
	 * Trigger an activation with the last specified input
	 * @return the activation of the last specified input
	 */
	public DoubleMatrix activate() {
		return activationFunction.apply(this.input.mmul(W).addRowVector(b));
	}

	/**
	 * Initialize the layer with the given input
	 * and return the activation for this layer
	 * given this input
	 * @param input the input to use
	 * @return
	 */
	public DoubleMatrix activate(DoubleMatrix input) {
		this.input = input;
		return activate();
	}

	/**
	 * Sample this hidden layer given the input
	 * and initialize this layer with the given input
	 * @param input the input to sample
	 * @return the activation for this layer
	 * given the input
	 */
	public DoubleMatrix sampleHGivenV(DoubleMatrix input) {
		this.input = input;
		DoubleMatrix ret = MatrixUtil.binomial(activate(), 1, rng);
		return ret;
	}

	/**
	 * Sample this hidden layer given the last input.
	 * @return the activation for this layer given 
	 * the previous input
	 */
	public DoubleMatrix sample_h_given_v() {
		DoubleMatrix output = activate();
		//reset the seed to ensure consistent generation of data
		DoubleMatrix ret = MatrixUtil.binomial(output, 1, rng);
		return ret;
	}
	
	
	
	public static class Builder {
		private int n_in;
		private int n_out;
		private DoubleMatrix W;
		private DoubleMatrix b;
		private RandomGenerator rng;
		private DoubleMatrix input;
		private ActivationFunction activationFunction = new Sigmoid();
		
		
		public Builder nIn(int nIn) {
			this.n_in = nIn;
			return this;
		}
		
		public Builder nOut(int nOut) {
			this.n_out = nOut;
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
		
		public HiddenLayer build() {
			HiddenLayer ret =  new HiddenLayer(n_in,n_out,W,b,rng,input); 
			ret.activationFunction = activationFunction;
			return ret;
		}
		
	}
	
}