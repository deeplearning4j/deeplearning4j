package org.deeplearning4j.nn;

import java.io.Serializable;

import org.apache.commons.math3.distribution.NormalDistribution;
import org.apache.commons.math3.distribution.RealDistribution;
import org.apache.commons.math3.random.MersenneTwister;
import org.apache.commons.math3.random.RandomGenerator;
import org.deeplearning4j.nn.activation.ActivationFunction;
import org.deeplearning4j.nn.activation.Sigmoid;
import org.deeplearning4j.util.MatrixUtil;
import org.jblas.DoubleMatrix;


/**
 * Vectorized Hidden Layer
 * @author Adam Gibson
 *
 */
public class HiddenLayer implements Serializable {

	private static final long serialVersionUID = 915783367350830495L;
	private int nIn;
	private int nOut;
	private DoubleMatrix W;
	private DoubleMatrix b;
	private RandomGenerator rng;
	private DoubleMatrix input;
	private ActivationFunction activationFunction = new Sigmoid();
	private RealDistribution dist;

	private HiddenLayer() {}

	public HiddenLayer(int nIn, int nOut, DoubleMatrix W, DoubleMatrix b, RandomGenerator rng,DoubleMatrix input,ActivationFunction activationFunction) {
		this(nIn,nOut,W,b,rng,input,activationFunction,null);
	}


	public HiddenLayer(int nIn, int nOut, DoubleMatrix W, DoubleMatrix b, RandomGenerator rng,DoubleMatrix input) {
		this(nIn,nOut,W,b,rng,input,null,null);
	}




	public HiddenLayer(int nIn, int nOut, DoubleMatrix W, DoubleMatrix b, RandomGenerator rng,DoubleMatrix input,ActivationFunction activationFunction,RealDistribution dist) {
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


	public HiddenLayer(int nIn, int nOut, DoubleMatrix W, DoubleMatrix b, RandomGenerator rng,DoubleMatrix input,RealDistribution dist) {
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
	public synchronized int getnIn() {
		return nIn;
	}

	public synchronized void setnIn(int nIn) {
		this.nIn = nIn;
	}

	public synchronized int getnOut() {
		return nOut;
	}

	public synchronized void setnOut(int nOut) {
		this.nOut = nOut;
	}

	public synchronized DoubleMatrix getW() {
		return W;
	}

	public synchronized void setW(DoubleMatrix w) {
		W = w;
	}

	public synchronized DoubleMatrix getB() {
		return b;
	}

	public synchronized void setB(DoubleMatrix b) {
		this.b = b;
	}

	public synchronized RandomGenerator getRng() {
		return rng;
	}

	public synchronized void setRng(RandomGenerator rng) {
		this.rng = rng;
	}

	public synchronized DoubleMatrix getInput() {
		return input;
	}

	public synchronized void setInput(DoubleMatrix input) {
		this.input = input;
	}

	public synchronized ActivationFunction getActivationFunction() {
		return activationFunction;
	}

	public synchronized void setActivationFunction(
			ActivationFunction activationFunction) {
		this.activationFunction = activationFunction;
	}

	@Override
	public HiddenLayer clone() {
		HiddenLayer layer = new HiddenLayer();
		layer.b = b.dup();
		layer.W = W.dup();
		if(input != null)
			layer.input = input.dup();
		if(dist != null)
			layer.dist = dist;
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
		if(input != null)
			layer.input = input.transpose();
		if(dist != null)
			layer.dist = dist;
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
	public synchronized DoubleMatrix activate() {
		return getActivationFunction().apply(getInput().mmul(getW()).addRowVector(getB()));
	}

	/**
	 * Initialize the layer with the given input
	 * and return the activation for this layer
	 * given this input
	 * @param input the input to use
	 * @return
	 */
	public synchronized DoubleMatrix activate(DoubleMatrix input) {
		if(input != null)
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
		private int nIn;
		private int nOut;
		private DoubleMatrix W;
		private DoubleMatrix b;
		private RandomGenerator rng;
		private DoubleMatrix input;
		private ActivationFunction activationFunction = new Sigmoid();
		private RealDistribution dist;
		
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

		public HiddenLayer build() {
			HiddenLayer ret =  new HiddenLayer(nIn,nOut,W,b,rng,input); 
			ret.activationFunction = activationFunction;
			ret.dist = dist;
			return ret;
		}

	}

}