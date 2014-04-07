package org.deeplearning4j.nn;

import static org.deeplearning4j.util.MatrixUtil.sigmoid;
import static org.jblas.MatrixFunctions.sqrt;

import org.apache.commons.math3.distribution.RealDistribution;
import org.apache.commons.math3.random.RandomGenerator;
import org.deeplearning4j.nn.activation.ActivationFunction;
import org.deeplearning4j.nn.activation.Sigmoid;
import org.deeplearning4j.util.MatrixUtil;
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
		super(nIn,nOut,W,b,rng,input,activationFunction,dist);
	}


	public RectifiedLinearHiddenLayer(int nIn, int nOut, DoubleMatrix W, DoubleMatrix b, RandomGenerator rng,DoubleMatrix input,RealDistribution dist) {
		super(nIn,nOut,W,b,rng,input,dist);
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


		DoubleMatrix sigH1Mean = sigmoid(output);
		/*
		 * Rectified linear part
		 */
		DoubleMatrix h1Sample = output.addi(MatrixUtil.normal(getRng(), output,1).mul(sqrt(sigH1Mean)));
		MatrixUtil.max(0.0, h1Sample);

		return h1Sample;
	}


    /**
     * Trigger an activation with the last specified input
     * @return the activation of the last specified input
     */
    public synchronized DoubleMatrix activate() {
        DoubleMatrix activation = input.mmul(getW());
        MatrixUtil.max(0.0,activation);
        activation.subiRowVector(activation.columnMeans());
        activation.diviRowVector(MatrixUtil.columnStd(activation));
        return activation;
    }


    /**
	 * Sample this hidden layer given the last input.
	 * @return the activation for this layer given 
	 * the previous input
	 */
	@Override
	public DoubleMatrix sampleHiddenGivenVisible() {
		return sampleHGivenV(input);

	}
}
