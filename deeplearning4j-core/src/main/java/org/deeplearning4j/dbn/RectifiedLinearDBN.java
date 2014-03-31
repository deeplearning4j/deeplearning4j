package org.deeplearning4j.dbn;

import org.apache.commons.math3.distribution.RealDistribution;
import org.apache.commons.math3.random.RandomGenerator;
import org.deeplearning4j.nn.BaseMultiLayerNetwork;
import org.deeplearning4j.nn.HiddenLayer;
import org.deeplearning4j.nn.NeuralNetwork;
import org.deeplearning4j.nn.RectifiedLinearHiddenLayer;
import org.deeplearning4j.nn.activation.ActivationFunction;
import org.deeplearning4j.rbm.RBM;
import org.deeplearning4j.rbm.RectifiedLinearRBM;
import org.jblas.DoubleMatrix;

public class RectifiedLinearDBN extends DBN {

	/**
	 * 
	 */
	private static final long serialVersionUID = 8727630396836587100L;

	public RectifiedLinearDBN() {}

	public RectifiedLinearDBN(int nIn, int[] hiddenLayerSizes, int nOuts, int nLayers,
			RandomGenerator rng, DoubleMatrix input,DoubleMatrix labels) {
		super(nIn, hiddenLayerSizes, nOuts, nLayers, rng, input,labels);
	}

	public RectifiedLinearDBN(int nIns, int[] hiddenLayerSizes, int nOuts, int nLayers,
			RandomGenerator rng) {
		super(nIns, hiddenLayerSizes, nOuts, nLayers, rng);
	}


	@Override
	public NeuralNetwork createLayer(DoubleMatrix input, int nVisible,
			int nHidden, DoubleMatrix W, DoubleMatrix hBias,
			DoubleMatrix vBias, RandomGenerator rng,int index) {
		RectifiedLinearRBM ret = new RectifiedLinearRBM.Builder()
		.useRegularization(isUseRegularization())
		.useAdaGrad(isUseAdaGrad()).normalizeByInputRows(isNormalizeByInputRows())
		.withMomentum(getMomentum()).withSparsity(getSparsity()).withDistribution(getDist())
		.numberOfVisible(nVisible).numHidden(nHidden).withWeights(W)
		.withInput(input).withVisibleBias(vBias).withHBias(hBias).withDistribution(getDist())
		.withRandom(rng).renderWeights(getRenderWeightsEveryNEpochs())
		.fanIn(getFanIn()).build();
		if(gradientListeners != null && !gradientListeners.isEmpty())
			ret.setGradientListeners(gradientListeners.get(index));
		return ret;

	}



	/**
	 * Creates a hidden layer with the given parameters.
	 * The default implementation is a binomial sampling 
	 * hidden layer, but this can be overriden 
	 * for other kinds of hidden units
	 * @param nIn the number of inputs
	 * @param nOut the number of outputs
	 * @param activation the activation function for the layer
	 * @param rng the rng to use for sampling
	 * @param layerInput the layer starting input
	 * @param dist the probability distribution to use
	 * for generating weights
	 * @return a hidden layer with the given paremters
	 */
	public  HiddenLayer createHiddenLayer(int index,int nIn,int nOut,ActivationFunction activation,RandomGenerator rng,DoubleMatrix layerInput,RealDistribution dist) {
		return new HiddenLayer.Builder()
		.nIn(nIn).nOut(nOut).withActivation(activation)
		.withRng(rng).withInput(layerInput).dist(dist)
		.build();


	}

	@Override
	public NeuralNetwork[] createNetworkLayers(int numLayers) {
		return new RBM[numLayers];
	}


	public static class Builder extends BaseMultiLayerNetwork.Builder<RectifiedLinearDBN> {
		public Builder() {
			this.clazz = RectifiedLinearDBN.class;
		}
	}

}
