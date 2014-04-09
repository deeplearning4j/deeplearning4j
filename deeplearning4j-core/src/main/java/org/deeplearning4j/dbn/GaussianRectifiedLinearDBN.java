package org.deeplearning4j.dbn;

import org.apache.commons.math3.distribution.RealDistribution;
import org.apache.commons.math3.random.RandomGenerator;
import org.deeplearning4j.nn.BaseMultiLayerNetwork;
import org.deeplearning4j.nn.HiddenLayer;
import org.deeplearning4j.nn.NeuralNetwork;
import org.deeplearning4j.nn.RectifiedLinearHiddenLayer;
import org.deeplearning4j.nn.activation.ActivationFunction;
import org.deeplearning4j.rbm.GaussianRectifiedLinearRBM;
import org.deeplearning4j.rbm.RBM;
import org.jblas.DoubleMatrix;
/**
 * Rectified Linear Units for activations and
 * gaussian inputs
 * @author Adam Gibson
 *
 */
public class GaussianRectifiedLinearDBN extends DBN {

	private static final long serialVersionUID = 3838174630098935941L;


	public GaussianRectifiedLinearDBN() {}

	public GaussianRectifiedLinearDBN(int nIn, int[] hiddenLayerSizes, int nOuts, int nLayers,
			RandomGenerator rng, DoubleMatrix input,DoubleMatrix labels) {
		super(nIn, hiddenLayerSizes, nOuts, nLayers, rng, input,labels);
	}

	public GaussianRectifiedLinearDBN(int nIns, int[] hiddenLayerSizes, int nOuts, int nLayers,
			RandomGenerator rng) {
		super(nIns, hiddenLayerSizes, nOuts, nLayers, rng);
		
	}



	@Override
	public NeuralNetwork createLayer(DoubleMatrix input, int nVisible,
			int nHidden, DoubleMatrix W, DoubleMatrix hBias,
			DoubleMatrix vBias, RandomGenerator rng,int index) {
		NeuralNetwork ret = new GaussianRectifiedLinearRBM.Builder()
                .useRegularization(isUseRegularization()).withLossFunction(getLossFunction())
				.withDistribution(getDist())
                .useAdaGrad(isUseAdaGrad())
                .normalizeByInputRows(normalizeByInputRows)
				.withHBias(hBias).numberOfVisible(nVisible)
                .numHidden(nHidden).withSparsity(getSparsity()).withOptmizationAlgo(getOptimizationAlgorithm())
				.withInput(input).withL2(getL2())
                .fanIn(getFanIn())
                .renderWeights(getRenderWeightsEveryNEpochs())
				.withRandom(rng).withWeights(W).build();

		if(gradientListeners.get(index) != null)
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
		return new RectifiedLinearHiddenLayer.Builder()
		.nIn(nIn).nOut(nOut).withActivation(activation)
		.withRng(rng).withInput(layerInput).dist(dist)
		.build();

	}
	
	@Override
	public NeuralNetwork[] createNetworkLayers(int numLayers) {
		return new RBM[numLayers];
	}

	public static class Builder extends BaseMultiLayerNetwork.Builder<GaussianRectifiedLinearDBN> {
		public Builder() {
			this.clazz = GaussianRectifiedLinearDBN.class;
		}
	}


}
