package org.deeplearning4j.dbn;

import org.apache.commons.math3.random.RandomGenerator;
import org.deeplearning4j.nn.BaseMultiLayerNetwork;
import org.deeplearning4j.nn.NeuralNetwork;
import org.deeplearning4j.rbm.GaussianRectifiedLinearRBM;
import org.deeplearning4j.rbm.RBM;
import org.jblas.DoubleMatrix;

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
		NeuralNetwork ret = null;
		ret = new GaussianRectifiedLinearRBM.Builder().useRegularization(isUseRegularization())
				.withDistribution(getDist()).useAdaGrad(isUseAdaGrad()).normalizeByInputRows(normalizeByInputRows)
				.withHBias(hBias).numberOfVisible(nVisible).numHidden(nHidden).withSparsity(getSparsity())
				.withInput(input).withL2(getL2()).fanIn(getFanIn()).renderWeights(getRenderWeightsEveryNEpochs())
				.withRandom(rng).withWeights(W).build();

		if(gradientListeners.get(index) != null)
			ret.setGradientListeners(gradientListeners.get(index));
		return ret;
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
