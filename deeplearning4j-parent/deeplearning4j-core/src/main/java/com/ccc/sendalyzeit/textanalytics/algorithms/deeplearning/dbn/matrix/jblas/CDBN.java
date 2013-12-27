package com.ccc.sendalyzeit.textanalytics.algorithms.deeplearning.dbn.matrix.jblas;

import org.apache.commons.math3.random.RandomGenerator;
import org.jblas.DoubleMatrix;

import com.ccc.sendalyzeit.textanalytics.algorithms.deeplearning.nn.matrix.jblas.BaseMultiLayerNetwork;
import com.ccc.sendalyzeit.textanalytics.algorithms.deeplearning.nn.matrix.jblas.NeuralNetwork;
import com.ccc.sendalyzeit.textanalytics.algorithms.deeplearning.rbm.matrix.jblas.CRBM;
import com.ccc.sendalyzeit.textanalytics.algorithms.deeplearning.rbm.matrix.jblas.RBM;
/**
 * Continuous Deep Belief Network.
 * 
 * Uses a continuous RBM in the first layer
 * @author Adam Gibson
 *
 */
public class CDBN extends DBN {

	private static final long serialVersionUID = 3838174630098935941L;

	
	public CDBN() {}
	
	public CDBN(int n_ins, int[] hidden_layer_sizes, int n_outs, int n_layers,
			RandomGenerator rng, DoubleMatrix input,DoubleMatrix labels) {
		super(n_ins, hidden_layer_sizes, n_outs, n_layers, rng, input,labels);
	}

	public CDBN(int n_ins, int[] hidden_layer_sizes, int n_outs, int n_layers,
			RandomGenerator rng) {
		super(n_ins, hidden_layer_sizes, n_outs, n_layers, rng);
	}

	@Override
	public NeuralNetwork createLayer(DoubleMatrix input, int nVisible,
			int nHidden, DoubleMatrix W, DoubleMatrix hBias,
			DoubleMatrix vBias, RandomGenerator rng,int index) {
		if(index == 0)
			return new CRBM(input, nVisible, nHidden, W, hBias, vBias, rng);
		else
			return new RBM(input,nVisible,nHidden,W,hBias,vBias,rng);
	}

	@Override
	public NeuralNetwork[] createNetworkLayers(int numLayers) {
		return new RBM[numLayers];
	}
	
	public static class Builder extends BaseMultiLayerNetwork.Builder<CDBN> {
		public Builder() {
			this.clazz = CDBN.class;
		}
	}
	

}
