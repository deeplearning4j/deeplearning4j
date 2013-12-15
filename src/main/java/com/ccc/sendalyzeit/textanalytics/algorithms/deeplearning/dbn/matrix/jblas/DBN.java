package com.ccc.sendalyzeit.textanalytics.algorithms.deeplearning.dbn.matrix.jblas;

import org.apache.commons.math3.random.RandomGenerator;
import org.jblas.DoubleMatrix;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.ccc.sendalyzeit.textanalytics.algorithms.deeplearning.nn.matrix.jblas.BaseMultiLayerNetwork;
import com.ccc.sendalyzeit.textanalytics.algorithms.deeplearning.nn.matrix.jblas.HiddenLayerMatrix;
import com.ccc.sendalyzeit.textanalytics.algorithms.deeplearning.nn.matrix.jblas.NeuralNetwork;
import com.ccc.sendalyzeit.textanalytics.algorithms.deeplearning.rbm.matrix.jblas.RBM;
/**
 * Deep Belief Network
 * @author Adam Gibson
 *
 */
public class DBN extends BaseMultiLayerNetwork {

	private static final long serialVersionUID = -9068772752220902983L;
	private static Logger log = LoggerFactory.getLogger(DBN.class);

	
	
	public DBN(int n_ins, int[] hidden_layer_sizes, int n_outs, int n_layers,
			RandomGenerator rng, DoubleMatrix input,DoubleMatrix labels) {
		super(n_ins, hidden_layer_sizes, n_outs, n_layers, rng, input,labels);
	}


	
	public void pretrain(int k,double learningRate,int epochs) {
		DoubleMatrix layerInput = null;
		for(int i = 0; i < nLayers; i++) {
			if(i == 0)
				layerInput = this.input;
			else 
				layerInput = sigmoidLayers[i-1].sample_h_given_v(layerInput);
			RBM r = (RBM) this.layers[i];
			HiddenLayerMatrix h = this.sigmoidLayers[i];

			for(int  epoch = 0; epoch < epochs; epoch++) {
				r.contrastiveDivergence(learningRate, k, layerInput);
				h.W = r.W;
				h.b = r.hBias;
			}
				
		}
	}
	
	
	@Override
	public NeuralNetwork createLayer(DoubleMatrix input, int nVisible,
			int nHidden, DoubleMatrix W, DoubleMatrix hBias,
			DoubleMatrix vBias, RandomGenerator rng,int index) {
		return new RBM(input, nVisible, nHidden, W, hBias, vBias, rng);
	}

	@Override
	public NeuralNetwork[] createNetworkLayers(int numLayers) {
		return new RBM[numLayers];
	}
	

	public static class Builder extends BaseMultiLayerNetwork.Builder<DBN> {
		public Builder() {
			this.clazz = DBN.class;
		}
	}
	

}
