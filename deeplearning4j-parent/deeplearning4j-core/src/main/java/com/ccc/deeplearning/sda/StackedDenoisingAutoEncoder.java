package com.ccc.deeplearning.sda;

import org.apache.commons.math3.random.RandomGenerator;
import org.jblas.DoubleMatrix;

import com.ccc.deeplearning.da.DenoisingAutoEncoder;
import com.ccc.deeplearning.nn.BaseMultiLayerNetwork;
import com.ccc.deeplearning.nn.HiddenLayer;
import com.ccc.deeplearning.nn.NeuralNetwork;


/**
 * A JBlas implementation of 
 * stacked denoising auto encoders.
 * @author Adam Gibson
 *
 */
public class StackedDenoisingAutoEncoder extends BaseMultiLayerNetwork  {

	private static final long serialVersionUID = 1448581794985193009L;


	public StackedDenoisingAutoEncoder() {}

	public StackedDenoisingAutoEncoder(int n_ins, int[] hiddenLayerSizes, int nOuts,
			int nLayers, RandomGenerator rng, DoubleMatrix input,DoubleMatrix labels) {
		super(n_ins, hiddenLayerSizes, nOuts, nLayers, rng, input,labels);

	}


	public StackedDenoisingAutoEncoder(int nIns, int[] hiddenLayerSizes, int nOuts,
			int n_layers, RandomGenerator rng) {
		super(nIns, hiddenLayerSizes, nOuts, n_layers, rng);
	}


	public void pretrain( double lr,  double corruptionLevel,  int epochs) {
		pretrain(this.input,lr,corruptionLevel,epochs);
	}

	/**
	 * Unsupervised pretraining based on reconstructing the input
	 * from a corrupted version
	 * @param input the input to train on
	 * @param lr the starting learning rate
	 * @param corruptionLevel the corruption level (the smaller number of inputs; the higher the 
	 * corruption level should be) the percent of inputs to corrupt
	 * @param epochs the number of iterations to run
	 */
	public void pretrain(DoubleMatrix input,double lr,  double corruptionLevel,  int epochs) {
		if(this.input == null)
			initializeLayers(input);

		DoubleMatrix layerInput = null;

		for(int i = 0; i < nLayers; i++) {  // layer-wise                        
			//input layer
			if(i == 0)
				layerInput = input;
			else
				layerInput = this.sigmoidLayers[i - 1].sampleHGivenV(layerInput);
			DenoisingAutoEncoder da = (DenoisingAutoEncoder) this.layers[i];
			HiddenLayer h = this.sigmoidLayers[i];
			for(int l = 0; l < epochs; l++) {
				da.trainTillConverge(layerInput, lr, corruptionLevel);
				if(h.W != (da.W))
					h.W = da.W;
				if(h.b != da.hBias)
					h.b = da.hBias;
			}
			
		}	


	}

	/**
	 * 
	 * @param input input examples
	 * @param labels output labels
	 * @param otherParams
	 * 
	 * (double) learningRate
	 * (double) corruptionLevel
	 * (int) epochs
	 * 
	 * Optional:
	 * (double) finetune lr
	 * (int) finetune epochs
	 * 
	 */
	@Override
	public void trainNetwork(DoubleMatrix input, DoubleMatrix labels,
			Object[] otherParams) {
		Double lr = (Double) otherParams[0];
		Double corruptionLevel = (Double) otherParams[1];
		Integer epochs = (Integer) otherParams[2];

		pretrain(input, lr, corruptionLevel, epochs);
		if(otherParams.length <= 3)
			finetune(labels, lr, epochs);
		else {
			Double finetuneLr = (Double) otherParams[3];
			Integer fineTuneEpochs = (Integer) otherParams[4];
			finetune(labels,finetuneLr,fineTuneEpochs);
		}
	}



	@Override
	public NeuralNetwork createLayer(DoubleMatrix input, int nVisible,
			int nHidden, DoubleMatrix W, DoubleMatrix hbias,
			DoubleMatrix vBias, RandomGenerator rng,int index) {
		return new DenoisingAutoEncoder(input, nVisible, nHidden, vBias, vBias, vBias, rng);
	}


	@Override
	public NeuralNetwork[] createNetworkLayers(int numLayers) {
		return new DenoisingAutoEncoder[numLayers];
	}


	public static class Builder extends BaseMultiLayerNetwork.Builder<StackedDenoisingAutoEncoder> {
		public Builder() {
			this.clazz = StackedDenoisingAutoEncoder.class;
		}
	}





}