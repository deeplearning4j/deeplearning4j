package com.ccc.sendalyzeit.textanalytics.algorithms.deeplearning.sda.matrix.jblas;

import org.apache.commons.math3.random.RandomGenerator;
import org.jblas.DoubleMatrix;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.ccc.sendalyzeit.textanalytics.algorithms.deeplearning.nn.matrix.jblas.BaseMultiLayerNetwork;
import com.ccc.sendalyzeit.textanalytics.algorithms.deeplearning.nn.matrix.jblas.HiddenLayerMatrix;
import com.ccc.sendalyzeit.textanalytics.algorithms.deeplearning.nn.matrix.jblas.NeuralNetwork;


/**
 * A JBlas implementation of 
 * stacked denoising auto encoders.
 * @author Adam Gibson
 *
 */
public class SdAMatrix extends BaseMultiLayerNetwork  {





	private static final long serialVersionUID = 1448581794985193009L;
	private static Logger log = LoggerFactory.getLogger(BaseMultiLayerNetwork.class);

	
	public SdAMatrix() {}
	
	public SdAMatrix(int n_ins, int[] hidden_layer_sizes, int n_outs,
			int n_layers, RandomGenerator rng, DoubleMatrix input,DoubleMatrix labels) {
		super(n_ins, hidden_layer_sizes, n_outs, n_layers, rng, input,labels);

	}


	public SdAMatrix(int nIns, int[] hiddenLayerSizes, int nOuts,
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
	 * @param corruption_level the corruption level (the smaller number of inputs; the higher the 
	 * corruption level should be) the percent of inputs to corrupt
	 * @param epochs the number of iterations to run
	 */
	public void pretrain( DoubleMatrix input,double lr,  double corruption_level,  int epochs) {
		if(this.input == null)
			initializeLayers(input);

		DoubleMatrix layerInput = null;

		for(int i = 0; i < nLayers; i++) {  // layer-wise                        
			//input layer
			if(i == 0)
				layerInput = input;
			else
				layerInput = this.sigmoidLayers[i - 1].sampleHGivenV(layerInput);

			DenoisingAutoEncoderMatrix da = (DenoisingAutoEncoderMatrix) this.layers[i];
			HiddenLayerMatrix h = this.sigmoidLayers[i];
			for(int epoch = 0; epoch < epochs; epoch++)  {
				da.train(layerInput, lr, corruption_level);
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
		return new DenoisingAutoEncoderMatrix(input, nVisible, nHidden, vBias, vBias, vBias, rng);
	}


	@Override
	public NeuralNetwork[] createNetworkLayers(int numLayers) {
		return new DenoisingAutoEncoderMatrix[numLayers];
	}


	public static class Builder extends BaseMultiLayerNetwork.Builder<SdAMatrix> {
		public Builder() {
			this.clazz = SdAMatrix.class;
		}
	}





}