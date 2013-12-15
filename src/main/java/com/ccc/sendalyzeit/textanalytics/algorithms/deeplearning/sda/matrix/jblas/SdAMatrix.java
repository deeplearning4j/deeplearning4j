package com.ccc.sendalyzeit.textanalytics.algorithms.deeplearning.sda.matrix.jblas;

import java.io.InputStream;
import java.io.ObjectInputStream;
import java.io.Serializable;

import org.apache.commons.math3.random.RandomGenerator;
import org.jblas.DoubleMatrix;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.ccc.sendalyzeit.textanalytics.algorithms.deeplearning.nn.matrix.jblas.BaseMultiLayerNetwork;
import com.ccc.sendalyzeit.textanalytics.algorithms.deeplearning.nn.matrix.jblas.HiddenLayerMatrix;
import com.ccc.sendalyzeit.textanalytics.algorithms.deeplearning.nn.matrix.jblas.NeuralNetwork;


/**
 * A JBlas implementation of 
 * stacked deep auto encoders.
 * @author Adam Gibson
 *
 */
public class SdAMatrix extends BaseMultiLayerNetwork implements Serializable {





	private static final long serialVersionUID = 1448581794985193009L;
	private static Logger log = LoggerFactory.getLogger(SdAMatrix.class);

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

	public void pretrain( DoubleMatrix input,double lr,  double corruption_level,  int epochs) {
		if(this.input == null)
			initializeLayers(input);
		
		DoubleMatrix layerInput = null;

		for(int i = 0; i < nLayers; i++) {  // layer-wise                        
			//input layer
			if(i == 0)
				layerInput = input;
			else
				layerInput = this.sigmoidLayers[i - 1].sample_h_given_v(layerInput);

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


	public void load(InputStream is) {
		try {
			ObjectInputStream ois = new ObjectInputStream(is);
			SdAMatrix loaded = (SdAMatrix) ois.readObject();
			update(loaded);
		} catch (Exception e) {
			throw new RuntimeException(e);
		}

	}





	public void update(SdAMatrix matrix) {
		this.layers = matrix.layers;
		this.hiddenLayerSizes = matrix.hiddenLayerSizes;
		this.logLayer = matrix.logLayer;
		this.nIns = matrix.nIns;
		this.nLayers = matrix.nLayers;
		this.nOuts = matrix.nOuts;
		this.rng = matrix.rng;
		this.sigmoidLayers = matrix.sigmoidLayers;

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