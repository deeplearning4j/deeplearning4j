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
import com.ccc.sendalyzeit.textanalytics.util.MatrixUtil;


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


	public void pretrain( double lr,  double corruption_level,  int epochs) {
		DoubleMatrix layer_input = null;

		for(int i = 0; i < n_layers; i++) {  // layer-wise                        
			//input layer
			if(i == 0)
				layer_input = input;
			else
				layer_input = this.sigmoid_layers[i - 1].sample_h_given_v(layer_input);

			DenoisingAutoEncoderMatrix da = (DenoisingAutoEncoderMatrix) this.layers[i];
			HiddenLayerMatrix h = this.sigmoid_layers[i];
			for(int epoch = 0; epoch < epochs; epoch++)  {
				da.train(layer_input, lr, corruption_level);
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
		this.hidden_layer_sizes = matrix.hidden_layer_sizes;
		this.log_layer = matrix.log_layer;
		this.n_ins = matrix.n_ins;
		this.n_layers = matrix.n_layers;
		this.n_outs = matrix.n_outs;
		this.rng = matrix.rng;
		this.sigmoid_layers = matrix.sigmoid_layers;

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





}