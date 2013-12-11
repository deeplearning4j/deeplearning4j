package com.ccc.sendalyzeit.textanalytics.algorithms.deeplearning.nn.matrix.jblas;

import java.io.ObjectOutputStream;
import java.io.OutputStream;
import java.io.Serializable;

import org.apache.commons.math3.random.MersenneTwister;
import org.apache.commons.math3.random.RandomGenerator;
import org.jblas.DoubleMatrix;
/**
 * A base class for a multi layer neural network with a logistic output layer
 * and multiple hidden layers.
 * @author Adam Gibson
 *
 */
public abstract class BaseMultiLayerNetwork implements Serializable {
	
	private static final long serialVersionUID = -5029161847383716484L;
	public int n_ins;
	public int[] hidden_layer_sizes;
	public int n_outs;
	public int n_layers;
	public HiddenLayerMatrix[] sigmoid_layers;
	public LogisticRegressionMatrix log_layer;
	public RandomGenerator rng;
	public DoubleMatrix input,labels;
	public NeuralNetwork[] layers;
	
	public BaseMultiLayerNetwork(int n_ins, int[] hidden_layer_sizes, int n_outs, int n_layers, RandomGenerator rng,DoubleMatrix input,DoubleMatrix labels) {
		int input_size;
		this.n_ins = n_ins;
		this.hidden_layer_sizes = hidden_layer_sizes;
		this.input = input;
		this.labels = labels;
		
		if(hidden_layer_sizes.length != n_layers)
			throw new IllegalArgumentException("Te number of hidden layer sizes must be equivalent to the n_layers argument which is a value of " + n_layers);

		this.n_outs = n_outs;
		this.n_layers = n_layers;

		this.sigmoid_layers = new HiddenLayerMatrix[n_layers];
		this.layers = createNetworkLayers(n_layers);

		if(rng == null)   
			this.rng = new MersenneTwister(123);


		else 
			this.rng = rng;                
		DoubleMatrix layer_input = input;
		// construct multi-layer
		for(int i = 0; i < this.n_layers; i++) {
			if(i == 0) 
				input_size = this.n_ins;
			else 
				input_size = this.hidden_layer_sizes[i-1];

			if(i == 0) {
				// construct sigmoid_layer
				this.sigmoid_layers[i] = new HiddenLayerMatrix(input_size, this.hidden_layer_sizes[i], null, null, rng,layer_input);

			}
			else {
				layer_input = sigmoid_layers[i - 1].sample_h_given_v();
				// construct sigmoid_layer
				this.sigmoid_layers[i] = new HiddenLayerMatrix(input_size, this.hidden_layer_sizes[i], null, null, rng,layer_input);

			}

			// construct dA_layer
			this.layers[i] = createLayer(layer_input,input_size, this.hidden_layer_sizes[i], this.sigmoid_layers[i].W, this.sigmoid_layers[i].b, null, rng,i);
		}

		// layer for output using LogisticRegressionMatrix
		this.log_layer = new LogisticRegressionMatrix(layer_input, this.hidden_layer_sizes[this.n_layers-1], this.n_outs);
	}

	
	public void finetune(double lr, int epochs) {

		DoubleMatrix layer_input = this.sigmoid_layers[sigmoid_layers.length - 1].sample_h_given_v();

		for(int epoch = 0; epoch < epochs; epoch++) {
			log_layer.train(layer_input, labels, lr);
			lr *= 0.95;
		}


	}
	
	
	public DoubleMatrix predict(DoubleMatrix x) {
		DoubleMatrix input = x;
		for(int i = 0; i < n_layers; i++) {
			HiddenLayerMatrix layer = sigmoid_layers[i];
			input = layer.outputMatrix(input);
		}
		return log_layer.predict(input);
	}
	
	
	

	public void write(OutputStream os) {
		try {
			ObjectOutputStream oos = new ObjectOutputStream(os);
			oos.writeObject(this);
		}catch(Exception e) {
			throw new RuntimeException(e);
		}
	}

	
	public abstract NeuralNetwork createLayer(DoubleMatrix input,int nVisible,int nHidden, DoubleMatrix W,DoubleMatrix hbias,DoubleMatrix vBias,RandomGenerator rng,int index);
	
	
	public abstract NeuralNetwork[] createNetworkLayers(int numLayers);

}
