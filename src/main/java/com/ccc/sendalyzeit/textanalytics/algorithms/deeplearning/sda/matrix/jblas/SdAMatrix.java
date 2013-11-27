package com.ccc.sendalyzeit.textanalytics.algorithms.deeplearning.sda.matrix.jblas;

import java.io.Serializable;
import java.util.Arrays;
import java.util.Random;

import org.jblas.DoubleMatrix;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * A JBlas implementation of 
 * stacked deep auto encoders.
 * @author Adam Gibson
 *
 */
public class SdAMatrix implements Serializable {
	/**
	 * 
	 */
	private static final long serialVersionUID = 1448581794985193009L;
	public int N;
	public int n_ins;
	public int[] hidden_layer_sizes;
	public int n_outs;
	public int n_layers;
	public HiddenLayerMatrix[] sigmoid_layers;
	public DeepAutoEncoderMatrix[] dA_layers;
	public LogisticRegressionMatrix log_layer;
	public Random rng;
	private static Logger log = LoggerFactory.getLogger(SdAMatrix.class);

	public static double sigmoid(double x) {
		return 1.0 / (1.0 + Math.pow(Math.E, -x));
	}
	
	/**
	 * Constructor for an SdAMatrix (stacked deep auto encoders vectorized)
	 * @param N the number of training examples
	 * @param n_ins the number of inputs
	 * @param hidden_layer_sizes the array of hidden layer sizes
	 * @param n_outs the number of outbound nodes in the network
	 * @param n_layers the number of layers 
	 * @param rng an rng where relevant
	 */
	public SdAMatrix(int N, int n_ins, int[] hidden_layer_sizes, int n_outs, int n_layers, Random rng) {
		int input_size;

		this.N = N;
		this.n_ins = n_ins;
		this.hidden_layer_sizes = hidden_layer_sizes;
		this.n_outs = n_outs;
		this.n_layers = n_layers;

		this.sigmoid_layers = new HiddenLayerMatrix[n_layers];
		this.dA_layers = new DeepAutoEncoderMatrix[n_layers];

		if(rng == null)        this.rng = new Random(1234);
		else this.rng = rng;                

		// construct multi-layer
		for(int i = 0; i < this.n_layers; i++) {
			if(i == 0) {
				input_size = this.n_ins;
			} else {
				input_size = this.hidden_layer_sizes[i-1];
			}

			// construct sigmoid_layer
			this.sigmoid_layers[i] = new HiddenLayerMatrix(this.N, input_size, this.hidden_layer_sizes[i], null, null, rng);

			// construct dA_layer
			this.dA_layers[i] = new DeepAutoEncoderMatrix(this.N, input_size, this.hidden_layer_sizes[i], this.sigmoid_layers[i].W, this.sigmoid_layers[i].b, null, rng);
		}

		// layer for output using LogisticRegressionMatrix
		this.log_layer = new LogisticRegressionMatrix(this.N, this.hidden_layer_sizes[this.n_layers-1], this.n_outs);
	}

	public void pretrain(DoubleMatrix train_X,  double lr, final double corruption_level, final int epochs) {
		for(int i = 0; i < n_layers; i++) {  // layer-wise                        
			log.info("Layer " + i);
			final int copyI = i;

			DoubleMatrix layer_input = null;
			DoubleMatrix prev_layer_input;

			for(int epoch = 0; epoch < epochs; epoch++) {  // training epochs
				log.info("Epoch " + (epoch + 1));

				for(int n = 0; n < N; n++) {  // input x1...xN
					// layer input/forward propagation
					for(int l = 0; l <= copyI; l++) {
						//initial input: 
						if(l == 0) 
							layer_input = train_X.getRow(n).transpose();


						else {
							prev_layer_input = layer_input.dup();
							//note that this will be a column vector on input for consistency
							layer_input = sigmoid_layers[l-1].sample_h_given_v(prev_layer_input,hidden_layer_sizes[copyI]);
						}
					}
					dA_layers[copyI].train(layer_input, lr, corruption_level);
				}
			}	


		}



	}

	public void finetune(DoubleMatrix train_X,DoubleMatrix train_Y, double lr, int epochs) {
		for(int epoch = 0; epoch < epochs; epoch++) {
			log.info("Epoch " + epoch + " on finetune");
			for(int n = 0; n < N; n++) {
				DoubleMatrix layer_input = null;
				// int prev_layer_input_size;
				DoubleMatrix prev_layer_input;
				// layer input
				for(int i = 0; i < n_layers; i++) {
					log.info("Training on " + i + " layer ");
					//always be a column vector
					if(i == 0) 
						prev_layer_input = train_X.getRow(n);
					else 
						prev_layer_input = layer_input.dup();

					//shift to a column vector
					layer_input = sigmoid_layers[i].sample_h_given_v(prev_layer_input.transpose(),hidden_layer_sizes[i]);
				}
				DoubleMatrix inputCopy = layer_input.dup();
				log.info("Output training");
				log_layer.train(inputCopy, train_Y.getRow(n), lr);

			}
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


	public DoubleMatrix feedForward(DoubleMatrix layer_input2) {
		double[] layer_input = new double[0];


		double linear_output;
		double[] prev_layer_input = new double[n_ins];

		for(int j=0; j<n_ins; j++) 
			prev_layer_input[j] = layer_input2.get(j);

		// layer activation
		for(int i = 0; i < n_layers; i++) {
			layer_input = new double[sigmoid_layers[i].n_out];

			for(int k = 0; k < sigmoid_layers[i].n_out; k++) {
				linear_output = 0.0;
				double[] w = sigmoid_layers[i].W.getRow(k).toArray();
				DoubleMatrix w2 = new DoubleMatrix(w);
				DoubleMatrix prevInput = new DoubleMatrix(prev_layer_input);
				double score = sigmoid_layers[i].b.get(k) + w2.dot(prevInput);
				log.info(String.valueOf("Score for layer 1 " + score));
				layer_input[k] = sigmoid(linear_output);
			}


			log.info("Layer input arr " + Arrays.toString(layer_input));
			if(i < n_layers-1) {
				prev_layer_input = new double[sigmoid_layers[i].n_out];
				for(int j=0; j<sigmoid_layers[i].n_out; j++) 
					prev_layer_input[j] = layer_input[j];
			}
		}
		return new DoubleMatrix(layer_input);
	}

	

	public static boolean continues(DoubleMatrix current, DoubleMatrix previous,double distanceThreshold) {
		int index = outcome(current);
		double curr = current.get(index);
		double prev = previous.get(index);
		return Math.abs(curr - prev) <= distanceThreshold;
	}


	public static DoubleMatrix outcomes(DoubleMatrix y) {
		DoubleMatrix ret = DoubleMatrix.zeros(y.rows, 1);
		for(int i = 0; i < y.rows; i++)
			ret.put(i,outcome(y.getRow(i)));
		return ret;
	}


	public static int outcome(DoubleMatrix outcomes) {
		if(!outcomes.isColumnVector() && !outcomes.isRowVector())
			throw new IllegalArgumentException("Outcomes is not a vector");
		return outcomes.elementsAsList().indexOf(outcomes.max());
	}

	
}