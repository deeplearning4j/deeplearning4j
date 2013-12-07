package com.ccc.sendalyzeit.textanalytics.algorithms.deeplearning.sda.matrix.jblas;

import java.io.InputStream;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.OutputStream;
import java.io.Serializable;
import java.util.Arrays;

import org.apache.commons.math3.random.JDKRandomGenerator;
import org.jblas.DoubleMatrix;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.ccc.sendalyzeit.textanalytics.util.MathUtils;

/**
 * A JBlas implementation of 
 * stacked deep auto encoders.
 * @author Adam Gibson
 *
 */
public class SdAMatrix implements Serializable {

	private static final long serialVersionUID = 1448581794985193009L;
	public int N;
	public int n_ins;
	public int[] hidden_layer_sizes;
	public int n_outs;
	public int n_layers;
	public HiddenLayerMatrix[] sigmoid_layers;
	public DenoisingAutoEncoderMatrix[] dA_layers;
	public LogisticRegressionMatrix log_layer;
	public JDKRandomGenerator rng;
	private static Logger log = LoggerFactory.getLogger(SdAMatrix.class);
	private DoubleMatrix input;

	public SdAMatrix() 	{}

	/**
	 * Constructor for an SdAMatrix (stacked denoising auto encoders vectorized)
	 * @param N the number of training examples
	 * @param n_ins the number of inputs
	 * @param hidden_layer_sizes the array of hidden layer sizes
	 * @param n_outs the number of outbound nodes in the network
	 * @param n_layers the number of layers 
	 * @param rng an rng where relevant
	 */
	public SdAMatrix(int N, int n_ins, int[] hidden_layer_sizes, int n_outs, int n_layers, JDKRandomGenerator rng,DoubleMatrix input) {
		int input_size;

		this.N = N;
		this.n_ins = n_ins;
		this.hidden_layer_sizes = hidden_layer_sizes;
		this.input = input;
		if(hidden_layer_sizes.length != n_layers)
			throw new IllegalArgumentException("Te number of hidden layer sizes must be equivalent to the n_layers argument which is a value of " + n_layers);

		this.n_outs = n_outs;
		this.n_layers = n_layers;

		this.sigmoid_layers = new HiddenLayerMatrix[n_layers];
		this.dA_layers = new DenoisingAutoEncoderMatrix[n_layers];

		if(rng == null)   {
			this.rng = new JDKRandomGenerator();
			this.rng.setSeed(1);

		}
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
				this.sigmoid_layers[i] = new HiddenLayerMatrix(this.N, input_size, this.hidden_layer_sizes[i], null, null, rng,layer_input);

			}
			else {
				layer_input = this.sigmoid_layers[i - 1].sample_h_given_v();
				// construct sigmoid_layer
				this.sigmoid_layers[i] = new HiddenLayerMatrix(this.N, input_size, this.hidden_layer_sizes[i], null, null, rng,layer_input);

			}
			
			// construct dA_layer
			this.dA_layers[i] = new DenoisingAutoEncoderMatrix(this.N, input_size, this.hidden_layer_sizes[i], this.sigmoid_layers[i].W, this.sigmoid_layers[i].b, null, rng);
		}

		// layer for output using LogisticRegressionMatrix
		this.log_layer = new LogisticRegressionMatrix(this.N, this.hidden_layer_sizes[this.n_layers-1], this.n_outs);
	}

	public void pretrain( double lr,  double corruption_level,  int epochs) {
		DoubleMatrix layer_input = null;

		for(int i = 0; i < n_layers; i++) {  // layer-wise                        
			//input layer
			if(i == 0)
				layer_input = input;
			else
				layer_input = this.sigmoid_layers[i - 1].sample_h_given_v(layer_input);

			DenoisingAutoEncoderMatrix da = this.dA_layers[i];

			for(int epoch = 0; epoch < epochs; epoch++)  
				da.train(layer_input, lr, corruption_level);
			
		}	


	}


	public void finetune(DoubleMatrix train_Y, double lr, int epochs) {

		DoubleMatrix layer_input = this.sigmoid_layers[sigmoid_layers.length - 1].sample_h_given_v();

		for(int epoch = 0; epoch < epochs; epoch++) 
			log_layer.train(layer_input, train_Y, lr);
		


	}

	public void write(OutputStream os) {
		try {
			ObjectOutputStream oos = new ObjectOutputStream(os);
			oos.writeObject(this);
		}catch(Exception e) {
			throw new RuntimeException(e);
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
		this.dA_layers = matrix.dA_layers;
		this.hidden_layer_sizes = matrix.hidden_layer_sizes;
		this.log_layer = matrix.log_layer;
		this.N = matrix.N;
		this.n_ins = matrix.n_ins;
		this.n_layers = matrix.n_layers;
		this.n_outs = matrix.n_outs;
		this.rng = matrix.rng;
		this.sigmoid_layers = matrix.sigmoid_layers;

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

		for(int j = 0; j < n_ins; j++) 
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
				layer_input[k] = MathUtils.sigmoid(linear_output);
			}


			log.info("Layer input arr " + Arrays.toString(layer_input));
			if(i < n_layers-1) {
				prev_layer_input = new double[sigmoid_layers[i].n_out];
				for(int j = 0; j<sigmoid_layers[i].n_out; j++) 
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