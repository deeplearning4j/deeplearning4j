package com.ccc.sendalyzeit.textanalytics.algorithms.deeplearning.nn.matrix.jblas;

import java.io.IOException;
import java.io.InputStream;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.OutputStream;
import java.io.Serializable;

import org.apache.commons.math3.random.MersenneTwister;
import org.apache.commons.math3.random.RandomGenerator;
import org.jblas.DoubleMatrix;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.ccc.sendalyzeit.textanalytics.util.MatrixUtil;
/**
 * A base class for a multi layer neural network with a logistic output layer
 * and multiple hidden layers.
 * @author Adam Gibson
 *
 */
public abstract class BaseMultiLayerNetwork implements Serializable {

	private static Logger log = LoggerFactory.getLogger(BaseMultiLayerNetwork.class);
	private static final long serialVersionUID = -5029161847383716484L;
	//number of columns in the input matrix
	public int nIns;
	//the hidden layer sizes at each layer
	public int[] hiddenLayerSizes;
	//the number of outputs/labels for logistic regression
	public int nOuts;
	public int nLayers;
	//the hidden layers
	public HiddenLayer[] sigmoidLayers;
	//logistic regression output layer (aka the softmax layer) for translating network outputs in to probabilities
	public LogisticRegression logLayer;
	public RandomGenerator rng;
	//default training examples and associated layers
	public DoubleMatrix input,labels;
	/*
	 * Hinton's Practical guide to RBMS:
	 * 
	 * Learning rate updates over time.
	 * Usually when weights have a large fan in (starting point for a random distribution during initialization)
	 * you want a smaller update rate.
	 * 
	 * For biases this can be bigger.
	 */
	public double learningRateUpdate = 0.95;
	/*
	 * Any neural networks used as layers.
	 * This will always have an equivalent sigmoid layer
	 * with shared weight matrices for training.
	 * 
	 * Typically, this is some mix of:
	 *        RBMs,Denoising AutoEncoders, or their Continuous counterparts
	 */
	public NeuralNetwork[] layers;
	
	/*
	 * The delta from the previous iteration to this iteration for
	 * cross entropy must change >= this amount in order to continue.
	 */
	public double errorTolerance = 0.0001;

	/* Reflection/factory constructor */
	public BaseMultiLayerNetwork() {}

	public BaseMultiLayerNetwork(int n_ins, int[] hidden_layer_sizes, int n_outs, int n_layers, RandomGenerator rng) {
		this(n_ins,hidden_layer_sizes,n_outs,n_layers,rng,null,null);
	}


	public BaseMultiLayerNetwork(int n_ins, int[] hidden_layer_sizes, int n_outs, int n_layers, RandomGenerator rng,DoubleMatrix input,DoubleMatrix labels) {
		this.nIns = n_ins;
		this.hiddenLayerSizes = hidden_layer_sizes;
		this.input = input;
		this.labels = labels;

		if(hidden_layer_sizes.length != n_layers)
			throw new IllegalArgumentException("The number of hidden layer sizes must be equivalent to the nLayers argument which is a value of " + n_layers);

		this.nOuts = n_outs;
		this.nLayers = n_layers;

		this.sigmoidLayers = new HiddenLayer[n_layers];
		this.layers = createNetworkLayers(n_layers);

		if(rng == null)   
			this.rng = new MersenneTwister(123);


		else 
			this.rng = rng;  


		if(input != null) 
			initializeLayers(input);


	}

	/**
	 * Base class for initializing the layers based on the input.
	 * This is meant for capturing numbers such as input columns or other things.
	 * @param input the input matrix for training
	 */
	protected void initializeLayers(DoubleMatrix input) {
		DoubleMatrix layer_input = input;
		int input_size;

		// construct multi-layer
		for(int i = 0; i < this.nLayers; i++) {
			if(i == 0) 
				input_size = this.nIns;
			else 
				input_size = this.hiddenLayerSizes[i-1];

			if(i == 0) {
				// construct sigmoid_layer
				this.sigmoidLayers[i] = new HiddenLayer(input_size, this.hiddenLayerSizes[i], null, null, rng,layer_input);

			}
			else {
				layer_input = sigmoidLayers[i - 1].sample_h_given_v();
				// construct sigmoid_layer
				this.sigmoidLayers[i] = new HiddenLayer(input_size, this.hiddenLayerSizes[i], null, null, rng,layer_input);

			}

			// construct dA_layer
			this.layers[i] = createLayer(layer_input,input_size, this.hiddenLayerSizes[i], this.sigmoidLayers[i].W, this.sigmoidLayers[i].b, null, rng,i);
		}

		// layer for output using LogisticRegression
		this.logLayer = new LogisticRegression(layer_input, this.hiddenLayerSizes[this.nLayers-1], this.nOuts);

	}


	public void finetune(double lr, int epochs) {
		finetune(this.labels,lr,epochs);

	}

	/**
	 * Run SGD based on the given labels
	 * @param labels the labels to use
	 * @param lr the learning rate during training
	 * @param epochs the number of times to iterate
	 */
	public void finetune(DoubleMatrix labels,double lr, int epochs) {
		MatrixUtil.ensureValidOutcomeMatrix(labels);
		//sample from the final layer in the network and train on the result
		DoubleMatrix layerInput = this.sigmoidLayers[sigmoidLayers.length - 1].sample_h_given_v();
		logLayer.input = layerInput;
		logLayer.labels = labels;
		double cost = this.negativeLogLikelihood();
		boolean done = false;
		while(!done) {
			DoubleMatrix W = logLayer.W.dup();
			logLayer.train(layerInput, labels, lr);
			lr *= learningRateUpdate;
			double currCost = this.negativeLogLikelihood();
			if(currCost <= cost) {
				double diff = Math.abs(cost - currCost);
				if(diff <= 0.0000001) {
					done = true;
					log.info("Converged on finetuning at " + cost);
					break;
				}
				else
					cost = currCost;
				log.info("Found new log likelihood " + cost);
			}
			
			else if(currCost > cost) {
				done = true;
				logLayer.W = W;
				log.info("Converged on finetuning at " + cost + " due to a higher cost coming out than " + currCost);
				break;
			}
		}


	}



	/**
	 * Label the probabilities of the input
	 * @param x the input to label
	 * @return a vector of probabilities
	 * given each label.
	 * 
	 * This is typically of the form:
	 * [0.5, 0.5] or some other probability distribution summing to one
	 */
	public DoubleMatrix predict(DoubleMatrix x) {
		DoubleMatrix input = x;
		for(int i = 0; i < nLayers; i++) {
			HiddenLayer layer = sigmoidLayers[i];
			input = layer.outputMatrix(input);
		}
		return logLayer.predict(input);
	}


	/**
	 * Serializes this to the output stream.
	 * @param os the output stream to write to
	 */
	public void write(OutputStream os) {
		try {
			ObjectOutputStream oos = new ObjectOutputStream(os);
			oos.writeObject(this);

		} catch (IOException e) {
			throw new RuntimeException(e);
		}

	}

	/**
	 * Load (using {@link ObjectInputStream}
	 * @param is the input stream to load from (usually a file)
	 */
	public void load(InputStream is) {
		try {
			ObjectInputStream ois = new ObjectInputStream(is);
			BaseMultiLayerNetwork loaded = (BaseMultiLayerNetwork) ois.readObject();
			update(loaded);
		} catch (Exception e) {
			throw new RuntimeException(e);
		}

	}

	/**
	 * Load (using {@link ObjectInputStream}
	 * @param is the input stream to load from (usually a file)
	 */
	public static BaseMultiLayerNetwork loadFromFile(InputStream is) {
		try {
			ObjectInputStream ois = new ObjectInputStream(is);
			BaseMultiLayerNetwork loaded = (BaseMultiLayerNetwork) ois.readObject();
			return loaded;
		} catch (Exception e) {
			throw new RuntimeException(e);
		}

	}



	/**
	 * Assigns the parameters of this model to the ones specified by this
	 * network. This is used in loading from input streams, factory methods, etc
	 * @param matrix the network to get parameters from
	 */
	protected void update(BaseMultiLayerNetwork matrix) {
		this.layers = matrix.layers;
		this.hiddenLayerSizes = matrix.hiddenLayerSizes;
		this.logLayer = matrix.logLayer;
		this.nIns = matrix.nIns;
		this.nLayers = matrix.nLayers;
		this.nOuts = matrix.nOuts;
		this.rng = matrix.rng;
		this.sigmoidLayers = matrix.sigmoidLayers;

	}

	/**
	 * Negative log likelihood of the model
	 * @return the negative log likelihood of the model
	 */
	public double negativeLogLikelihood() {
		return logLayer.negativeLogLikelihood();
	}
	
	/**
	 * Train the network running some unsupervised 
	 * pretraining followed by SGD/finetune
	 * @param input the input to train on
	 * @param labels the labels for the training examples(a matrix of the following format:
	 * [0,1,0] where 0 represents the labels its not and 1 represents labels for the positive outcomes 
	 * @param otherParams the other parameters for child classes (algorithm specific parameters such as corruption level for SDA)
	 */
	public abstract void trainNetwork(DoubleMatrix input,DoubleMatrix labels,Object[] otherParams);

	/**
	 * Creates a layer depending on the index.
	 * The main reason this matters is for continuous variations such as the {@link CDBN}
	 * where the first layer needs to be an {@link CRBM} for continuous inputs
	 * @param input the input to the layer
	 * @param nVisible the number of visible inputs
	 * @param nHidden the number of hidden units
	 * @param W the weight vector
	 * @param hbias the hidden bias
	 * @param vBias the visible bias
	 * @param rng the rng to use (THiS IS IMPORTANT; YOU DO NOT WANT TO HAVE A MIS REFERENCED RNG OTHERWISE NUMBERS WILL BE MEANINGLESS)
	 * @param index the index of the layer
	 * @return a neural network layer such as {@link RBM} 
	 */
	public abstract NeuralNetwork createLayer(DoubleMatrix input,int nVisible,int nHidden, DoubleMatrix W,DoubleMatrix hbias,DoubleMatrix vBias,RandomGenerator rng,int index);


	public abstract NeuralNetwork[] createNetworkLayers(int numLayers);




	public static class Builder<E extends BaseMultiLayerNetwork> {
		protected Class<? extends BaseMultiLayerNetwork> clazz;
		private E ret;
		private int nIns;
		private int[] hiddenLayerSizes;
		private int nOuts;
		private int nLayers;
		private RandomGenerator rng;
		private DoubleMatrix input,labels;




		public Builder<E> numberOfInputs(int nIns) {
			this.nIns = nIns;
			return this;
		}

		public Builder<E> hiddenLayerSizes(int[] hiddenLayerSizes) {
			this.hiddenLayerSizes = hiddenLayerSizes;
			this.nLayers = hiddenLayerSizes.length;
			return this;
		}

		public Builder<E> numberOfOutPuts(int nOuts) {
			this.nOuts = nOuts;
			return this;
		}

		public Builder<E> withRng(RandomGenerator gen) {
			this.rng = gen;
			return this;
		}

		public Builder<E> withInput(DoubleMatrix input) {
			this.input = input;
			return this;
		}

		public Builder<E> withLabels(DoubleMatrix labels) {
			this.labels = labels;
			return this;
		}

		public Builder<E> withClazz(Class<? extends BaseMultiLayerNetwork> clazz) {
			this.clazz =  clazz;
			return this;
		}


		@SuppressWarnings("unchecked")
		public E buildEmpty() {
			try {
				return (E) clazz.newInstance();
			} catch (Exception e) {
				throw new RuntimeException(e);
			} 
		}

		@SuppressWarnings("unchecked")
		public E build() {
			try {
				ret = (E) clazz.newInstance();
				ret.nOuts = this.nOuts;
				ret.nIns = this.nIns;
				ret.labels = this.labels;
				ret.hiddenLayerSizes = this.hiddenLayerSizes;
				ret.nLayers = this.nLayers;
				ret.rng = this.rng;
				ret.sigmoidLayers = new HiddenLayer[ret.nLayers];
				ret.layers = ret.createNetworkLayers(ret.nLayers);

				return ret;
			} catch (InstantiationException | IllegalAccessException e) {
				throw new RuntimeException(e);
			}
			
		}

	

	}


}
