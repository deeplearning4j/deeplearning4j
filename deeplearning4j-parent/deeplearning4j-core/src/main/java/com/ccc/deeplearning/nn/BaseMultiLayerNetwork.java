package com.ccc.deeplearning.nn;

import java.io.IOException;
import java.io.InputStream;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.OutputStream;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;

import org.apache.commons.math3.random.MersenneTwister;
import org.apache.commons.math3.random.RandomGenerator;
import org.jblas.DoubleMatrix;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.ccc.deeplearning.berkeley.Pair;
import com.ccc.deeplearning.dbn.CDBN;
import com.ccc.deeplearning.nn.activation.ActivationFunction;
import com.ccc.deeplearning.nn.activation.Sigmoid;
import com.ccc.deeplearning.optimize.MultiLayerNetworkOptimizer;
import com.ccc.deeplearning.rbm.CRBM;
import com.ccc.deeplearning.rbm.RBM;
import com.ccc.deeplearning.util.MatrixUtil;


/**
 * A base class for a multi layer neural network with a logistic output layer
 * and multiple hidden layers.
 * @author Adam Gibson
 *
 */
public abstract class BaseMultiLayerNetwork implements Serializable,Persistable {

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
	public double momentum = 0.1;
	//default training examples and associated layers
	public DoubleMatrix input,labels;
	public MultiLayerNetworkOptimizer optimizer;
	public ActivationFunction activation = new Sigmoid();
	public boolean toDecode;
	public boolean shouldInit = true;

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
		this.input = input.dup();
		this.labels = labels.dup();

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

	private void dimensionCheck() {

		for(int i = 0; i < nLayers; i++) {
			HiddenLayer h = sigmoidLayers[i];
			NeuralNetwork network = layers[i];
			h.W.assertSameSize(network.getW());
			h.b.assertSameSize(network.gethBias());

			if(i < nLayers - 1) {
				HiddenLayer h1 = sigmoidLayers[i + 1];
				NeuralNetwork network1 = layers[i + 1];
				if(h1.n_in != h.n_out)
					throw new IllegalStateException("Invalid structure: hidden layer in for " + (i + 1) + " not equal to number of ins " + i);
				if(network.getnHidden() != network1.getnVisible())
					throw new IllegalStateException("Invalid structure: network hidden for " + (i + 1) + " not equal to number of visible " + i);

			}
		}

		if(sigmoidLayers[sigmoidLayers.length - 1].n_out != logLayer.nIn)
			throw new IllegalStateException("Number of outputs for final hidden layer not equal to the number of logistic input units for output layer");


	}

	/**
	 * Set as decoder for another neural net
	 * designed for encoding (primary output is
	 * encoding input)
	 * @param network the network to decode
	 */
	public void asDecoder(BaseMultiLayerNetwork network) {

		//need the plus 1 adjustment for handling inputs from
		//the logistic softmax of the output layer
		//of the encoder
		createNetworkLayers(network.nLayers + 1);
		this.layers = new NeuralNetwork[network.nLayers];
		this.sigmoidLayers = new HiddenLayer[network.nLayers];
		hiddenLayerSizes = new int[network.nLayers];
		this.nIns = network.nOuts;
		this.nOuts = network.nIns;
		this.nLayers = network.nLayers;



		int count = 0;
		for(int i = network.nLayers - 1; i >= 0; i--) {
			layers[count] = network.layers[i].clone();
			layers[count].setRng(network.layers[i].getRng());
			hiddenLayerSizes[count] = network.hiddenLayerSizes[i];

			count++;
		}
		this.rng = network.rng;


		//disable normal initialization
		shouldInit = false;
	}

	/**
	 * Base class for initializing the layers based on the input.
	 * This is meant for capturing numbers such as input columns or other things.
	 * @param input the input matrix for training
	 */
	protected void initializeLayers(DoubleMatrix input) {
		if(input == null)
			throw new IllegalArgumentException("Unable to initialize layers with empty input");

		this.input = input.dup();
		DoubleMatrix layer_input = input;
		int inputSize;

		// construct multi-layer
		for(int i = 0; i < this.nLayers; i++) {
			if(i == 0) 
				inputSize = this.nIns;
			else 
				inputSize = this.hiddenLayerSizes[i-1];

			if(i == 0) {
				// construct sigmoid_layer
				this.sigmoidLayers[i] = new HiddenLayer(inputSize, this.hiddenLayerSizes[i], null, null, rng,layer_input);
				sigmoidLayers[i].activationFunction = activation;
			}
			else {
				layer_input = sigmoidLayers[i - 1].sample_h_given_v();
				// construct sigmoid_layer
				this.sigmoidLayers[i] = new HiddenLayer(inputSize, this.hiddenLayerSizes[i], null, null, rng,layer_input);
				sigmoidLayers[i].activationFunction = activation;


			}

			// construct dA_layer
			//if(shouldInit)
			this.layers[i] = createLayer(layer_input,inputSize, this.hiddenLayerSizes[i], this.sigmoidLayers[i].W, this.sigmoidLayers[i].b, null, rng,i);
		}

		// layer for output using LogisticRegression
		this.logLayer = new LogisticRegression(layer_input, this.hiddenLayerSizes[this.nLayers-1], this.nOuts);



		dimensionCheck();

	}


	public void finetune(double lr, int epochs) {
		finetune(this.labels,lr,epochs);

	}

	public List<DoubleMatrix> feedForward() {
		if(this.input == null)
			throw new IllegalStateException("Unable to perform feed forward; no input found");
		List<DoubleMatrix> activations = new ArrayList<>();
		DoubleMatrix input = this.input;
		activations.add(input);

		for(int i = 0; i < nLayers; i++) {
			HiddenLayer layer = sigmoidLayers[i];
			layers[i].setInput(input);
			input = layer.activate(input);
			activations.add(input);
		}

		activations.add(logLayer.predict(input));
		return activations;
	}

	private void computeDeltas(List<DoubleMatrix> activations,List<Pair<DoubleMatrix,DoubleMatrix>> deltaRet) {
		DoubleMatrix[] gradients = new DoubleMatrix[nLayers + 2];
		DoubleMatrix[] deltas = new DoubleMatrix[nLayers + 2];
		ActivationFunction derivative = sigmoidLayers[0].activationFunction;
		//- y - h
		DoubleMatrix delta = null;


		/*
		 * Precompute activations and z's (pre activation network outputs)
		 */
		List<DoubleMatrix> weights = new ArrayList<>();
		for(int j = 0; j < layers.length; j++)
			weights.add(layers[j].getW());
		weights.add(logLayer.W);

		List<DoubleMatrix> zs = new ArrayList<>();
		zs.add(input);
		for(int i = 0; i < layers.length; i++) {
			if(layers[i].getInput() == null && i == 0) {
				layers[i].setInput(input);
			}
			else if(layers[i].getInput() == null){
				this.feedForward();
				if(layers[i].getInput() == null)	
					throw new IllegalStateException("WTF IS THIS");
			}

			zs.add(layers[i].getInput().mmul(weights.get(i)).addRowVector(layers[i].gethBias()));
		}
		zs.add(logLayer.input.mmul(logLayer.W).addRowVector(logLayer.b));

		//errors
		for(int i = nLayers + 1; i >= 0; i--) {
			if(i >= nLayers + 1) {
				DoubleMatrix z = zs.get(i);
				//- y - h
				delta = labels.sub(activations.get(i)).neg();

				//(- y - h) .* f'(z^l) where l is the output layer
				DoubleMatrix initialDelta = delta.mul(derivative.applyDerivative(z));
				deltas[i] = initialDelta;

			}
			else {
				delta = deltas[i + 1];
				DoubleMatrix w = weights.get(i).transpose();
				DoubleMatrix z = zs.get(i);
				DoubleMatrix a = activations.get(i + 1);
				//W^t * error^l + 1

				DoubleMatrix error = delta.mmul(w);
				deltas[i] = error;

				error = error.mul(derivative.applyDerivative(z));

				deltas[i] = error;
				gradients[i] = a.transpose().mmul(error).transpose();
			}

		}




		for(int i = 0; i < gradients.length; i++) {
			deltaRet.add(new Pair<>(gradients[i],deltas[i]));
		}






	}

	/**
	 * Backpropagation of errors for weights
	 * @param lr the learning rate to use
	 * @param epochs  the number of epochs to iterate (this is already called in finetune)
	 */
	public void backProp(double lr,int epochs) {
		double errorThreshold = 0.0001;
		Double lastEntropy = null;
		NeuralNetwork[] copies = new NeuralNetwork[this.layers.length];
		LogisticRegression reg = this.logLayer;
		double numMistakes = 0;

		for(int i = 0; i < copies.length; i++) {
			copies[i] = layers[i].clone();
		}






		for(int i = 0; i < epochs; i++) {
			List<DoubleMatrix> activations = feedForward();

			//precompute deltas
			List<Pair<DoubleMatrix,DoubleMatrix>> deltas = new ArrayList<>();
			computeDeltas(activations, deltas);
			double sse = this.negativeLogLikelihood();
			if(lastEntropy == null)
				lastEntropy = sse;
			else if(sse < lastEntropy) {
				lastEntropy = sse;
				copies = new NeuralNetwork[this.layers.length];
				reg = this.logLayer;
				for(int j = 0; j < copies.length; j++) {
					copies[j] = layers[j].clone();
				}

			}
			else if(sse > lastEntropy || sse == lastEntropy || Double.isNaN(sse) || Double.isInfinite(sse)) {
				numMistakes++;
				if(numMistakes >= 30) {
					this.logLayer = reg;
					this.layers = copies;
					log.info("Entropy went up; restoring from last good state");
					break;
				}



			}

			if(sse < errorThreshold )
				break;
			if(i % 10 == 0 || i == 0) {
				log.info("SSE on epoch " + i + " is  " + sse);
				log.info("Negative log likelihood is " + this.negativeLogLikelihood());
			}

			for(int l = 0; l < nLayers; l++) {
				DoubleMatrix add = deltas.get(l).getFirst().div(input.rows).mul(lr);
				layers[l].setW(layers[l].getW().sub(add));
				sigmoidLayers[l].W = layers[l].getW();
				DoubleMatrix deltaColumnSums = deltas.get(l + 1).getSecond().columnSums();
				layers[l].gethBias().subi(deltaColumnSums);
				sigmoidLayers[l].b = layers[l].gethBias();
			}


			logLayer.W.addi(deltas.get(nLayers).getFirst());


		}





	}

	/**
	 * Run SGD based on the given labels
	 * @param labels the labels to use
	 * @param lr the learning rate during training
	 * @param epochs the number of times to iterate
	 */
	public void finetune(DoubleMatrix labels,double lr, int epochs) {
		if(labels != null)
			this.labels = labels;
		optimizer = new MultiLayerNetworkOptimizer(this,lr);
		optimizer.optimize(this.labels, lr,epochs);
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
			input = layer.activate(input);
		}
		return (logLayer.predict(input));
	}


	/**
	 * Reconstructs the input.
	 * This is equivalent functionality to a 
	 * deep autoencoder.
	 * @param x the input to reconstruct
	 * @param layerNum the layer to output for encoding
	 * @return a reconstructed matrix
	 * relative to the size of the last hidden layer.
	 * This is great for data compression and visualizing
	 * high dimensional data (or just doing dimensionality reduction).
	 * 
	 * This is typically of the form:
	 * [0.5, 0.5] or some other probability distribution summing to one
	 */
	public DoubleMatrix reconstruct(DoubleMatrix x,int layerNum) {
		if(layerNum > nLayers || layerNum < 0)
			throw new IllegalArgumentException("Layer number " + layerNum + " does not exist");

		DoubleMatrix input = x;
		for(int i = 0; i < layerNum; i++) {
			HiddenLayer layer = sigmoidLayers[i];
			input = layer.activate(input);

		}
		return MatrixUtil.softmax(input);
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
			log.info("Loading network model...");

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


	/**
	 * Merges this network with the other one.
	 * This is a weight averaging with the update of:
	 * a += b - a / n
	 * where a is a matrix on the network
	 * b is the incoming matrix and n
	 * is the batch size.
	 * This update is performed across the network layers
	 * as well as hidden layers and logistic layers
	 * 
	 * @param network the network to merge with
	 * @param batchSize the batch size (number of training examples)
	 * to average by
	 */
	public void merge(BaseMultiLayerNetwork network,int batchSize) {
		if(network.nLayers != nLayers)
			throw new IllegalArgumentException("Unable to merge networks that are not of equal length");
		for(int i = 0; i < nLayers; i++) {
			NeuralNetwork n = layers[i];
			NeuralNetwork otherNetwork = network.layers[i];
			n.merge(otherNetwork, batchSize);
			//tied weights: must be updated at the same time
			HiddenLayer l = sigmoidLayers[i];
			l.b = n.gethBias();
			l.W = n.getW();
		}

		logLayer.merge(network.logLayer, batchSize);
	}

	public void encode(BaseMultiLayerNetwork network) {
		this.createNetworkLayers(network.nLayers);
		this.layers = new NeuralNetwork[network.nLayers];
		hiddenLayerSizes = new int[nLayers];

		int count = 0;
		for(int i = nLayers - 1; i > 0; i--) {
			NeuralNetwork n = network.layers[i].clone();
			//tied weights: must be updated at the same time
			HiddenLayer l = network.sigmoidLayers[i].clone();
			layers[count] = n;
			sigmoidLayers[count] = l;
			hiddenLayerSizes[count] = network.hiddenLayerSizes[i];
			count++;
		}

		this.logLayer = new LogisticRegression(hiddenLayerSizes[nLayers - 1],network.input.columns);

	}


	public static class Builder<E extends BaseMultiLayerNetwork> {
		protected Class<? extends BaseMultiLayerNetwork> clazz;
		private E ret;
		private int nIns;
		private int[] hiddenLayerSizes;
		private int nOuts;
		private int nLayers;
		private RandomGenerator rng = new MersenneTwister(1234);
		private DoubleMatrix input,labels;
		private ActivationFunction activation;
		private boolean decode = false;

		public Builder<E> withActivation(ActivationFunction activation) {
			this.activation = activation;
			return this;
		}


		public Builder<E> numberOfInputs(int nIns) {
			this.nIns = nIns;
			return this;
		}

		public Builder<E> decodeNetwork(boolean decode) {
			this.decode = decode;
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
				ret.input = this.input;
				ret.nOuts = this.nOuts;
				ret.nIns = this.nIns;
				ret.labels = this.labels;
				ret.hiddenLayerSizes = this.hiddenLayerSizes;
				ret.nLayers = this.nLayers;
				ret.rng = this.rng;
				ret.sigmoidLayers = new HiddenLayer[ret.nLayers];
				ret.layers = ret.createNetworkLayers(ret.nLayers);
				ret.toDecode = this.decode;
				if(activation != null)
					ret.activation = activation;
				return ret;
			} catch (InstantiationException | IllegalAccessException e) {
				throw new RuntimeException(e);
			}

		}





	}


}
