package com.ccc.deeplearning.nn;

import java.io.IOException;
import java.io.InputStream;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.OutputStream;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.apache.commons.math3.distribution.RealDistribution;
import org.apache.commons.math3.random.MersenneTwister;
import org.apache.commons.math3.random.RandomGenerator;
import org.jblas.DoubleMatrix;
import org.jblas.MatrixFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.ccc.deeplearning.berkeley.Pair;
import com.ccc.deeplearning.dbn.CDBN;
import com.ccc.deeplearning.nn.activation.ActivationFunction;
import com.ccc.deeplearning.nn.activation.Sigmoid;
import com.ccc.deeplearning.optimize.MultiLayerNetworkOptimizer;
import com.ccc.deeplearning.rbm.CRBM;
import com.ccc.deeplearning.rbm.RBM;
import com.ccc.deeplearning.transformation.MatrixTransform;
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
	/* probability distribution for generation of weights */
	public RealDistribution dist;
	public double momentum = 0.1;
	//default training examples and associated layers
	public DoubleMatrix input,labels;
	public MultiLayerNetworkOptimizer optimizer;
	public ActivationFunction activation = new Sigmoid();
	public boolean toDecode;
	public double l2 = 0.01;
	public boolean shouldInit = true;
	public double fanIn = -1;
	public int renderWeightsEveryNEpochs = -1;
	public boolean useRegularization = true;
	protected Map<Integer,MatrixTransform> weightTransforms = new HashMap<Integer,MatrixTransform>();
	protected boolean shouldBackProp = true;
	protected boolean forceNumEpochs = false;
	//don't use sparsity by default
	public double sparsity = 0;
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


	public BaseMultiLayerNetwork(int nIn, int[] hiddenLayerSizes, int nOuts
			, int nLayers, RandomGenerator rng,DoubleMatrix input,DoubleMatrix labels) {
		this.nIns = nIn;
		this.hiddenLayerSizes = hiddenLayerSizes;
		this.input = input.dup();
		this.labels = labels.dup();

		if(hiddenLayerSizes.length != nLayers)
			throw new IllegalArgumentException("The number of hidden layer sizes must be equivalent to the nLayers argument which is a value of " + nLayers);

		this.nOuts = nOuts;
		this.nLayers = nLayers;

		this.sigmoidLayers = new HiddenLayer[nLayers];
		this.layers = createNetworkLayers(nLayers);

		if(rng == null)   
			this.rng = new MersenneTwister(123);


		else 
			this.rng = rng;  


		if(input != null) 
			initializeLayers(input);


	}

	protected boolean trainNetwork(NeuralNetwork network,HiddenLayer h,int epoch,int layer,DoubleMatrix input,double lr,Double bestLoss,Object[] params) {



		DoubleMatrix w = network.getW();
		DoubleMatrix hBias = network.gethBias();


		network.trainTillConvergence(input, lr, params);



		h.W = network.getW();
		h.b = network.gethBias();
		double entropy = network.getReConstructionCrossEntropy();


		if(Double.isNaN(entropy) || Double.isInfinite(entropy)) {
			network.setW(w.dup());
			network.sethBias(hBias.dup());
			h.W = network.getW();
			h.b = hBias; 
			log.info("Went over too many times....reverting to last known good state");

		}
		else if(entropy > bestLoss || entropy == bestLoss) {

			network.setW(w.dup());
			network.sethBias(hBias.dup());
			h.W = network.getW();
			h.b = hBias; 
			log.info("Went over too many times....reverting to last known good state");

			return false;
		}

		else if(entropy < bestLoss){
			w = network.getW().dup();
			hBias = network.gethBias().dup();
			bestLoss = entropy;
		}

		log.info("Cross entropy for layer " + (layer + 1) + " on epoch " + epoch + " is " + entropy);


		double curr = network.getReConstructionCrossEntropy();
		if(curr > bestLoss) {
			log.info("Converged past global minimum; reverting");
			network.setW(w.dup());
			network.sethBias(hBias.dup());
			return false;
		}
		return true;
	}


	/**
	 * Returns the -fanIn to fanIn
	 * coefficient used for initializing the
	 * weights.
	 * The default is 1 / nIns
	 * @return the fan in coefficient
	 */
	public double fanIn() {
		if(this.fanIn < 0)
			return 1.0 / nIns;
		return fanIn;
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
				if(h1.nIn != h.nOut)
					throw new IllegalStateException("Invalid structure: hidden layer in for " + (i + 1) + " not equal to number of ins " + i);
				if(network.getnHidden() != network1.getnVisible())
					throw new IllegalStateException("Invalid structure: network hidden for " + (i + 1) + " not equal to number of visible " + i);

			}
		}

		if(sigmoidLayers[sigmoidLayers.length - 1].nOut != logLayer.nIn)
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
		this.dist = network.dist;


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
		DoubleMatrix layerInput = input;
		int inputSize;

		// construct multi-layer
		for(int i = 0; i < this.nLayers; i++) {
			if(i == 0) 
				inputSize = this.nIns;
			else 
				inputSize = this.hiddenLayerSizes[i-1];

			if(i == 0) {
				// construct sigmoid_layer
				this.sigmoidLayers[i] = new HiddenLayer(inputSize, this.hiddenLayerSizes[i], null, null, rng,layerInput);
				sigmoidLayers[i].activationFunction = activation;
			}
			else {
				layerInput = sigmoidLayers[i - 1].sample_h_given_v();
				// construct sigmoid_layer
				this.sigmoidLayers[i] = new HiddenLayer(inputSize, this.hiddenLayerSizes[i], null, null, rng,layerInput);
				sigmoidLayers[i].activationFunction = activation;


			}

			this.layers[i] = createLayer(layerInput,inputSize, this.hiddenLayerSizes[i], this.sigmoidLayers[i].W, this.sigmoidLayers[i].b, null, rng,i);

		}

		// layer for output using LogisticRegression
		this.logLayer = new LogisticRegression(layerInput, this.hiddenLayerSizes[this.nLayers-1], this.nOuts);
		this.logLayer.useRegularization = this.useRegularization;
		this.logLayer.l2 = this.l2;


		dimensionCheck();
		applyTransforms();
	}


	protected void initializeNetwork(NeuralNetwork network) {
		network.setFanIn(fanIn);
		network.setRenderEpochs(this.renderWeightsEveryNEpochs);
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
		/*
		 * activations for each layer
		 */
		for(int i = 0; i < layers.length; i++) {
			if(layers[i].getInput() == null && i == 0) {
				layers[i].setInput(input);
			}
			else if(layers[i].getInput() == null){
				this.feedForward();
			}

			zs.add(MatrixUtil.sigmoid(layers[i].getInput().mmul(weights.get(i)).addRowVector(layers[i].gethBias())));
		}

		//account for final activation of log layer as well
		zs.add(logLayer.input.mmul(logLayer.W).addRowVector(logLayer.b));

		//errors
		for(int i = nLayers + 1; i >= 0; i--) {
			//output layer
			if(i >= nLayers + 1) {
				DoubleMatrix z = zs.get(i);
				//- y - h
				delta = labels.sub(activations.get(i)).neg();

				//(- y - h) .* f'(z^l) where l is the output layer
				DoubleMatrix initialDelta = delta.mul(derivative.applyDerivative(z));
				deltas[i] = initialDelta;

			}
			else {
				//derivative i + 1; aka gradient for bias
				delta = deltas[i + 1];
				DoubleMatrix w = weights.get(i).transpose();
				DoubleMatrix z = zs.get(i);
				DoubleMatrix a = activations.get(i);
				//W^t * error^l + 1

				DoubleMatrix error = delta.mmul(w);
				deltas[i] = error;

				error = error.mul(derivative.applyDerivative(z));

				deltas[i] = error;

				//calculate gradient for layer
				DoubleMatrix lastLayerDelta = deltas[i + 1].transpose();
				DoubleMatrix newGradient = lastLayerDelta.mmul(a);

				gradients[i] = newGradient.div(input.rows);
			}

		}

		for(int i = 0; i < gradients.length; i++) 
			deltaRet.add(new Pair<>(gradients[i],deltas[i]));

	}




	@Override
	protected BaseMultiLayerNetwork clone() {
		@SuppressWarnings("unchecked")
		BaseMultiLayerNetwork ret = new Builder<>().withClazz(getClass()).buildEmpty();
		ret.update(this);
		return ret;
	}

	/**
	 * Backpropagation of errors for weights
	 * @param lr the learning rate to use
	 * @param epochs  the number of epochs to iterate (this is already called in finetune)
	 */
	public void backProp(double lr,int epochs) {

		Double lastEntropy = null;
		//store a copy of the network for when binary cross entropy gets
		//worse after an iteration
		BaseMultiLayerNetwork revert = clone();

		if(forceNumEpochs) {
			for(int i = 0; i < epochs; i++) {
				backPropStep(lastEntropy,revert,lr,i);
				lastEntropy = negativeLogLikelihood();
			}
		}
		
		else {
			int count = 0;
			while(backPropStep(lastEntropy,revert,lr,count)) {
				count++;
				lastEntropy = this.negativeLogLikelihood();
			}
		}
		
		





	}
	
	protected boolean backPropStep(Double lastEntropy,BaseMultiLayerNetwork revert,double lr,int epoch) {
		//feedforward to compute activations
		List<DoubleMatrix> activations = feedForward();
		//initial error
		double error = this.negativeLogLikelihood();
		if(lastEntropy == null) {
			lastEntropy = error;
		}
		else if(error == lastEntropy) {
			log.info("Converged; no more stepping appears to do anything");
			return false;
		}
		
		else if(error > lastEntropy) {
			log.info("Error greater than previous; found global minima; converging");
			update(revert);
			return false;
		}
		else if(error < lastEntropy ) {
			lastEntropy = error;
			revert = clone();
			log.info("Found better error on epoch " + epoch + " " + lastEntropy);
		}

		//precompute deltas
		List<Pair<DoubleMatrix,DoubleMatrix>> deltas = new ArrayList<>();
		//compute derivatives and gradients given activations
		computeDeltas(activations, deltas);


		for(int l = 0; l < nLayers; l++) {
			DoubleMatrix add = deltas.get(l).getFirst().div(input.rows).mul(lr);
			add.divi(input.rows);
			//l2
			if(useRegularization) {
				add.muli(layers[l].getW().mul(l2));
			}

			//update W
			layers[l].setW(layers[l].getW().add(add.mul(lr)));
			sigmoidLayers[l].W = layers[l].getW();


			//update hidden bias
			DoubleMatrix deltaColumnSums = deltas.get(l + 1).getSecond().columnSums();
			deltaColumnSums.divi(input.rows);

			layers[l].gethBias().addi(deltaColumnSums.mul(lr));
			sigmoidLayers[l].b = layers[l].gethBias();
		}


		logLayer.W.addi(deltas.get(nLayers).getFirst());
		return true;

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
		for(int i = 0; i < nLayers; i++) 
			input = sigmoidLayers[i].activate(input);

		return logLayer.predict(input);
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
	 * @param network the network to get parameters from
	 */
	protected void update(BaseMultiLayerNetwork network) {
		this.layers = new NeuralNetwork[network.layers.length];
		for(int i = 0; i < layers.length; i++) {
			this.layers[i] = network.layers[i].clone();
		}
		this.hiddenLayerSizes = network.hiddenLayerSizes;
		this.logLayer = network.logLayer.clone();
		this.nIns = network.nIns;
		this.nLayers = network.nLayers;
		this.nOuts = network.nOuts;
		this.rng = network.rng;
		this.dist = network.dist;
		this.sigmoidLayers = new HiddenLayer[network.sigmoidLayers.length];
		for(int i = 0; i < sigmoidLayers.length; i++)
			this.sigmoidLayers[i] = network.sigmoidLayers[i].clone();


	}

	/**
	 * Negative log likelihood of the model
	 * @return the negative log likelihood of the model
	 */
	public double negativeLogLikelihood() {
		double ret  = 0.0;
		for(int i = 0; i < nLayers; i++) {
			ret += (MatrixFunctions.pow(layers[i].getW(),2).sum()/ 2.0)  * l2;
		}
		ret += (MatrixFunctions.pow(logLayer.W,2).sum()/ 2.0)  * l2;

		return ret;
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

	protected void applyTransforms() {
		if(layers == null || layers.length < 1) {
			throw new IllegalStateException("Layers not initialized");
		}

		for(int i = 0; i < layers.length; i++) {
			if(weightTransforms.containsKey(i)) 
				layers[i].setW(weightTransforms.get(i).apply(layers[i].getW()));
		}
	}


	public boolean isShouldBackProp() {
		return shouldBackProp;
	}

	/**
	 * Creates a layer depending on the index.
	 * The main reason this matters is for continuous variations such as the {@link CDBN}
	 * where the first layer needs to be an {@link CRBM} for continuous inputs.
	 * 
	 * Please be sure to call super.initializeNetwork
	 * 
	 * to handle the passing of baseline parameters such as fanin
	 * and rendering.
	 * 
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

	/**
	 * Transposes this network to turn it in to 
	 * ad ecoder for the given auto encoder networkk
	 * @param network the network to decode
	 */
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


	public boolean isForceNumEpochs() {
		return forceNumEpochs;
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
		private double fanIn = -1;
		private int renderWeithsEveryNEpochs = -1;
		private double l2 = 0.01;
		private boolean useRegularization = true;
		private double momentum;
		private RealDistribution dist;
		protected Map<Integer,MatrixTransform> weightTransforms = new HashMap<Integer,MatrixTransform>();
		protected boolean backProp = true;
		protected boolean shouldForceEpochs = false;
		private double sparsity = 0;
		
		
		public Builder<E> withSparsity(double sparsity) {
			this.sparsity = sparsity;
			return this;
		}
		
		/**
		 * Forces use of number of epochs for training
		 * SGD style rather than conjugate gradient
		 * @return
		 */
		public Builder<E> forceEpochs() {
			shouldForceEpochs = true;
			return this;
		}

		/**
		 * Disables back propagation
		 * @return
		 */
		public Builder<E> disableBackProp() {
			backProp = false;
			return this;
		}

		/**
		 * Transform the weights at the given layer
		 * @param layer the layer to transform
		 * @param transform the function used for transformation
		 * @return
		 */
		public Builder<E> transformWeightsAt(int layer,MatrixTransform transform) {
			weightTransforms.put(layer,transform);
			return this;
		}

		/**
		 * A map of transformations for transforming
		 * the given layers
		 * @param transforms
		 * @return
		 */
		public Builder<E> transformWeightsAt(Map<Integer,MatrixTransform> transforms) {
			weightTransforms.putAll(transforms);
			return this;
		}

		/**
		 * Probability distribution for generating weights
		 * @param dist
		 * @return
		 */
		public Builder<E> withDist(RealDistribution dist) {
			this.dist = dist;
			return this;
		}

		/**
		 * Specify momentum
		 * @param momentum
		 * @return
		 */
		public Builder<E> withMomentum(double momentum) {
			this.momentum = momentum;
			return this;
		}

		/**
		 * Use l2 reg
		 * @param useRegularization
		 * @return
		 */
		public Builder<E> useRegularization(boolean useRegularization) {
			this.useRegularization = useRegularization;
			return this;
		}

		/**
		 * L2 coefficient
		 * @param l2
		 * @return
		 */
		public Builder<E> withL2(double l2) {
			this.l2 = l2;
			return this;
		}

		/**
		 * Whether to plot weights or not
		 * @param everyN
		 * @return
		 */
		public Builder<E> renderWeights(int everyN) {
			this.renderWeithsEveryNEpochs = everyN;
			return this;
		}

		public Builder<E> withFanIn(Double fanIn) {
			this.fanIn = fanIn;
			return this;
		}

		/**
		 * Pick an activation function, default is sigmoid
		 * @param activation
		 * @return
		 */
		public Builder<E> withActivation(ActivationFunction activation) {
			this.activation = activation;
			return this;
		}


		public Builder<E> numberOfInputs(int nIns) {
			this.nIns = nIns;
			return this;
		}

		/**
		 * Whether the network is a decoder for an auto encoder
		 * @param decode
		 * @return
		 */
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
				ret.shouldBackProp = this.backProp;
				ret.sigmoidLayers = new HiddenLayer[ret.nLayers];
				ret.layers = ret.createNetworkLayers(ret.nLayers);
				ret.toDecode = this.decode;
				ret.input = this.input;
				ret.momentum = this.momentum;
				ret.labels = this.labels;
				ret.fanIn = this.fanIn;
				ret.sparsity = this.sparsity;
				ret.renderWeightsEveryNEpochs = this.renderWeithsEveryNEpochs;
				ret.l2 = l2;
				ret.forceNumEpochs = shouldForceEpochs;
				ret.useRegularization = useRegularization;
				if(activation != null)
					ret.activation = activation;
				if(dist != null)
					ret.dist = dist;
				ret.weightTransforms.putAll(weightTransforms);


				return ret;
			} catch (InstantiationException | IllegalAccessException e) {
				throw new RuntimeException(e);
			}

		}

	}


}
