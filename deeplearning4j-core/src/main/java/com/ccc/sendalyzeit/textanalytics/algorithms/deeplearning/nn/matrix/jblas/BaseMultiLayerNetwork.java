package com.ccc.sendalyzeit.textanalytics.algorithms.deeplearning.nn.matrix.jblas;

import java.io.IOException;
import java.io.InputStream;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.OutputStream;
import java.io.Serializable;
import java.lang.reflect.Constructor;

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
	public int nIns;
	public int[] hiddenLayerSizes;
	public int nOuts;
	public int nLayers;
	public HiddenLayerMatrix[] sigmoidLayers;
	public LogisticRegressionMatrix logLayer;
	public RandomGenerator rng;
	public DoubleMatrix input,labels;
	public NeuralNetwork[] layers;


	protected BaseMultiLayerNetwork() {}

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

		this.sigmoidLayers = new HiddenLayerMatrix[n_layers];
		this.layers = createNetworkLayers(n_layers);

		if(rng == null)   
			this.rng = new MersenneTwister(123);


		else 
			this.rng = rng;  


		if(input != null) {
			initializeLayers(input);
		}

	}


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
				this.sigmoidLayers[i] = new HiddenLayerMatrix(input_size, this.hiddenLayerSizes[i], null, null, rng,layer_input);

			}
			else {
				layer_input = sigmoidLayers[i - 1].sample_h_given_v();
				// construct sigmoid_layer
				this.sigmoidLayers[i] = new HiddenLayerMatrix(input_size, this.hiddenLayerSizes[i], null, null, rng,layer_input);

			}

			// construct dA_layer
			this.layers[i] = createLayer(layer_input,input_size, this.hiddenLayerSizes[i], this.sigmoidLayers[i].W, this.sigmoidLayers[i].b, null, rng,i);
		}

		// layer for output using LogisticRegressionMatrix
		this.logLayer = new LogisticRegressionMatrix(layer_input, this.hiddenLayerSizes[this.nLayers-1], this.nOuts);

	}


	public void finetune(double lr, int epochs) {
		finetune(this.labels,lr,epochs);

	}


	public void finetune(DoubleMatrix labels,double lr, int epochs) {

		DoubleMatrix layer_input = this.sigmoidLayers[sigmoidLayers.length - 1].sample_h_given_v();

		for(int epoch = 0; epoch < epochs; epoch++) {
			logLayer.train(layer_input, labels, lr);
			lr *= 0.95;
		}


	}




	public DoubleMatrix predict(DoubleMatrix x) {
		DoubleMatrix input = x;
		for(int i = 0; i < nLayers; i++) {
			HiddenLayerMatrix layer = sigmoidLayers[i];
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


	public void load(InputStream is) {
		try {
			ObjectInputStream ois = new ObjectInputStream(is);
			BaseMultiLayerNetwork loaded = (BaseMultiLayerNetwork) ois.readObject();
			update(loaded);
		} catch (Exception e) {
			throw new RuntimeException(e);
		}

	}





	public void update(BaseMultiLayerNetwork matrix) {
		this.layers = matrix.layers;
		this.hiddenLayerSizes = matrix.hiddenLayerSizes;
		this.logLayer = matrix.logLayer;
		this.nIns = matrix.nIns;
		this.nLayers = matrix.nLayers;
		this.nOuts = matrix.nOuts;
		this.rng = matrix.rng;
		this.sigmoidLayers = matrix.sigmoidLayers;

	}



	public abstract void trainNetwork(DoubleMatrix input,DoubleMatrix labels,Object[] otherParams);

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

		public E build() {
			if(input != null && labels != null) {
				return buildWithInputsAndLabels();
			}
			else 
				return buildWithoutInputsAndLabels();

		}

		@SuppressWarnings("unchecked")
		private E buildWithoutInputsAndLabels() {
			Constructor<?>[] constructors = clazz.getDeclaredConstructors();
			for(Constructor<?> c : constructors)  {
				Class<?>[] clazzes = c.getParameterTypes();
				if(clazzes[clazzes.length - 1].isAssignableFrom(RandomGenerator.class)) {
					try {
						ret = (E) c.newInstance(nIns, hiddenLayerSizes,nOuts, nLayers,rng);
					}catch(Exception e) {
						throw new RuntimeException(e);
					}
				}
			}



			return ret;
		}

		@SuppressWarnings("unchecked")
		private E buildWithInputsAndLabels() {
			Constructor<?>[] constructors = clazz.getDeclaredConstructors();
			for(Constructor<?> c : constructors)  {
				Class<?>[] clazzes = c.getParameterTypes();
				if(clazzes[clazzes.length - 1].isAssignableFrom(DoubleMatrix.class)) {
					try {
						ret = (E) c.newInstance(nIns, hiddenLayerSizes,nOuts, nLayers,rng,input,labels);
					}catch(Exception e) {
						throw new RuntimeException(e);
					}
				}
			}


			return ret;
		}


	}


}
