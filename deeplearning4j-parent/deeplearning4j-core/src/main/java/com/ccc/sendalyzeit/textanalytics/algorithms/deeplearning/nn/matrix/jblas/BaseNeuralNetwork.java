package com.ccc.sendalyzeit.textanalytics.algorithms.deeplearning.nn.matrix.jblas;

import java.io.Serializable;
import java.lang.reflect.Constructor;

import org.apache.commons.math3.distribution.UniformRealDistribution;
import org.apache.commons.math3.random.MersenneTwister;
import org.apache.commons.math3.random.RandomGenerator;
import org.jblas.DoubleMatrix;
import org.jblas.MatrixFunctions;

import com.ccc.sendalyzeit.textanalytics.util.MatrixUtil;

/**
 * Baseline class for any Neural Network used
 * as a layer in a deep network such as an {@link DBN}
 * @author Adam Gibson
 *
 */
public abstract class BaseNeuralNetwork implements NeuralNetwork,Serializable {


	private static final long serialVersionUID = -7074102204433996574L;
	/* Number of visible inputs */
	public int nVisible;
	/**
	 * Number of hidden units
	 * One tip with this is usually having
	 * more hidden units than inputs (read: input rows here)
	 * will typically cause terrible overfitting.
	 * 
	 * Another rule worthy of note: more training data typically results
	 * in more redundant data. It is usually a better idea to use a smaller number
	 * of hidden units.
	 *  
	 *  
	 *   
	 **/
	public int nHidden;
	/* Weight matrix */
	public DoubleMatrix W;
	/* hidden bias */
	public DoubleMatrix hBias;
	/* visible bias */
	public DoubleMatrix vBias;
	/* RNG for sampling. */
	public RandomGenerator rng;
	/* input to the network */
	public DoubleMatrix input;



	public BaseNeuralNetwork() {}
	/**
	 * 
	 * @param nVisible the number of outbound nodes
	 * @param nHidden the number of nodes in the hidden layer
	 * @param W the weights for this vector, maybe null, if so this will
	 * create a matrix with nHidden x nVisible dimensions.
	 * @param hBias the hidden bias
	 * @param vBias the visible bias (usually b for the output layer)
	 * @param rng the rng, if not a seed of 1234 is used.
	 */
	public BaseNeuralNetwork(int nVisible, int nHidden, 
			DoubleMatrix W, DoubleMatrix hbias, DoubleMatrix vbias, RandomGenerator rng) {
		this.nVisible = nVisible;
		this.nHidden = nHidden;

		if(rng == null)	
			this.rng = new MersenneTwister(1234);

		else 
			this.rng = rng;

		if(W == null) {
			double a = 1.0 / (double) nVisible;
			/*
			 * Initialize based on the number of visible units..
			 * The lower bound is called the fan in
			 * The outer bound is called the fan out.
			 * 
			 * Below's advice works for Denoising AutoEncoders and other 
			 * neural networks you will use due to the same baseline guiding principles for
			 * both RBMs and Denoising Autoencoders.
			 * 
			 * Hinton's Guide to practical RBMs:
			 * The weights are typically initialized to small random values chosen from a zero-mean Gaussian with
			 * a standard deviation of about 0.01. Using larger random values can speed the initial learning, but
			 * it may lead to a slightly worse final model. Care should be taken to ensure that the initial weight
			 * values do not allow typical visible vectors to drive the hidden unit probabilities very close to 1 or 0
			 * as this significantly slows the learning.
			 */
			UniformRealDistribution u = new UniformRealDistribution(rng,-a,a);

			this.W = DoubleMatrix.zeros(nVisible,nHidden);

			for(int i = 0; i < this.W.rows; i++) {
				for(int j = 0; j < this.W.columns; j++) 
					this.W.put(i,j,u.sample());

			}


		}
		else	
			this.W = W;


		if(hbias == null) 
			this.hBias = DoubleMatrix.zeros(nHidden);

		else if(hbias.length != nHidden)
			throw new IllegalArgumentException("Hidden bias must have a length of " + nHidden + " length was " + hbias.length);

		else
			this.hBias = hbias;

		if(vbias == null) 
			this.vBias = DoubleMatrix.zeros(nVisible);

		else if(vbias.length != nVisible) 
			throw new IllegalArgumentException("Visible bias must have a length of " + nVisible + " but length was " + vbias.length);

		else 
			this.vBias = vbias;
	}

	/**
	 * 
	 * @param input the input examples
	 * @param nVisible the number of outbound nodes
	 * @param nHidden the number of nodes in the hidden layer
	 * @param W the weights for this vector, maybe null, if so this will
	 * create a matrix with nHidden x nVisible dimensions.
	 * @param hBias the hidden bias
	 * @param vBias the visible bias (usually b for the output layer)
	 * @param rng the rng, if not a seed of 1234 is used.
	 */
	public BaseNeuralNetwork(DoubleMatrix input, int n_visible, int n_hidden, 
			DoubleMatrix W, DoubleMatrix hbias, DoubleMatrix vbias, RandomGenerator rng) {
		this(n_visible,n_hidden,W,hbias,vbias,rng);
		this.input = input;
	}


	public static class Builder<E extends BaseNeuralNetwork> {
		private E ret = null;
		private DoubleMatrix W;
		protected Class<? extends NeuralNetwork> clazz;
		private DoubleMatrix vBias;
		private DoubleMatrix hBias;
		private int numVisible;
		private int numHidden;
		private RandomGenerator gen;
		private DoubleMatrix input;

		public Builder<E> withInput(DoubleMatrix input) {
			this.input = input;
			return this;
		}

		public Builder<E> asType(Class<E> clazz) {
			this.clazz = clazz;
			return this;
		}


		public Builder<E> withWeights(DoubleMatrix W) {
			this.W = W;
			return this;
		}

		public Builder<E> withVisibleBias(DoubleMatrix vBias) {
			this.vBias = vBias;
			return this;
		}

		public Builder<E> withHBias(DoubleMatrix hBias) {
			this.hBias = hBias;
			return this;
		}

		public Builder<E> numberOfVisible(int numVisible) {
			this.numVisible = numVisible;
			return this;
		}

		public Builder<E> numHidden(int numHidden) {
			this.numHidden = numHidden;
			return this;
		}

		public Builder<E> withRandom(RandomGenerator gen) {
			this.gen = gen;
			return this;
		}

		public E build() {
			if(input != null) 
				return buildWithInput();
			else 
				return buildWithoutInput();
		}

		@SuppressWarnings("unchecked")
		private  E buildWithoutInput() {
			Constructor<?>[] c = clazz.getDeclaredConstructors();
			for(int i = 0; i < c.length; i++) {
				Constructor<?> curr = c[i];
				Class<?>[] classes = curr.getParameterTypes();

				//input matrix found
				if(classes.length > 0 && classes[0].isAssignableFrom(Integer.class) || classes[0].isPrimitive()) {
					try {
						ret = (E) curr.newInstance(numVisible, numHidden, 
								W, hBias,vBias, gen);
						return ret;
					}catch(Exception e) {
						throw new RuntimeException(e);
					}

				}
			}
			return ret;
		}


		@SuppressWarnings("unchecked")
		private  E buildWithInput()  {
			Constructor<?>[] c = clazz.getDeclaredConstructors();
			for(int i = 0; i < c.length; i++) {
				Constructor<?> curr = c[i];
				Class<?>[] classes = curr.getParameterTypes();
				//input matrix found
				if(classes.length > 0 && classes[0].isAssignableFrom(DoubleMatrix.class)) {
					try {
						ret = (E) curr.newInstance(numVisible, numHidden, 
								W, hBias,vBias, gen);
						return ret;
					}catch(Exception e) {
						throw new RuntimeException(e);
					}

				}
			}
			return ret;
		}
	}

	/**
	 * Reconstruction error.
	 * @return reconstruction error
	 */
	public double getReConstructionCrossEntropy() {
		DoubleMatrix preSigH = input.mmul(W).addRowVector(hBias);
		DoubleMatrix sigH = MatrixUtil.sigmoid(preSigH);

		DoubleMatrix preSigV = sigH.mmul(W.transpose()).addRowVector(vBias);
		DoubleMatrix sigV = MatrixUtil.sigmoid(preSigV);
       
		DoubleMatrix inner = 
				input.mul(MatrixFunctions.log(sigV))
				.add(MatrixUtil.oneMinus(input)
				.mul(MatrixFunctions.log(MatrixUtil.oneMinus(sigV))));
		
		return - inner.rowSums().mean();
	}


	/* (non-Javadoc)
	 * @see com.ccc.sendalyzeit.textanalytics.algorithms.deeplearning.nn.matrix.jblas.NeuralNetwork#getnVisible()
	 */
	@Override
	public int getnVisible() {
		return nVisible;
	}

	/* (non-Javadoc)
	 * @see com.ccc.sendalyzeit.textanalytics.algorithms.deeplearning.nn.matrix.jblas.NeuralNetwork#setnVisible(int)
	 */
	@Override
	public void setnVisible(int nVisible) {
		this.nVisible = nVisible;
	}

	/* (non-Javadoc)
	 * @see com.ccc.sendalyzeit.textanalytics.algorithms.deeplearning.nn.matrix.jblas.NeuralNetwork#getnHidden()
	 */
	@Override
	public int getnHidden() {
		return nHidden;
	}

	/* (non-Javadoc)
	 * @see com.ccc.sendalyzeit.textanalytics.algorithms.deeplearning.nn.matrix.jblas.NeuralNetwork#setnHidden(int)
	 */
	@Override
	public void setnHidden(int nHidden) {
		this.nHidden = nHidden;
	}

	/* (non-Javadoc)
	 * @see com.ccc.sendalyzeit.textanalytics.algorithms.deeplearning.nn.matrix.jblas.NeuralNetwork#getW()
	 */
	@Override
	public DoubleMatrix getW() {
		return W;
	}

	/* (non-Javadoc)
	 * @see com.ccc.sendalyzeit.textanalytics.algorithms.deeplearning.nn.matrix.jblas.NeuralNetwork#setW(org.jblas.DoubleMatrix)
	 */
	@Override
	public void setW(DoubleMatrix w) {
		W = w;
	}

	/* (non-Javadoc)
	 * @see com.ccc.sendalyzeit.textanalytics.algorithms.deeplearning.nn.matrix.jblas.NeuralNetwork#gethBias()
	 */
	@Override
	public DoubleMatrix gethBias() {
		return hBias;
	}

	/* (non-Javadoc)
	 * @see com.ccc.sendalyzeit.textanalytics.algorithms.deeplearning.nn.matrix.jblas.NeuralNetwork#sethBias(org.jblas.DoubleMatrix)
	 */
	@Override
	public void sethBias(DoubleMatrix hBias) {
		this.hBias = hBias;
	}

	/* (non-Javadoc)
	 * @see com.ccc.sendalyzeit.textanalytics.algorithms.deeplearning.nn.matrix.jblas.NeuralNetwork#getvBias()
	 */
	@Override
	public DoubleMatrix getvBias() {
		return vBias;
	}

	/* (non-Javadoc)
	 * @see com.ccc.sendalyzeit.textanalytics.algorithms.deeplearning.nn.matrix.jblas.NeuralNetwork#setvBias(org.jblas.DoubleMatrix)
	 */
	@Override
	public void setvBias(DoubleMatrix vBias) {
		this.vBias = vBias;
	}

	/* (non-Javadoc)
	 * @see com.ccc.sendalyzeit.textanalytics.algorithms.deeplearning.nn.matrix.jblas.NeuralNetwork#getRng()
	 */
	@Override
	public RandomGenerator getRng() {
		return rng;
	}

	/* (non-Javadoc)
	 * @see com.ccc.sendalyzeit.textanalytics.algorithms.deeplearning.nn.matrix.jblas.NeuralNetwork#setRng(org.apache.commons.math3.random.RandomGenerator)
	 */
	@Override
	public void setRng(RandomGenerator rng) {
		this.rng = rng;
	}

	/* (non-Javadoc)
	 * @see com.ccc.sendalyzeit.textanalytics.algorithms.deeplearning.nn.matrix.jblas.NeuralNetwork#getInput()
	 */
	@Override
	public DoubleMatrix getInput() {
		return input;
	}

	/* (non-Javadoc)
	 * @see com.ccc.sendalyzeit.textanalytics.algorithms.deeplearning.nn.matrix.jblas.NeuralNetwork#setInput(org.jblas.DoubleMatrix)
	 */
	@Override
	public void setInput(DoubleMatrix input) {
		this.input = input;
	}





}
