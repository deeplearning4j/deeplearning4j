package com.ccc.sendalyzeit.textanalytics.algorithms.deeplearning.nn.matrix.jblas;

import java.io.Serializable;
import java.lang.reflect.Constructor;

import org.apache.commons.math3.distribution.UniformRealDistribution;
import org.apache.commons.math3.random.MersenneTwister;
import org.apache.commons.math3.random.RandomGenerator;
import org.jblas.DoubleMatrix;

public abstract class BaseNeuralNetwork implements NeuralNetwork,Serializable {

	public int nVisible;
	public int nHidden;
	public DoubleMatrix W;
	public DoubleMatrix hBias;
	public DoubleMatrix vBias;
	public RandomGenerator rng;
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
	public BaseNeuralNetwork(int n_visible, int n_hidden, 
			DoubleMatrix W, DoubleMatrix hbias, DoubleMatrix vbias, RandomGenerator rng) {
		this.nVisible = n_visible;
		this.nHidden = n_hidden;

		if(rng == null)	
			this.rng = new MersenneTwister(1234);

		else 
			this.rng = rng;

		if(W == null) {
			double a = 1.0 / (double) n_visible;
			UniformRealDistribution u = new UniformRealDistribution(rng,-a,a);

			this.W = DoubleMatrix.zeros(n_visible,n_hidden);

			for(int i = 0; i < this.W.rows; i++) {
				for(int j = 0; j < this.W.columns; j++) 
					this.W.put(i,j,u.sample());

			}


		}
		else	
			this.W = W;


		if(hbias == null) 
			this.hBias = DoubleMatrix.zeros(n_hidden);

		else if(hbias.length != n_hidden)
			throw new IllegalArgumentException("Hidden bias must have a length of " + n_hidden + " length was " + hbias.length);

		else
			this.hBias = hbias;

		if(vbias == null) 
			this.vBias = DoubleMatrix.zeros(n_visible);

		else if(vbias.length != n_visible) 
			throw new IllegalArgumentException("Visible bias must have a length of " + n_visible + " but length was " + vbias.length);

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
		this.input = input;
		this.nVisible = n_visible;
		this.nHidden = n_hidden;

		if(rng == null)	
			this.rng = new MersenneTwister(1234);

		else 
			this.rng = rng;

		if(W == null) {
			double a = 1.0 / (double) n_visible;
			UniformRealDistribution u = new UniformRealDistribution(rng,-a,a);

			this.W = DoubleMatrix.zeros(n_visible,n_hidden);

			for(int i = 0; i < this.W.rows; i++) {
				for(int j = 0; j < this.W.columns; j++) 
					this.W.put(i,j,u.sample());

			}


		}
		else	
			this.W = W;


		if(hbias == null) 
			this.hBias = DoubleMatrix.zeros(n_hidden);

		else if(hbias.length != n_hidden)
			throw new IllegalArgumentException("Hidden bias must have a length of " + n_hidden + " length was " + hbias.length);

		else
			this.hBias = hbias;

		if(vbias == null) 
			this.vBias = DoubleMatrix.zeros(n_visible);

		else if(vbias.length != n_visible) 
			throw new IllegalArgumentException("Visible bias must have a length of " + n_visible + " but length was " + vbias.length);

		else 
			this.vBias = vbias;
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
