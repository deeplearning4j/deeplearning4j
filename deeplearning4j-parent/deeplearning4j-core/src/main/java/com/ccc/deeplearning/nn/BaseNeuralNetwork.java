package com.ccc.deeplearning.nn;

import static com.ccc.deeplearning.util.MatrixUtil.binomial;
import static com.ccc.deeplearning.util.MatrixUtil.log;
import static com.ccc.deeplearning.util.MatrixUtil.oneMinus;
import static com.ccc.deeplearning.util.MatrixUtil.sigmoid;

import java.io.IOException;
import java.io.InputStream;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.OutputStream;
import java.lang.reflect.Constructor;

import org.apache.commons.math3.distribution.NormalDistribution;
import org.apache.commons.math3.random.MersenneTwister;
import org.apache.commons.math3.random.RandomGenerator;
import org.jblas.DoubleMatrix;
import org.jblas.MatrixFunctions;

import com.ccc.deeplearning.berkeley.Counter;
import com.ccc.deeplearning.dbn.DBN;
import com.ccc.deeplearning.optimize.NeuralNetworkOptimizer;

/**
 * Baseline class for any Neural Network used
 * as a layer in a deep network such as an {@link DBN}
 * @author Adam Gibson
 *
 */
public abstract class BaseNeuralNetwork implements NeuralNetwork,Persistable {

	


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
	/* sparsity target */
	public double sparsity = 0.01;
	/* momentum for learning */
	public double momentum = 0.1;
	/* L2 Regularization constant */
	public double l2 = 0.1;
	public transient NeuralNetworkOptimizer optimizer;
	public int renderWeightsEveryNumEpochs = -1;
	public double fanIn = -1;
	public boolean useRegularization = true;
	
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
			DoubleMatrix W, DoubleMatrix hbias, DoubleMatrix vbias, RandomGenerator rng,double fanIn) {
		this(null,nVisible,nHidden,W,hbias,vbias,rng,fanIn);

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
	public BaseNeuralNetwork(DoubleMatrix input, int nVisible, int nHidden, 
			DoubleMatrix W, DoubleMatrix hbias, DoubleMatrix vbias, RandomGenerator rng,double fanIn) {
		this.nVisible = nVisible;
		this.nHidden = nHidden;
		this.fanIn = fanIn;
		this.input = input;
		if(rng == null)	
			this.rng = new MersenneTwister(1234);

		else 
			this.rng = rng;
		this.W = W;
		this.vBias = vbias;
		this.hBias = hbias;

		initWeights();	


	}



	@Override
	public double l2RegularizedCoefficient() {
		return (MatrixFunctions.pow(getW(),2).sum()/ 2.0)  * l2;
	}
	protected void initWeights()  {
		
		if(this.nVisible < 1)
			throw new IllegalStateException("Number of visible can not be less than 1");
		if(this.nHidden < 1)
			throw new IllegalStateException("Number of hidden can not be less than 1");
		
		
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
		if(this.W == null) {
			NormalDistribution u = new NormalDistribution(rng,0,.01,0.99);

			this.W = DoubleMatrix.zeros(nVisible,nHidden);

			for(int i = 0; i < this.W.rows; i++) 
				this.W.putRow(i,new DoubleMatrix(u.sample(this.W.columns)));

		}

		if(this.hBias == null) {
			this.hBias = DoubleMatrix.zeros(nHidden);
			/*
			 * Encourage sparsity.
			 * See Hinton's Practical guide to RBMs
			 */
			//this.hBias.subi(4);
		}

		if(this.vBias == null) {
			if(this.input != null) {
		
				this.vBias = DoubleMatrix.zeros(nVisible);


			}
			else
				this.vBias = DoubleMatrix.zeros(nVisible);
		}



	}


	@Override
	public void setRenderEpochs(int renderEpochs) {
		this.renderWeightsEveryNumEpochs = renderEpochs;

	}
	@Override
	public int getRenderEpochs() {
		return renderWeightsEveryNumEpochs;
	}

	@Override
	public double fanIn() {
		return fanIn < 0 ? 1 / nVisible : fanIn;
	}

	@Override
	public void setFanIn(double fanIn) {
		this.fanIn = fanIn;
	}

	public void regularize() {
		this.W.addi(W.mul(0.01));
		this.W.divi(this.momentum);

	}

	public void jostleWeighMatrix() {
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
		NormalDistribution u = new NormalDistribution(rng,0,.01,fanIn());

		DoubleMatrix W = DoubleMatrix.zeros(nVisible,nHidden);
		for(int i = 0; i < this.W.rows; i++) 
			W.putRow(i,new DoubleMatrix(u.sample(this.W.columns)));




	}

	@Override
	public NeuralNetwork transpose() {
		try {
			NeuralNetwork ret = getClass().newInstance();
			ret.sethBias(hBias.dup());
			ret.setvBias(vBias.dup());
			ret.setnHidden(getnVisible());
			ret.setnVisible(getnHidden());
			ret.setW(W.transpose());
			ret.setRng(getRng());

			return ret;
		} catch (Exception e) {
			throw new RuntimeException(e);
		} 


	}

	@Override
	public NeuralNetwork clone() {
		try {
			NeuralNetwork ret = getClass().newInstance();
			ret.sethBias(hBias.dup());
			ret.setvBias(vBias.dup());
			ret.setnHidden(getnHidden());
			ret.setnVisible(getnVisible());
			ret.setW(W.dup());
			ret.setRng(getRng());

			return ret;
		} catch (Exception e) {
			throw new RuntimeException(e);
		} 


	}

	@Override
	public void merge(NeuralNetwork network,int batchSize) {
		W.addi(network.getW().mini(W).div(batchSize));
		hBias.addi(network.gethBias().subi(hBias).divi(batchSize));
		vBias.addi(network.getvBias().subi(vBias).divi(batchSize));
	}


	/**
	 * Regularize weights or weight averaging.
	 * This accounts for momentum, sparsity target,
	 * and batch size
	 * @param batchSize the batch size of the recent training set
	 * @param lr the learning rate
	 */
	public void regularizeWeights(int batchSize,double lr) {
		if(batchSize < 1)
			throw new IllegalArgumentException("Batch size must be at least 1");
		this.W = W.div(batchSize).mul(1 - momentum).add(W.min(W.mul(l2)));
	}



	/**
	 * Copies params from the passed in network
	 * to this one
	 * @param n the network to copy
	 */
	public void update(BaseNeuralNetwork n) {
		this.W = n.W;
		this.hBias = n.hBias;
		this.vBias = n.vBias;
		this.l2 = n.l2;
		this.momentum = n.momentum;
		this.nHidden = n.nHidden;
		this.nVisible = n.nVisible;
		this.rng = n.rng;
		this.sparsity = n.sparsity;
	}

	/**
	 * Load (using {@link ObjectInputStream}
	 * @param is the input stream to load from (usually a file)
	 */
	public void load(InputStream is) {
		try {
			ObjectInputStream ois = new ObjectInputStream(is);
			BaseNeuralNetwork loaded = (BaseNeuralNetwork) ois.readObject();
			update(loaded);
		} catch (Exception e) {
			throw new RuntimeException(e);
		}

	}


	/**
	 * Reconstruction error.
	 * @return reconstruction error
	 */
	public double getReConstructionCrossEntropy() {
		DoubleMatrix preSigH = input.mmul(W).addRowVector(hBias);
		DoubleMatrix sigH = sigmoid(preSigH);

		DoubleMatrix preSigV = sigH.mmul(W.transpose()).addRowVector(vBias);
		DoubleMatrix sigV = sigmoid(preSigV);
		DoubleMatrix inner = 
				input.mul(log(sigV))
				.add(oneMinus(input)
						.mul(log(oneMinus(sigV))));
		double l = inner.length;
		if(this.useRegularization) {
			double normalized = l + l2RegularizedCoefficient();
			return - inner.rowSums().mean() / normalized;
		}
		
		return - inner.rowSums().mean();
	}


	/* (non-Javadoc)
	 * @see com.ccc.deeplearning.nn.NeuralNetwork#getnVisible()
	 */
	@Override
	public int getnVisible() {
		return nVisible;
	}

	/* (non-Javadoc)
	 * @see com.ccc.deeplearning.nn.NeuralNetwork#setnVisible(int)
	 */
	@Override
	public void setnVisible(int nVisible) {
		this.nVisible = nVisible;
	}

	/* (non-Javadoc)
	 * @see com.ccc.deeplearning.nn.NeuralNetwork#getnHidden()
	 */
	@Override
	public int getnHidden() {
		return nHidden;
	}

	/* (non-Javadoc)
	 * @see com.ccc.deeplearning.nn.NeuralNetwork#setnHidden(int)
	 */
	@Override
	public void setnHidden(int nHidden) {
		this.nHidden = nHidden;
	}

	/* (non-Javadoc)
	 * @see com.ccc.deeplearning.nn.NeuralNetwork#getW()
	 */
	@Override
	public DoubleMatrix getW() {
		return W;
	}

	/* (non-Javadoc)
	 * @see com.ccc.deeplearning.nn.NeuralNetwork#setW(org.jblas.DoubleMatrix)
	 */
	@Override
	public void setW(DoubleMatrix w) {
		W = w;
	}

	/* (non-Javadoc)
	 * @see com.ccc.deeplearning.nn.NeuralNetwork#gethBias()
	 */
	@Override
	public DoubleMatrix gethBias() {
		return hBias;
	}

	/* (non-Javadoc)
	 * @see com.ccc.deeplearning.nn.NeuralNetwork#sethBias(org.jblas.DoubleMatrix)
	 */
	@Override
	public void sethBias(DoubleMatrix hBias) {
		this.hBias = hBias;
	}

	/* (non-Javadoc)
	 * @see com.ccc.deeplearning.nn.NeuralNetwork#getvBias()
	 */
	@Override
	public DoubleMatrix getvBias() {
		return vBias;
	}

	/* (non-Javadoc)
	 * @see com.ccc.deeplearning.nn.NeuralNetwork#setvBias(org.jblas.DoubleMatrix)
	 */
	@Override
	public void setvBias(DoubleMatrix vBias) {
		this.vBias = vBias;
	}

	/* (non-Javadoc)
	 * @see com.ccc.deeplearning.nn.NeuralNetwork#getRng()
	 */
	@Override
	public RandomGenerator getRng() {
		return rng;
	}

	/* (non-Javadoc)
	 * @see com.ccc.deeplearning.nn.NeuralNetwork#setRng(org.apache.commons.math3.random.RandomGenerator)
	 */
	@Override
	public void setRng(RandomGenerator rng) {
		this.rng = rng;
	}

	/* (non-Javadoc)
	 * @see com.ccc.deeplearning.nn.NeuralNetwork#getInput()
	 */
	@Override
	public DoubleMatrix getInput() {
		return input;
	}

	/* (non-Javadoc)
	 * @see com.ccc.deeplearning.nn.NeuralNetwork#setInput(org.jblas.DoubleMatrix)
	 */
	@Override
	public void setInput(DoubleMatrix input) {
		this.input = input;
	}


	public double getSparsity() {
		return sparsity;
	}
	public void setSparsity(double sparsity) {
		this.sparsity = sparsity;
	}
	public double getMomentum() {
		return momentum;
	}
	public void setMomentum(double momentum) {
		this.momentum = momentum;
	}
	public double getL2() {
		return l2;
	}
	public void setL2(double l2) {
		this.l2 = l2;
	}

	/**
	 * Write this to an object output stream
	 * @param os the output stream to write to
	 */
	public void write(OutputStream os) {
		try {
			ObjectOutputStream os2 = new ObjectOutputStream(os);
			os2.writeObject(this);
		} catch (IOException e) {
			throw new RuntimeException(e);
		}
	}

	/**
	 * All neural networks are based on this idea of 
	 * minimizing reconstruction error.
	 * Both RBMs and Denoising AutoEncoders
	 * have a component for reconstructing, ala different implementations.
	 *  
	 * @param x the input to reconstruct
	 * @return the reconstructed input
	 */
	public abstract DoubleMatrix reconstruct(DoubleMatrix x);

	/**
	 * The loss function (cross entropy, reconstruction error,...)
	 * @return the loss function
	 */
	public abstract double lossFunction(Object[] params);

	
	public double lossFunction() {
		return lossFunction(null);
	}

	/**
	 * Train one iteration of the network
	 * @param input the input to train on
	 * @param lr the learning rate to train at
	 * @param params the extra params (k, corruption level,...)
	 */
	@Override
	public abstract void train(DoubleMatrix input,double lr,Object[] params);

	@Override
	public double squaredLoss() {
		DoubleMatrix reconstructed = reconstruct(input);
		double loss = MatrixFunctions.powi(reconstructed.sub(input), 2).sum() / input.rows;
		if(this.useRegularization) {
			loss += 0.5 * l2 * MatrixFunctions.pow(W,2).sum();
		}
		
		return -loss;
	}



	public static class Builder<E extends BaseNeuralNetwork> {
		private E ret = null;
		private DoubleMatrix W;
		protected Class<? extends NeuralNetwork> clazz;
		private DoubleMatrix vBias;
		private DoubleMatrix hBias;
		private int numVisible;
		private int numHidden;
		private RandomGenerator gen = new MersenneTwister(123);
		private DoubleMatrix input;
		private double sparsity = 0.01;
		private double l2 = 0.01;
		private double momentum = 0.1;
		private int renderWeightsEveryNumEpochs = -1;
		private double fanIn = 0.1;
		private boolean useRegularization = true;
		
		
		public Builder<E> useRegularization(boolean useRegularization) {
			this.useRegularization = useRegularization;
			return this;
		}

		public Builder<E> fanIn(double fanIn) {
			this.fanIn = fanIn;
			return this;
		}

		public Builder<E> withL2(double l2) {
			this.l2 = l2;
			return this;
		}


		public Builder<E> renderWeights(int numEpochs) {
			this.renderWeightsEveryNumEpochs = numEpochs;
			return this;
		}

		@SuppressWarnings("unchecked")
		public E buildEmpty() {
			try {
				return (E) clazz.newInstance();
			} catch (InstantiationException | IllegalAccessException e) {
				throw new RuntimeException(e);
			}
		}



		public Builder<E> withClazz(Class<? extends BaseNeuralNetwork> clazz) {
			this.clazz = clazz;
			return this;
		}

		public Builder<E> withSparsity(double sparsity) {
			this.sparsity = sparsity;
			return this;
		}
		public Builder<E> withMomentum(double momentum) {
			this.momentum = momentum;
			return this;
		}

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
								W, hBias,vBias, gen,fanIn);
						ret.renderWeightsEveryNumEpochs = this.renderWeightsEveryNumEpochs;
						ret.useRegularization = this.useRegularization;
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
						ret = (E) curr.newInstance(input,numVisible, numHidden, W, hBias,vBias, gen,fanIn);
						ret.sparsity = this.sparsity;
						ret.renderWeightsEveryNumEpochs = this.renderWeightsEveryNumEpochs;
						ret.l2 = this.l2;
						ret.momentum = this.momentum;
						ret.useRegularization = this.useRegularization;
						return ret;
					}catch(Exception e) {
						throw new RuntimeException(e);
					}

				}
			}
			return ret;
		}
	}

}
