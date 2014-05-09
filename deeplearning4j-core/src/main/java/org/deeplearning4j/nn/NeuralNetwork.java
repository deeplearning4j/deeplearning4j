package org.deeplearning4j.nn;

import java.io.Serializable;
import java.util.List;

import org.apache.commons.math3.distribution.RealDistribution;
import org.apache.commons.math3.random.RandomGenerator;
import org.deeplearning4j.berkeley.Pair;
import org.deeplearning4j.gradient.NeuralNetworkGradientListener;
import org.deeplearning4j.nn.gradient.NeuralNetworkGradient;
import org.deeplearning4j.nn.learning.AdaGrad;
import org.deeplearning4j.optimize.NeuralNetEpochListener;
import org.jblas.DoubleMatrix;
/**
 * Single layer neural network, this is typically one that has 
 * the objective function of reconstruction the input: also called feature detectors
 * @author Adam Gibson
 *
 */
public interface NeuralNetwork extends Serializable,Cloneable,NeuralNetEpochListener {

	
	
	/**
	 * Optimization algorithm to use
	 * @author Adam Gibson
	 *
	 */
	public static enum OptimizationAlgorithm {
		GRADIENT_DESCENT,CONJUGATE_GRADIENT
	}
	/**
	 * Which loss function to use
	 * @author Adam Gibson
	 *
	 */
	public static enum LossFunction {
		SQUARED_LOSS,RECONSTRUCTION_CROSSENTROPY,NEGATIVELOGLIKELIHOOD,MSE
	}

    /**
     * Clears the input from the neural net
     */
    public void clearInput();

    /**
     * Whether to cache the input at training time
     * @return true if the input should be cached at training time otherwise false
     */
    public boolean cacheInput();

    /**
     * Backprop with the output being the reconstruction
     * @param lr the learning rate to use
     * @param epochs the max number of epochs to run
     * @param extraParams implementation specific params
     */
    public void backProp(double lr,int epochs,Object[] extraParams);

    /**
     * The loss function for this network
     * @return the loss function for this network
     */
	public LossFunction getLossFunction();
	public void setLossFunction(LossFunction lossFunction);

    /**
     * The optimization algorithm.
     * SGD and CG are currently supported
     * @return the optimization algorithm for this network
     */
	public OptimizationAlgorithm getOptimizationAlgorithm();
	public void setOptimizationAlgorithm(OptimizationAlgorithm optimziationAlgorithm);
	
	public boolean normalizeByInputRows();
	
	public  int getnVisible();

	public  void setnVisible(int nVisible);

	public  int getnHidden();

	public  void setnHidden(int nHidden);

	public  DoubleMatrix getW();

	public  void setW(DoubleMatrix w);

	public  DoubleMatrix gethBias();

	public  void sethBias(DoubleMatrix hBias);

	public  DoubleMatrix getvBias();

	public  void setvBias(DoubleMatrix vBias);

	public  RandomGenerator getRng();

	public  void setRng(RandomGenerator rng);

	public  DoubleMatrix getInput();

	public  void setInput(DoubleMatrix input);
	
	
	public double squaredLoss();
	
	public double negativeLogLikelihood();
	
	public double getSparsity();
	public  void setSparsity(double sparsity);
	
	public void setDist(RealDistribution dist);
	public RealDistribution getDist();
	
	
	List<NeuralNetworkGradientListener> getGradientListeners();
	void setGradientListeners(List<NeuralNetworkGradientListener> gradientListeners);
	
	
	DoubleMatrix hBiasMean();
	
	public AdaGrad getAdaGrad();
	public void setAdaGrad(AdaGrad adaGrad);
	

    public boolean isUseRegularization();

	public AdaGrad gethBiasAdaGrad();
	public void setHbiasAdaGrad(AdaGrad adaGrad);
	
	
	public AdaGrad getVBiasAdaGrad();
	public void setVBiasAdaGrad(AdaGrad adaGrad);
	
	
	public NeuralNetworkGradient getGradient(Object[] params);
	
	public double getL2();
	public void setL2(double l2);

    /**
     * Mean squared error. Used for regression tasks
     * @return the mean squared error with respect to the output
     * of the network
     */
    public double mse();

	public double getMomentum();
	public void setMomentum(double momentum);
	
	public void setRenderEpochs(int renderEpochs);
	public int getRenderEpochs();

	public NeuralNetwork transpose();
	public  NeuralNetwork clone();

	public double fanIn();
	public void setFanIn(double fanIn);
	/**
	 * Sample hidden mean and sample
	 * given visible
	 * @param v the  the visible input
	 * @return a pair with mean, sample
	 */
	public Pair<DoubleMatrix,DoubleMatrix> sampleHiddenGivenVisible(DoubleMatrix v);


    public boolean isUseAdaGrad();
	
	public void setDropOut(double dropOut);
	public double dropOut();
	
	/**
	 * Sample visible mean and sample
	 * given hidden
	 * @param h the  the hidden input
	 * @return a pair with mean, sample
	 */
	public Pair<DoubleMatrix,DoubleMatrix> sampleVisibleGivenHidden(DoubleMatrix h);
	
	void resetAdaGrad(double lr);
	
	void epochDone(int epoch);
	
	public double l2RegularizedCoefficient();

    /**
     * Error on reconstruction
     * @return the error on reconstruction
     */
	public double getReConstructionCrossEntropy();

    /**
     * Run one iteration
     * @param input the input to train on
     * @param lr the learning rate to use
     * @param params the extra params for the neural network(k, corruption level, max epochs,...)
     */
	public void train(DoubleMatrix input,double lr,Object[] params);

    /**
     * Trains via an optimization algorithm such as SGD or Conjugate Gradient
     * @param input the input to train on
     * @param lr the learning rate to use
     * @param params the params (k,corruption level, max epochs,...)
     */
	public void trainTillConvergence(DoubleMatrix input,double lr,Object[] params);
	/**
	 * Performs a network merge in the form of
	 * a += b - a / n
	 * where a is a matrix here
	 * b is a matrix on the incoming network
	 * and n is the batch size
	 * @param network the network to merge with
	 * @param batchSize the batch size (number of training examples)
	 * to average by
	 */
	void merge(NeuralNetwork network,int batchSize);
	
	
	


}