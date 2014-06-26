package org.deeplearning4j.nn;

import java.io.Serializable;
import java.util.Map;

import org.apache.commons.math3.distribution.RealDistribution;
import org.apache.commons.math3.random.RandomGenerator;
import org.deeplearning4j.berkeley.Pair;
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
		GRADIENT_DESCENT,CONJUGATE_GRADIENT,HESSIAN_FREE
	}
	/**
	 * Which loss function to use
	 * @author Adam Gibson
	 *
	 */
	public static enum LossFunction {
		SQUARED_LOSS,RECONSTRUCTION_CROSSENTROPY,NEGATIVELOGLIKELIHOOD,MSE,RMSE_XENT
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
	
	public WeightInit getWeightInit();
    public void setWeightInit(WeightInit weightInit);
	public double squaredLoss();
	
	public double negativeLogLikelihood();
	
	public double getSparsity();
	public  void setSparsity(double sparsity);
	
	public void setDist(RealDistribution dist);
	public RealDistribution getDist();
	

    public DoubleMatrix reconstruct(DoubleMatrix input);

	
	DoubleMatrix hBiasMean();
	
	public AdaGrad getAdaGrad();
	public void setAdaGrad(AdaGrad adaGrad);
	

    public void setUseRegularization(boolean useRegularization);
    public boolean isUseRegularization();

	public AdaGrad gethBiasAdaGrad();
	public void setHbiasAdaGrad(AdaGrad adaGrad);
	
	
	public AdaGrad getVBiasAdaGrad();
	public void setVBiasAdaGrad(AdaGrad adaGrad);
	
	
	public NeuralNetworkGradient getGradient(Object[] params);


    public void setConstrainGradientToUnitNorm(boolean constrainGradientToUnitNorm);
    /**
     * Whether to constrain the gradient to unit norm or not
     */
    public boolean isConstrainGradientToUnitNorm();

    /**
     * L2 regularization coefficient
     * @return
     */
	public double getL2();
	public void setL2(double l2);

    /**
     * Mean squared error. Used for regression tasks
     * @return the mean squared error with respect to the output
     * of the network
     */
    public double mse();

    /**
     * RMSE for reconstruction entropy
     * @return rmse for reconstruction entropy
     */
    public double mseRecon();

	public double getMomentum();
	public void setMomentum(double momentum);
	
	public void setRenderEpochs(int renderEpochs);
	public int getRenderIterations();

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
	
	void iterationDone(int epoch);
	
	public double l2RegularizedCoefficient();



    public void setConcatBiases(boolean concatBiases);
    boolean isConcatBiases();

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

    /**
     * Whether to apply sparsity or not
     * @return
     */
    boolean isApplySparsity();

    /**
     * Reset ada grad after n iterations
     * @return
     */
    public int getResetAdaGradIterations();

    /**
     * Reset adagrad after n iterations
     * @param resetAdaGradEpochs
     */
    public void setResetAdaGradIterations(int resetAdaGradEpochs);

    /**
     * Getter for momentum after n iterations
     * @return
     */
    public Map<Integer, Double> getMomentumAfter();

    /**
     * Setter for momentum after n iterations
     * @param momentumAfter
     */
    public void setMomentumAfter(Map<Integer, Double> momentumAfter);

}