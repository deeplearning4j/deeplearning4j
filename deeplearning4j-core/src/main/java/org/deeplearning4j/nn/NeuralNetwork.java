package org.deeplearning4j.nn;

import java.io.Serializable;
import java.util.List;

import org.apache.commons.math3.distribution.RealDistribution;
import org.apache.commons.math3.random.RandomGenerator;
import org.deeplearning4j.gradient.NeuralNetworkGradientListener;
import org.deeplearning4j.nn.gradient.NeuralNetworkGradient;
import org.deeplearning4j.nn.learning.AdaGrad;
import org.deeplearning4j.optimize.NeuralNetEpochListener;
import org.jblas.DoubleMatrix;

public interface NeuralNetwork extends Serializable,Cloneable,NeuralNetEpochListener {

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
	
	
	public double getSparsity();
	public  void setSparsity(double sparsity);
	
	public void setDist(RealDistribution dist);
	public RealDistribution getDist();
	
	
	List<NeuralNetworkGradientListener> getGradientListeners();
	void setGradientListeners(List<NeuralNetworkGradientListener> gradientListeners);
	
	
	DoubleMatrix hBiasMean();
	
	public AdaGrad getAdaGrad();
	public void setAdaGrad(AdaGrad adaGrad);
	
	public NeuralNetworkGradient getGradient(Object[] params);
	
	public double getL2();
	public void setL2(double l2);
	
	public double getMomentum();
	public void setMomentum(double momentum);
	
	public void setRenderEpochs(int renderEpochs);
	public int getRenderEpochs();

	public NeuralNetwork transpose();
	public  NeuralNetwork clone();

	public double fanIn();
	public void setFanIn(double fanIn);
	
	
	void resetAdaGrad(double lr);
	
	void epochDone(int epoch);
	
	public double l2RegularizedCoefficient();
	
	public double getReConstructionCrossEntropy();
	
	public void train(DoubleMatrix input,double lr,Object[] params);
	
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