package org.deeplearning4j.nn;

import java.io.Serializable;

import org.apache.commons.math3.distribution.RealDistribution;
import org.apache.commons.math3.random.RandomGenerator;
import org.jblas.DoubleMatrix;

public interface NeuralNetwork extends Serializable,Cloneable {

	public abstract int getnVisible();

	public abstract void setnVisible(int nVisible);

	public abstract int getnHidden();

	public abstract void setnHidden(int nHidden);

	public abstract DoubleMatrix getW();

	public abstract void setW(DoubleMatrix w);

	public abstract DoubleMatrix gethBias();

	public abstract void sethBias(DoubleMatrix hBias);

	public abstract DoubleMatrix getvBias();

	public abstract void setvBias(DoubleMatrix vBias);

	public abstract RandomGenerator getRng();

	public abstract void setRng(RandomGenerator rng);

	public abstract DoubleMatrix getInput();

	public abstract void setInput(DoubleMatrix input);
	
	
	public double squaredLoss();
	
	
	public double getSparsity();
	public abstract void setSparsity(double sparsity);
	
	public void setDist(RealDistribution dist);
	public RealDistribution getDist();
	
	
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