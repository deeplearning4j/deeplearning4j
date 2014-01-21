package com.ccc.deeplearning.optimize;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;

import org.jblas.DoubleMatrix;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import cc.mallet.optimize.Optimizable;
import cc.mallet.optimize.Optimizer;

import com.ccc.deeplearning.nn.BaseNeuralNetwork;
/**
 * Performs basic beam search based on the network's loss function
 * @author Adam Gibson
 *
 */
public abstract class NeuralNetworkOptimizer implements Optimizable.ByGradientValue,Serializable {

	public NeuralNetworkOptimizer(BaseNeuralNetwork network,double lr,Object[] trainingParams) {
		this.network = network;
		this.lr = lr;
		this.extraParams = trainingParams;
	}


	private static final long serialVersionUID = 4455143696487934647L;
	protected BaseNeuralNetwork network;
	protected double lr;
	protected Object[] extraParams;
	protected double tolerance = 0.000001;
	protected static Logger log = LoggerFactory.getLogger(NeuralNetworkOptimizer.class);
	protected List<Double> errors = new ArrayList<Double>();
	protected double minLearningRate = 0.001;
	protected transient Optimizer opt;

	public void train(DoubleMatrix x) {
		if(opt == null)
			opt = new com.ccc.deeplearning.util.MyConjugateGradient(this);
	        
	        try {
	            opt.optimize();
	        } catch (Throwable e) {
	            log.error("", e);
	        }

	}


	public List<Double> getErrors() {
		return errors;
	}


	@Override
	public int getNumParameters() {
		return network.W.length + network.hBias.length + network.vBias.length;
	}


	@Override
	public void getParameters(double[] buffer) {
		/*
		 * If we think of the parameters of the model (W,vB,hB)
		 * as a solid line for the optimizer, we get the following:
		 * 
		 */

		int idx = 0;
		for(int i = 0; i < network.W.length; i++)
			buffer[idx++] = network.W.get(i);
		for(int i = 0; i < network.vBias.length; i++)
			buffer[idx++] = network.vBias.get(i);
		for(int i =0; i < network.hBias.length; i++)
			buffer[idx++] = network.hBias.get(i);
	}


	@Override
	public double getParameter(int index) {
		//beyond weight matrix
		if(index >= network.W.length) {
			//beyond visible bias
			if(index >= network.vBias.length) {
				return network.hBias.get(index);
			}
			else
				return network.vBias.get(index);

		}
		return network.W.get(index);

	}


	@Override
	public void setParameters(double[] params) {
		/*
		 * If we think of the parameters of the model (W,vB,hB)
		 * as a solid line for the optimizer, we get the following:
		 * 
		 */

		int idx = 0;
		for(int i = 0; i < network.W.length; i++)
			network.W.put(i,params[idx++]);
		for(int i = 0; i < network.vBias.length; i++)
			network.vBias.put(i,params[idx++]);
		for(int i =0; i < network.hBias.length; i++)
			network.hBias.put(i,params[idx++]);

	}


	@Override
	public void setParameter(int index, double value) {
		//beyond weight matrix
		if(index >= network.W.length) {
			//beyond visible bias
			if(index >= network.vBias.length) {
				int i = index - network.hBias.length;
				network.hBias.put(i, value);
			}
			else {
				int i = index - network.vBias.length;
				network.vBias.put(i,value);

			}

		}
		network.W.put(index, value);
	}


	@Override
	public abstract void getValueGradient(double[] buffer);


	@Override
	public double getValue() {
		return network.squaredLoss();
	}



}
