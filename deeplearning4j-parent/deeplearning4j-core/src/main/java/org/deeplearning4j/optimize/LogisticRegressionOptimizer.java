package org.deeplearning4j.optimize;

import java.io.Serializable;

import org.deeplearning4j.nn.LogisticRegression;
import org.deeplearning4j.nn.LogisticRegressionGradient;


import cc.mallet.optimize.Optimizable;

public class LogisticRegressionOptimizer implements Optimizable.ByGradientValue,Serializable {

	/**
	 * 
	 */
	private static final long serialVersionUID = 5229426347154854746L;
	private LogisticRegression logReg;
	private double lr;
	
	
	
	public LogisticRegressionOptimizer(LogisticRegression logReg, double lr) {
		super();
		this.logReg = logReg;
		this.lr = lr;
	}

	@Override
	public int getNumParameters() {
		return logReg.getW().length + logReg.getB().length;
		
	}

	@Override
	public void getParameters(double[] buffer) {
		for(int i = 0; i < buffer.length; i++) {
			buffer[i] = getParameter(i);
		}

		


	}

	@Override
	public double getParameter(int index) {
		if(index >= logReg.getW().length)
			return logReg.getB().get(index - logReg.getW().length);
		return logReg.getW().get(index);
	}

	@Override
	public void setParameters(double[] params) {
		for(int i = 0; i < params.length; i++) {
			setParameter(i,params[i]);
		}
	}

	@Override
	public void setParameter(int index, double value) {
		if(index >= logReg.getW().length)
			logReg.getB().put(index - logReg.getW().length,value);
		else
			logReg.getW().put(index,value);
	}

	@Override
	public void getValueGradient(double[] buffer) {
		LogisticRegressionGradient grad = logReg.getGradient(lr);
		for(int i = 0; i < buffer.length; i++) {
			if(i < logReg.getW().length)
				buffer[i] = grad.getwGradient().get(i);
			else
				buffer[i] = grad.getbGradient().get(i - logReg.getW().length);
			
		}
	}

	@Override
	public double getValue() {
		return -logReg.negativeLogLikelihood();
	}



}
