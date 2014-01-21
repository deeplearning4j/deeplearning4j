package com.ccc.deeplearning.optimize;

import java.io.Serializable;

import com.ccc.deeplearning.nn.LogisticRegression;
import com.ccc.deeplearning.nn.LogisticRegressionGradient;

import cc.mallet.optimize.Optimizable;

public class LogisticRegressionOptimizer implements Optimizable.ByGradientValue,Serializable {

	private LogisticRegression logReg;
	private double lr;
	
	
	
	public LogisticRegressionOptimizer(LogisticRegression logReg, double lr) {
		super();
		this.logReg = logReg;
		this.lr = lr;
	}

	@Override
	public int getNumParameters() {
		return logReg.W.length + logReg.b.length;
	}

	@Override
	public void getParameters(double[] buffer) {
		for(int i = 0; i < buffer.length; i++) {
			buffer[i] = getParameter(i);
		}

		


	}

	@Override
	public double getParameter(int index) {
		if(index >= logReg.W.length)
			return logReg.b.get(index - logReg.W.length);
		return logReg.W.get(index);
	}

	@Override
	public void setParameters(double[] params) {
		for(int i = 0; i < params.length; i++) {
			setParameter(i,params[i]);
		}
	}

	@Override
	public void setParameter(int index, double value) {
		if(index >= logReg.W.length)
			logReg.b.put(index - logReg.W.length,value);
		else
			logReg.W.put(index,value);
	}

	@Override
	public void getValueGradient(double[] buffer) {
		LogisticRegressionGradient grad = logReg.getGradient(lr);
		for(int i = 0; i < buffer.length; i++) {
			if(i < logReg.W.length)
				buffer[i] = grad.getwGradient().get(i);
			else
				buffer[i] = grad.getbGradient().get(i - logReg.W.length);
		}
	}

	@Override
	public double getValue() {
		return logReg.negativeLogLikelihood();
	}



}
