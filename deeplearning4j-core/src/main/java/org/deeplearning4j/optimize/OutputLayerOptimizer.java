package org.deeplearning4j.optimize;

import org.deeplearning4j.nn.OutputLayer;
import org.deeplearning4j.nn.gradient.OutputLayerGradient;
import org.jblas.DoubleMatrix;

import cc.mallet.optimize.Optimizable;

/**
 * Output layer optimizer
 * @author Adam Gibson
 */
public class OutputLayerOptimizer implements Optimizable.ByGradientValue,OptimizableByGradientValueMatrix {

	private OutputLayer logReg;
	private double lr;
    private int currIteration = -1;
	
	
	
	public OutputLayerOptimizer(OutputLayer logReg, double lr) {
		super();
		this.logReg = logReg;
		this.lr = lr;
	}

    @Override
    public void setCurrentIteration(int value) {
        this.currIteration = value;
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
		OutputLayerGradient grad = logReg.getGradient(lr);
		for(int i = 0; i < buffer.length; i++) {
			if(i < logReg.getW().length)
				buffer[i] = grad.getwGradient().get(i);
			else
				buffer[i] = grad.getbGradient().get(i - logReg.getW().length);
			
		}
	}

	@Override
	public double getValue() {
		return -logReg.score();
	}

	@Override
	public DoubleMatrix getParameters() {
		DoubleMatrix params = new DoubleMatrix(getNumParameters());
		for(int i = 0; i < params.length; i++) {
			params.put(i,getParameter(i));
		}
		return params;
	}

	@Override
	public void setParameters(DoubleMatrix params) {
		this.setParameters(params.toArray());
	}

	@Override
	public DoubleMatrix getValueGradient(int currIteration) {
        this.currIteration = currIteration;
		OutputLayerGradient grad = logReg.getGradient(lr);
		DoubleMatrix ret = new DoubleMatrix(getNumParameters());
        if(logReg.getW().length != grad.getwGradient().length)
            throw new IllegalStateException("Illegal length for gradient");
        if(logReg.getB().length != grad.getbGradient().length)
            throw new IllegalStateException("Illegal length for gradient");



        for(int i = 0; i < ret.length; i++) {
			if(i < logReg.getW().length)
				ret.put(i,grad.getwGradient().get(i));
			else
				ret.put(i,grad.getbGradient().get(i - logReg.getW().length));
			
		}
		return ret;
	}



}
