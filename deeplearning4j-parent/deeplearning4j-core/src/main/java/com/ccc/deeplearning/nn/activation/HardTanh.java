package com.ccc.deeplearning.nn.activation;

import org.jblas.DoubleMatrix;

public class HardTanh implements ActivationFunction {

	/**
	 * 
	 */
	private static final long serialVersionUID = -8484119406683594852L;

	@Override
	public DoubleMatrix apply(DoubleMatrix matrix) {
		for(int i = 0; i < matrix.length; i++) {
			double val = matrix.get(i);
			if(val < -1 )
				val = -1;
			else if(val > 1)
				val = 1;
			else
				val = Math.tanh(val);
			matrix.put(i,val);
		}
		
		return matrix;
	}

	

}
