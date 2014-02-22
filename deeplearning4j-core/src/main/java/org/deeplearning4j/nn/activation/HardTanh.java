package org.deeplearning4j.nn.activation;

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

	@Override
	public DoubleMatrix applyDerivative(DoubleMatrix input) {
		for(int i = 0; i < input.length; i++) {
			double val = input.get(i);
			if(val < -1 )
				val = -1;
			else if(val > 1)
				val = 1;
			else
				val = 1 - Math.pow(Math.tanh(val),2);
			input.put(i,val);
		}
		
		return input;
		
	}

}
