package com.ccc.deeplearning.nn.activation;

import org.jblas.DoubleMatrix;
import org.jblas.MatrixFunctions;

public class Tanh implements ActivationFunction {

	/**
	 * 
	 */
	private static final long serialVersionUID = 4499150153988528321L;

	@Override
	public DoubleMatrix apply(DoubleMatrix arg0) {
		return MatrixFunctions.tanh(arg0);
	}

	

}
