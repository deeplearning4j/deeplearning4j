package com.ccc.deeplearning.nn.activation;

import static com.ccc.deeplearning.util.MatrixUtil.*;

import org.jblas.DoubleMatrix;
import static org.jblas.MatrixFunctions.*;

public class Tanh implements ActivationFunction {

	/**
	 * 
	 */
	private static final long serialVersionUID = 4499150153988528321L;

	@Override
	public DoubleMatrix apply(DoubleMatrix arg0) {
		return tanh(arg0);
	}

	@Override
	public DoubleMatrix applyDerivative(DoubleMatrix input) {
		//1 - tanh^2 x
		return oneMinus(pow(tanh(input),2));
	}

	

}
