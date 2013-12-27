package com.ccc.sendalyzeit.textanalytics.algorithms.deeplearning.nn.matrix.jblas.activation;

import org.jblas.DoubleMatrix;
import org.jblas.MatrixFunctions;

public class Tanh implements ActivationFunction {

	@Override
	public DoubleMatrix apply(DoubleMatrix arg0) {
		return MatrixFunctions.tanh(arg0);
	}

	

}
