package com.ccc.deeplearning.nn.matrix.jblas.activation;

import org.jblas.DoubleMatrix;

import com.ccc.deeplearning.util.MatrixUtil;

public class Sigmoid implements ActivationFunction {

	@Override
	public DoubleMatrix apply(DoubleMatrix arg0) {
		return MatrixUtil.sigmoid(arg0);
	}

	

}
