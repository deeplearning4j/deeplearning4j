package com.ccc.deeplearning.transformation;

import org.jblas.DoubleMatrix;
import org.jblas.MatrixFunctions;

public class ExpTransform implements MatrixTransform {

	/**
	 * 
	 */
	private static final long serialVersionUID = 5544429281399904369L;

	@Override
	public DoubleMatrix apply(DoubleMatrix input) {
		return MatrixFunctions.exp(input);
	}

	

}
