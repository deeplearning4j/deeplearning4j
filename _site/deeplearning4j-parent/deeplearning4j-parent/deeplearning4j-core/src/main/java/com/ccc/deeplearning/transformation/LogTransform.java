package com.ccc.deeplearning.transformation;

import org.jblas.DoubleMatrix;
import org.jblas.MatrixFunctions;

public class LogTransform implements MatrixTransform {

	/**
	 * 
	 */
	private static final long serialVersionUID = 8144081928477158772L;

	@Override
	public DoubleMatrix apply(DoubleMatrix input) {
		return MatrixFunctions.log(input);
	}

	

}
