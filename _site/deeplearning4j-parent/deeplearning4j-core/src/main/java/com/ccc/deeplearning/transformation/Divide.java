package com.ccc.deeplearning.transformation;

import org.jblas.DoubleMatrix;

public class Divide implements MatrixTransform {

	/**
	 * 
	 */
	private static final long serialVersionUID = 3527150246619487854L;

	@Override
	public DoubleMatrix apply(DoubleMatrix input) {
		return input.div(input);
	}

	

}
