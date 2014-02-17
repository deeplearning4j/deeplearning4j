package com.ccc.deeplearning.transformation;

import org.jblas.DoubleMatrix;

public class Subtract implements MatrixTransform {

	/**
	 * 
	 */
	private static final long serialVersionUID = -604699802899787537L;

	@Override
	public DoubleMatrix apply(DoubleMatrix input) {
		return input.sub(input);
	}


}
