package com.ccc.deeplearning.transformation;

import org.jblas.DoubleMatrix;

public class Multiply implements MatrixTransform {

	

	/**
	 * 
	 */
	private static final long serialVersionUID = 6270130254778514061L;

	@Override
	public DoubleMatrix apply(DoubleMatrix input) {
		return input.mul(input);
	}

}
