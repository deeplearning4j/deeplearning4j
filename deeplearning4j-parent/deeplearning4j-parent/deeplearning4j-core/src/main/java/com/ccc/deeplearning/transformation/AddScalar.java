package com.ccc.deeplearning.transformation;

import org.jblas.DoubleMatrix;

public class AddScalar extends ScalarMatrixTransform {

	/**
	 * 
	 */
	private static final long serialVersionUID = -3327232631316515992L;

	public AddScalar(double scaleBy) {
		super(scaleBy);
	}

	@Override
	public DoubleMatrix apply(DoubleMatrix input) {
		return input.add(scaleBy);
	}

	
	
}
