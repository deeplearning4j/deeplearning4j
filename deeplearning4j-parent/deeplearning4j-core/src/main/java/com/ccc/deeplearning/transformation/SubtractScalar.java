package com.ccc.deeplearning.transformation;

import org.jblas.DoubleMatrix;

public class SubtractScalar extends ScalarMatrixTransform {

	/**
	 * 
	 */
	private static final long serialVersionUID = -5343280138416085068L;

	public SubtractScalar(double scaleBy) {
		super(scaleBy);
	}

	@Override
	public DoubleMatrix apply(DoubleMatrix input) {
		return input.sub(scaleBy);
	}


}
