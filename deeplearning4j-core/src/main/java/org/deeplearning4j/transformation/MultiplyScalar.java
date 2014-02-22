package org.deeplearning4j.transformation;

import org.jblas.DoubleMatrix;

public class MultiplyScalar extends ScalarMatrixTransform {

	/**
	 * 
	 */
	private static final long serialVersionUID = -6775591578587002601L;

	public MultiplyScalar(double scaleBy) {
		super(scaleBy);
	}

	@Override
	public DoubleMatrix apply(DoubleMatrix input) {
		return input.mul(scaleBy);
	}


}
