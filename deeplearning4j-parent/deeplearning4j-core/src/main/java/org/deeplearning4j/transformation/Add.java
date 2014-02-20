package org.deeplearning4j.transformation;

import org.jblas.DoubleMatrix;

public class Add implements MatrixTransform {

	/**
	 * 
	 */
	private static final long serialVersionUID = 9110741122587233634L;

	@Override
	public DoubleMatrix apply(DoubleMatrix input) {
		return input.add(input);
	}

}
