package org.deeplearning4j.transformation;

import org.jblas.DoubleMatrix;

public class DivideScalar extends ScalarMatrixTransform {

	/**
	 * 
	 */
	private static final long serialVersionUID = -739159171002026018L;

	public DivideScalar(double scaleBy) {
		super(scaleBy);
	}

	@Override
	public DoubleMatrix apply(DoubleMatrix input) {
		return input.div(scaleBy);
	}

}
