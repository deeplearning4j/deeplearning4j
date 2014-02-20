package org.deeplearning4j.transformation;

import org.jblas.DoubleMatrix;
import org.jblas.MatrixFunctions;

public class PowScale extends ScalarMatrixTransform {

	/**
	 * 
	 */
	private static final long serialVersionUID = 170216110009564940L;

	public PowScale(double scaleBy) {
		super(scaleBy);
	}

	@Override
	public DoubleMatrix apply(DoubleMatrix input) {
		return MatrixFunctions.pow(scaleBy, input);
	}

	

}
