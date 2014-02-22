package org.deeplearning4j.transformation;

import org.jblas.DoubleMatrix;
import org.jblas.MatrixFunctions;

public class SqrtScalar implements MatrixTransform  {

	/**
	 * 
	 */
	private static final long serialVersionUID = 6829106644052110114L;


	@Override
	public DoubleMatrix apply(DoubleMatrix input) {
		return MatrixFunctions.sqrt(input);
	}

	

}
