package com.ccc.deeplearning.transformation;

import org.jblas.DoubleMatrix;

public abstract class ScalarMatrixTransform implements MatrixTransform {

	/**
	 * 
	 */
	private static final long serialVersionUID = -2491009087533310977L;
	protected double scaleBy;
	
	public ScalarMatrixTransform(double scaleBy) {
		this.scaleBy = scaleBy;
	}

	@Override
	public abstract DoubleMatrix apply(DoubleMatrix input);
	
	
}
