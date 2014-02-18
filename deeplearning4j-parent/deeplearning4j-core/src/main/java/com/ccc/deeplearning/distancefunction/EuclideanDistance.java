package com.ccc.deeplearning.distancefunction;

import org.jblas.DoubleMatrix;

public class EuclideanDistance extends BaseDistanceFunction {

	/**
	 * 
	 */
	private static final long serialVersionUID = -8043867712569910917L;

	public EuclideanDistance(DoubleMatrix base) {
		super(base);
	}

	@Override
	public Double apply(DoubleMatrix input) {
		return base.distance2(input);
	}

}
