package com.ccc.deeplearning.distancefunction;

import org.jblas.DoubleMatrix;

public class ManhattanDistance extends BaseDistanceFunction {

	/**
	 * 
	 */
	private static final long serialVersionUID = -2421779223755051432L;

	public ManhattanDistance(DoubleMatrix base) {
		super(base);
	}

	@Override
	public Double apply(DoubleMatrix input) {
		return base.distance1(input);
	}

	
}
