package org.deeplearning4j.linalg.distancefunction;


import org.deeplearning4j.linalg.api.ndarray.INDArray;

public class ManhattanDistance extends BaseDistanceFunction {

	/**
	 * 
	 */
	private static final long serialVersionUID = -2421779223755051432L;

	public ManhattanDistance(INDArray base) {
		super(base);
	}

	@Override
	public Double apply(INDArray input) {
		return base.distance1(input);
	}

	
}
