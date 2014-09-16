package org.nd4j.linalg.distancefunction;


import org.nd4j.linalg.api.ndarray.INDArray;

public class ManhattanDistance extends BaseDistanceFunction {

	/**
	 * 
	 */
	private static final long serialVersionUID = -2421779223755051432L;

	public ManhattanDistance(INDArray base) {
		super(base);
	}

	@Override
	public Float apply(INDArray input) {
		return (float) base.distance1(input);
	}

	
}
