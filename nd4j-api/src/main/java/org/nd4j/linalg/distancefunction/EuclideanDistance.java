package org.nd4j.linalg.distancefunction;


import org.nd4j.linalg.api.ndarray.INDArray;

public class EuclideanDistance extends BaseDistanceFunction {

	/**
	 * 
	 */
	private static final long serialVersionUID = -8043867712569910917L;

	public EuclideanDistance(INDArray base) {
		super(base);
	}

	@Override
	public Float apply(INDArray input) {
		return (float) base.distance2(input);
	}

}
