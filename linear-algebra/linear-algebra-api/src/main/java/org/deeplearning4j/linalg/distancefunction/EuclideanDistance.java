package org.deeplearning4j.linalg.distancefunction;


import org.deeplearning4j.linalg.api.ndarray.INDArray;

public class EuclideanDistance extends BaseDistanceFunction {

	/**
	 * 
	 */
	private static final long serialVersionUID = -8043867712569910917L;

	public EuclideanDistance(INDArray base) {
		super(base);
	}

	@Override
	public Double apply(INDArray input) {
		return base.distance2(input);
	}

}
