package org.deeplearning4j.optimize.distance;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.distancefunction.BaseDistanceFunction;
import org.nd4j.linalg.distancefunction.CosineSimilarity;

public class CosineDistance extends BaseDistanceFunction {

	private static final long	serialVersionUID	= 693813798951786016L;
	
	private CosineSimilarity similarity;
	
	public CosineDistance(INDArray base) {
		super(base);
		similarity = new CosineSimilarity(base);
	}

	@Override
	public Float apply(INDArray input) {
		return (float) (1 - similarity.apply(input));
	}

}
