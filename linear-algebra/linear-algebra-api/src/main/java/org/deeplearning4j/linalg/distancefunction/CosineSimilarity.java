package org.deeplearning4j.linalg.distancefunction;

import org.deeplearning4j.linalg.api.ndarray.INDArray;
import org.deeplearning4j.linalg.ops.transforms.Transforms;


public class CosineSimilarity extends BaseDistanceFunction {

	
	private static final long serialVersionUID = 3272217249919443730L;

	public CosineSimilarity(INDArray base) {
		super(base);
	}

	@Override
	public Double apply(INDArray input) {
		return Transforms.cosineSim(input, base);
	}

	
	
}
