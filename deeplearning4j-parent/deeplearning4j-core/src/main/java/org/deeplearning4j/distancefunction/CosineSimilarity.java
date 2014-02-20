package org.deeplearning4j.distancefunction;

import org.deeplearning4j.util.MatrixUtil;
import org.jblas.DoubleMatrix;


public class CosineSimilarity extends BaseDistanceFunction {

	
	private static final long serialVersionUID = 3272217249919443730L;

	public CosineSimilarity(DoubleMatrix base) {
		super(base);
	}

	@Override
	public Double apply(DoubleMatrix input) {
		return MatrixUtil.cosineSim(input, base);
	}

	
	
}
