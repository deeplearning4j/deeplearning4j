package com.ccc.deeplearning.distancefunction;

import org.jblas.DoubleMatrix;

import com.ccc.deeplearning.util.MatrixUtil;

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
