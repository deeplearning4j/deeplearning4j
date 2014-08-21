package org.deeplearning4j.linalg.transformation;

import org.deeplearning4j.linalg.api.ndarray.INDArray;
import org.deeplearning4j.linalg.ops.transforms.Transforms;

public class ExpTransform implements MatrixTransform {

	/**
	 * 
	 */
	private static final long serialVersionUID = 5544429281399904369L;

	@Override
	public INDArray apply(INDArray input) {
		return Transforms.exp(input.dup());
	}

	

}
