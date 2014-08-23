package org.deeplearning4j.linalg.transformation;


import org.deeplearning4j.linalg.api.ndarray.INDArray;
import org.deeplearning4j.linalg.ops.transforms.Transforms;

public class SqrtScalar implements MatrixTransform  {

	/**
	 * 
	 */
	private static final long serialVersionUID = 6829106644052110114L;


	@Override
	public INDArray apply(INDArray input) {
		return Transforms.sqrt(input);
	}

	

}
