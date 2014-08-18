package org.deeplearning4j.linalg.transformation;


import org.deeplearning4j.linalg.api.ndarray.INDArray;

public class MultiplyScalar extends ScalarMatrixTransform {

	/**
	 * 
	 */
	private static final long serialVersionUID = -6775591578587002601L;

	public MultiplyScalar(double scaleBy) {
		super(scaleBy);
	}

	@Override
	public INDArray apply(INDArray input) {
		return input.mul(scaleBy);
	}


}
