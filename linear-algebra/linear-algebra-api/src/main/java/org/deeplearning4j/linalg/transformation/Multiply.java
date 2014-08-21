package org.deeplearning4j.linalg.transformation;


import org.deeplearning4j.linalg.api.ndarray.INDArray;

public class Multiply implements MatrixTransform {

	

	/**
	 * 
	 */
	private static final long serialVersionUID = 6270130254778514061L;

	@Override
	public INDArray apply(INDArray input) {
		return input.mul(input);
	}

}
