package org.deeplearning4j.linalg.transformation;


import org.deeplearning4j.linalg.api.ndarray.INDArray;

public class Add implements MatrixTransform {

	/**
	 * 
	 */
	private static final long serialVersionUID = 9110741122587233634L;

	@Override
	public INDArray apply(INDArray input) {
		return input.add(input);
	}

}
