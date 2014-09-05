package org.nd4j.linalg.transformation;


import org.nd4j.linalg.api.ndarray.INDArray;

public class Divide implements MatrixTransform {

	/**
	 * 
	 */
	private static final long serialVersionUID = 3527150246619487854L;

	@Override
	public INDArray apply(INDArray input) {
		return input.div(input);
	}

	

}
