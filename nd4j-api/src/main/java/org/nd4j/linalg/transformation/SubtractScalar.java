package org.nd4j.linalg.transformation;


import org.nd4j.linalg.api.ndarray.INDArray;

public class SubtractScalar extends ScalarMatrixTransform {

	/**
	 * 
	 */
	private static final long serialVersionUID = -5343280138416085068L;

	public SubtractScalar(double scaleBy) {
		super(scaleBy);
	}

	@Override
	public INDArray apply(INDArray input) {
		return input.sub(scaleBy);
	}


}
