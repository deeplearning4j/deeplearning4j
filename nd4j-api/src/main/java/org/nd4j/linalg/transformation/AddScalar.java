package org.nd4j.linalg.transformation;

import org.nd4j.linalg.api.ndarray.INDArray;

public class AddScalar extends ScalarMatrixTransform {

	/**
	 * 
	 */
	private static final long serialVersionUID = -3327232631316515992L;

	public AddScalar(double scaleBy) {
		super(scaleBy);
	}

	@Override
	public INDArray apply(INDArray input) {
		return input.add(scaleBy);
	}

	
	
}
