package org.nd4j.linalg.transformation;


import org.nd4j.linalg.api.ndarray.INDArray;

public class DivideScalar extends ScalarMatrixTransform {

	/**
	 * 
	 */
	private static final long serialVersionUID = -739159171002026018L;

	public DivideScalar(double scaleBy) {
		super(scaleBy);
	}

	@Override
	public INDArray apply(INDArray input) {
		return input.div(scaleBy);
	}

}
