package org.nd4j.linalg.transformation;


import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.ops.transforms.Transforms;

public class PowScale extends ScalarMatrixTransform {

	/**
	 * 
	 */
	private static final long serialVersionUID = 170216110009564940L;

	public PowScale(double scaleBy) {
		super(scaleBy);
	}

	@Override
	public INDArray apply(INDArray input) {
		return Transforms.pow(input.dup(),scaleBy);
	}

	

}
