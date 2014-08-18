package org.deeplearning4j.linalg.transformation;

import org.deeplearning4j.linalg.api.ndarray.INDArray;
import org.deeplearning4j.linalg.ops.transforms.Transforms;


public class LogTransform implements MatrixTransform {

	/**
	 * 
	 */
	private static final long serialVersionUID = 8144081928477158772L;

	@Override
	public INDArray apply(INDArray input) {
		return Transforms.log(input.dup());
	}

	

}
