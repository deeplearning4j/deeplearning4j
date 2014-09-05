package org.nd4j.linalg.transformation;


import org.nd4j.linalg.api.ndarray.INDArray;

public abstract class ScalarMatrixTransform implements MatrixTransform {

	/**
	 * 
	 */
	private static final long serialVersionUID = -2491009087533310977L;
	protected double scaleBy;
	
	public ScalarMatrixTransform(double scaleBy) {
		this.scaleBy = scaleBy;
	}

	@Override
	public abstract INDArray apply(INDArray input);
	
	
}
