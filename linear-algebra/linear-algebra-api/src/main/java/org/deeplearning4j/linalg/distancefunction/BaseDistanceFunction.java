package org.deeplearning4j.linalg.distancefunction;


import org.deeplearning4j.linalg.api.ndarray.INDArray;

/**
 * Takes in another matrix
 * @author Adam Gibson
 *
 */
public abstract class BaseDistanceFunction implements DistanceFunction {

	/**
	 * 
	 */
	private static final long serialVersionUID = 5462280593045742751L;
	protected INDArray base;

	public BaseDistanceFunction(INDArray base) {
		super();
		this.base = base;
	}
	
	

}
