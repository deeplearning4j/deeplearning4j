package org.deeplearning4j.distancefunction;

import org.jblas.DoubleMatrix;

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
	protected DoubleMatrix base;

	public BaseDistanceFunction(DoubleMatrix base) {
		super();
		this.base = base;
	}
	
	

}
