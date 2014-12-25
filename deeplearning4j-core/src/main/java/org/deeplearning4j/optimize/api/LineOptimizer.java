package org.deeplearning4j.optimize.api;

import org.deeplearning4j.exception.InvalidStepException;
import org.nd4j.linalg.api.ndarray.INDArray;

/**
 * Line optimizer interface adapted from mallet
 * @author Adam Gibson
 *
 */
public interface LineOptimizer {
	/** Returns the last step size used. */
	public double optimize (INDArray line,int iteration,double initialStep,INDArray x,INDArray g) throws InvalidStepException;


}
