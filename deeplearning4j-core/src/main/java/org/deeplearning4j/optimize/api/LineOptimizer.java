package org.deeplearning4j.optimize.api;

import org.deeplearning4j.exception.InvalidStepException;
import org.nd4j.linalg.api.ndarray.INDArray;

/**
 * Line optimizer interface adapted from mallet
 * @author Adam Gibson
 *
 */
public interface LineOptimizer {
    /**
     * Line optimizer
     * @param initialStep the initial step size
     * @param x the parameters to optimize
     * @param g the gradient
     * @return the last step size used
     * @throws InvalidStepException
     */
	public double optimize (double initialStep,INDArray x,INDArray g) throws InvalidStepException;




}
