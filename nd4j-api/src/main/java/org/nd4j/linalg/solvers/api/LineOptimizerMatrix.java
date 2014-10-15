package org.nd4j.linalg.solvers.api;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.solvers.exception.InvalidStepException;

/**
 * Line optimizer interface adapted from mallet
 * @author Adam Gibson
 *
 */
public interface LineOptimizerMatrix {
	/** Returns the last step size used. */
	public double optimize (INDArray line, int iteration, double initialStep) throws InvalidStepException;

	public interface ByGradient	{
		/** Returns the last step size used. */
		public double optimize (INDArray line, int iteration, double initialStep) throws InvalidStepException;
	}
}
