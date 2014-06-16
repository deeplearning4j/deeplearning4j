package org.deeplearning4j.optimize;

import org.jblas.DoubleMatrix;
/**
 * Line optimizer interface adapted from mallet
 * @author Adam Gibson
 *
 */
public interface LineOptimizerMatrix {
	/** Returns the last step size used. */
	public double optimize (DoubleMatrix line, int iteration,double initialStep);

	public interface ByGradient	{
		/** Returns the last step size used. */
		public double optimize (DoubleMatrix line, int iteration,double initialStep);
	}
}
