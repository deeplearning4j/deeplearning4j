package org.deeplearning4j.util;

import org.jblas.DoubleMatrix;

public interface LineOptimizerMatrix {
	/** Returns the last step size used. */
	public double optimize (DoubleMatrix line, double initialStep);

	public interface ByGradient	{
		/** Returns the last step size used. */
		public double optimize (DoubleMatrix line, double initialStep);
	}
}
