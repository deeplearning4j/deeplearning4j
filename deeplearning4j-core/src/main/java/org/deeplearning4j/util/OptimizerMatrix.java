package org.deeplearning4j.util;

import cc.mallet.optimize.Optimizable;

public interface OptimizerMatrix {

	
	public boolean optimize ();
	public boolean optimize (int numIterations);
	public boolean isConverged();
	
}
