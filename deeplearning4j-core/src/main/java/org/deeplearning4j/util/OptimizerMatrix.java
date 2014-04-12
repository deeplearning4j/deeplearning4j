package org.deeplearning4j.util;


public interface OptimizerMatrix {

	
	public boolean optimize ();
	public boolean optimize (int numIterations);
	public boolean isConverged();
	void setMaxIterations(int maxIterations);
    void setTolerance(double tolerance);

}
