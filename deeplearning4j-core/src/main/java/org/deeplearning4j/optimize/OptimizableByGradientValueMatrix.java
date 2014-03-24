package org.deeplearning4j.optimize;

import org.jblas.DoubleMatrix;

public interface OptimizableByGradientValueMatrix {

	public int getNumParameters ();

	public DoubleMatrix getParameters ();
	public double getParameter (int index);

	public void setParameters (DoubleMatrix params);
	public void setParameter (int index, double value);

	public DoubleMatrix getValueGradient ();
	public double getValue ();
}
