package org.nd4j.linalg.solvers.api;


import org.nd4j.linalg.api.ndarray.INDArray;

public interface OptimizableByGradientValueMatrix {

	public int getNumParameters();

	public INDArray getParameters();


    public double getParameter(int index);

	public void setParameters (INDArray params);


    public void setParameter(int index, double value);

	public INDArray getValueGradient(int iteration);


    public double getValue();


    void setCurrentIteration(int value);
}
