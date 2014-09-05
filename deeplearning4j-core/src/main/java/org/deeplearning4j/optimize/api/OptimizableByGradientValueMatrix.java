package org.deeplearning4j.optimize.api;


import org.nd4j.linalg.api.ndarray.INDArray;

public interface OptimizableByGradientValueMatrix {

	public int getNumParameters ();

	public INDArray getParameters ();


    public float getParameter(int index);

	public void setParameters (INDArray params);


    public void setParameter (int index, float value);

	public INDArray getValueGradient (int iteration);


    public float getValue ();


    void setCurrentIteration(int value);
}
