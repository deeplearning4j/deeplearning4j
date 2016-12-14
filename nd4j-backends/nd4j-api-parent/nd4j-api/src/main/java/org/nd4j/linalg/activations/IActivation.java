package org.nd4j.linalg.activations;

import org.nd4j.linalg.api.ndarray.INDArray;

import java.io.Serializable;

/**
 * Interface for implementing custom activation functions
 * @author Susan Eraly
 */
public interface IActivation extends Serializable {

    void setActivation(INDArray in, INDArray activation, boolean training);

    void setGradient(INDArray in, INDArray gradient);

    void setActivationAndGradient(INDArray in, INDArray activation, INDArray gradient);

}
