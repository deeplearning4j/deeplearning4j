package org.nd4j.linalg.activations;

import org.nd4j.linalg.api.ndarray.INDArray;

import java.io.Serializable;
import java.util.List;

/**
 * Interface for implementing custom activation functions
 * @author Susan Eraly
 */
public interface IActivation extends Serializable {

    /**
     * Carry out activation function on "in" and write in place to "activation".
     * "in" and "activation" will be of the same shape
     * Can support separate behaviour during test
     * @param in
     * @param activation
     * @param training
     */
    void setActivation(INDArray in, INDArray activation, boolean training);

    /**
     * Value of the partial derivative of the activation function at "in" with respect to the input (linear transformation with weights and biases, sometimes referred to as the preout)
     * Write in place to "gradient"
     * "in" and "gradient" will be of the same shape
     * @param in
     * @param gradient
     */
    void setGradient(INDArray in, INDArray gradient);

    /**
     * Do setActivation and setGradient in one go
     * @param in
     * @param activation
     * @param gradient
     */
    void setActivationAndGradient(INDArray in, INDArray activation, INDArray gradient);

}
