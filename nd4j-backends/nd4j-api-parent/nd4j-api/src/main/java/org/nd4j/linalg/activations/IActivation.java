package org.nd4j.linalg.activations;

import org.apache.commons.math3.util.Pair;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.io.Serializable;

/**
 * Interface for implementing custom activation functions
 * @author Susan Eraly
 */
public interface IActivation extends Serializable {

    /**
     * Carry out activation function on "in"
     * Best practice: Overwrite "in", transform in place and return "in"
     * Can support separate behaviour during test
     * @param in
     * @param training
     * @return tranformed activation
     */
    INDArray getActivation(INDArray in, boolean training);

    /**
     * Value of the partial derivative of the activation function at "in" with respect to the input (linear transformation with weights and biases, sometimes referred to as the preout)
     * Best practice: Overwrite "in" with the gradient and return "in"
     * @param in
     */
    INDArray getGradient(INDArray in);

    /**
     * return activation and gradient with respect to input "in"
     * @param in, input to activation node
     */
    Pair<INDArray, INDArray> getActivationAndGradient(INDArray in);

}
