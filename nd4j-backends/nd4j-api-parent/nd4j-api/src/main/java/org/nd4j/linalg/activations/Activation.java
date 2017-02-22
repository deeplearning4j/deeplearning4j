package org.nd4j.linalg.activations;

import org.nd4j.linalg.activations.impl.*;

/**
 * This enum is the factory for the activation function.
 *
 * Created by susaneraly on 12/8/16.
 */
public enum Activation {
    CUBE, ELU, HARDSIGMOID, HARDTANH, IDENTITY, LEAKYRELU, RATIONALTANH, RELU, RRELU, SIGMOID, SOFTMAX, SOFTPLUS, SOFTSIGN, TANH;

    /**
     * Creates an instance of the activation function
     *
     * @return an instance of the activation function
     */
    public IActivation getActivationFunction() {
        switch (this) {
            case CUBE:
                return new ActivationCube();
            case ELU:
                return new ActivationELU();
            case HARDSIGMOID:
                return new ActivationHardSigmoid();
            case HARDTANH:
                return new ActivationHardTanH();
            case IDENTITY:
                return new ActivationIdentity();
            case LEAKYRELU:
                return new ActivationLReLU();
            case RATIONALTANH:
                return new ActivationRationalTanh();
            case RELU:
                return new ActivationReLU();
            case RRELU:
                return new ActivationRReLU();
            case SIGMOID:
                return new ActivationSigmoid();
            case SOFTMAX:
                return new ActivationSoftmax();
            case SOFTPLUS:
                return new ActivationSoftPlus();
            case SOFTSIGN:
                return new ActivationSoftSign();
            case TANH:
                return new ActivationTanH();
            default:
                throw new UnsupportedOperationException("Unknown or not supported activation function: " + this);
        }
    }

    /**
     * Returns the activation function enum value
     *
     * @param name the case-insensitive name of the activation function
     * @return the activation function enum value
     */
    public static Activation fromString(String name) {
        return Activation.valueOf(name.toUpperCase());
    }

}
