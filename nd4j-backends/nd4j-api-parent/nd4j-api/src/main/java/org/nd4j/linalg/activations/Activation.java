package org.nd4j.linalg.activations;

import org.nd4j.linalg.activations.impl.*;

/**
 * Created by susaneraly on 12/8/16.
 */
public enum Activation {
    TANH,
    SIGMOID,
    IDENTITY,
    LEAKYRELU,
    RELU,
    RRELU,
    SOFTMAX,
    SOFTSIGN;

    public IActivation getActivationFunction() {
        switch(this) {
            case TANH:
                return new ActivationTanH();
            case SIGMOID:
                return new ActivationSigmoid();
            case IDENTITY:
                return new ActivationIdentity();
            case LEAKYRELU:
                return new ActivationLReLU();
            case RELU:
                return new ActivationReLU();
            case SOFTMAX:
                return new ActivationSoftmax();
            case RRELU:
                return new ActivationRReLU();
            case SOFTSIGN:
                return new ActivationSoftSign();
            default:
                throw new UnsupportedOperationException("Unknown or not supported activation function: " + this);
        }
    }

}
