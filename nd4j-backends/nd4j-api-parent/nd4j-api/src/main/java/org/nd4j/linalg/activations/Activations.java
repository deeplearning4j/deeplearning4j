package org.nd4j.linalg.activations;

import org.nd4j.linalg.activations.impl.*;

/**
 * Created by susaneraly on 12/8/16.
 */
public class Activations {

    public enum Activation {
        tanh,
        sigmoid,
        identity,
        leakyrelu,
        relu,
        rrelu,
        softmax,
        softsign;

        public IActivation getActivationFunction() {
            switch(this) {
                case tanh:
                    return new ActivationTanH();
                case sigmoid:
                    return new ActivationSigmoid();
                case identity:
                    return new ActivationIdentity();
                case leakyrelu:
                    return new ActivationLReLU();
                case relu:
                    return new ActivationReLU();
                case softmax:
                    return new ActivationSoftmax();
                case rrelu:
                    return new ActivationRReLU();
                case softsign:
                    return new ActivationSoftSign();
                default:
                    throw new UnsupportedOperationException("Unknown or not supported activation function: " + this);
            }
        }

    }
}
