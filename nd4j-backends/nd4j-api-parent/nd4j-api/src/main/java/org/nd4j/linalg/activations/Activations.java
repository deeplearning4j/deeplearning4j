package org.nd4j.linalg.activations;

import org.nd4j.linalg.activations.impl.ActivationSigmoid;

/**
 * Created by susaneraly on 12/8/16.
 */
public class Activations {

    public enum Activation {
        identity,
        tanh,
        sigmoid,
        leakyrelu,
        relu,
        softmax,
        softsign;

        public IActivation getActivationFunction() {
            switch(this) {
                case sigmoid:
                    return new ActivationSigmoid();
                default:
                    throw new UnsupportedOperationException("Unknown or not supported activation function: " + this);
            }
        }

    }
}
