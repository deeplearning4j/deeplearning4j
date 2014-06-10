package org.deeplearning4j.nn.activation;

/**
 * Base activation function: mainly to give the function a canonical representation
 */
public abstract class BaseActivationFunction implements ActivationFunction {
    /**
     * Name of the function
     *
     * @return the name of the function
     */
    @Override
    public String type() {
        return getClass().getName();
    }

    @Override
    public boolean equals(Object o) {
        return o != null && o instanceof ActivationFunction && o.getClass().getName().equals(type());
    }

    /**
     * The type()
     * @return
     */
    @Override
    public String toString() {
        return type();
    }


}
