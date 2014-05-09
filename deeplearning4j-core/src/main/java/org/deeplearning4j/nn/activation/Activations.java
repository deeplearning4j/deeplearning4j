package org.deeplearning4j.nn.activation;

/**
 * Activation Functions for neural nets
 * @author Adam Gibson
 */
public class Activations {

    /**
     * The e^x function
     * @return the e^x activation function
     */
    public static ActivationFunction exp(){ return new Exp(); }

    /**
     * Linear activation function, just returns the input as is
     * @return the linear activation function
     */
    public static ActivationFunction linear(){ return new Linear(); }

    /**
     * Tanh function
     * @return
     */
	public static ActivationFunction tanh() {
		return new Tanh();
	}

    /**
     * Sigmoid function
     * @return
     */
	public static ActivationFunction sigmoid() {
		return new Sigmoid();
	}

    /**
     * Hard Tanh is tanh constraining input to -1 to 1
     * @return the hard tanh function
     */
	public static ActivationFunction hardTanh() {
		return new HardTanh();
	}


    /**
     * Soft max function used for multinomial classification
     * @return
     */
	public static ActivationFunction softmax() {
		return new SoftMax();
	}
}
