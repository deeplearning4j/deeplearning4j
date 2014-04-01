package org.deeplearning4j.nn.activation;

public class Activations {

	public static ActivationFunction tanh() {
		return new Tanh();
	}
	
	public static ActivationFunction sigmoid() {
		return new Sigmoid();
	}

	
	public static ActivationFunction hardTanh() {
		return new HardTanh();
	}
	
	public static ActivationFunction rectifiedLinear() {
		return new RectifiedLinearActivation();
	}
	
	public static ActivationFunction softmax() {
		return new SoftMax();
	}
}
