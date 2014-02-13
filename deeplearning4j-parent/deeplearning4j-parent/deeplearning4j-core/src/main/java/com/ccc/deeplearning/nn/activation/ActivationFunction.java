package com.ccc.deeplearning.nn.activation;

import java.io.Serializable;

import org.jblas.DoubleMatrix;

import com.google.common.base.Function;
/**
 * An activation function for a hidden layer for neural networks
 * @author Adam Gibson
 *
 */
public interface ActivationFunction extends Function<DoubleMatrix,DoubleMatrix>,Serializable {

	/**
	 * Applies the derivative of this function
	 * @param input the input to apply it to
	 * @return the derivative of this function applied to 
	 * the input
	 */
	public DoubleMatrix applyDerivative(DoubleMatrix input);
}
