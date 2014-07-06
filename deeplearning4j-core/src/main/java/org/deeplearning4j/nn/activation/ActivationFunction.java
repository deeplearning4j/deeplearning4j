package org.deeplearning4j.nn.activation;

import java.io.Serializable;

import org.jblas.DoubleMatrix;

import com.google.common.base.Function;
import org.jblas.FloatMatrix;

/**
 * An activation function for a hidden layer for neural networks
 * @author Adam Gibson
 *
 */
public interface ActivationFunction extends Function<DoubleMatrix,DoubleMatrix>,Serializable {


    public FloatMatrix apply(FloatMatrix input);


    public FloatMatrix applyDerivative(FloatMatrix input);


    /**
     * Name of the function
     * @return the name of the function
     */
    public String type();


	/**
	 * Applies the derivative of this function
	 * @param input the input to apply it to
	 * @return the derivative of this function applied to 
	 * the input
	 */
	public DoubleMatrix applyDerivative(DoubleMatrix input);
}
