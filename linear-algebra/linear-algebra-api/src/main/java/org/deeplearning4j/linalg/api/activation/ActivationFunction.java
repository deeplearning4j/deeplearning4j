package org.deeplearning4j.linalg.api.activation;

import com.google.common.base.Function;
import org.deeplearning4j.linalg.api.ndarray.INDArray;
import org.deeplearning4j.linalg.ops.ElementWiseOp;


import java.io.Serializable;

/**
 * An activation function for a hidden layer for neural networks
 * @author Adam Gibson
 *
 */
public interface ActivationFunction extends Function<INDArray,INDArray>,Serializable {


    /**
     * The class used for transformation
     * @return the class used for transformation
     */
    public Class<? extends ElementWiseOp> transformClazz();

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
	public INDArray applyDerivative(INDArray input);
}
