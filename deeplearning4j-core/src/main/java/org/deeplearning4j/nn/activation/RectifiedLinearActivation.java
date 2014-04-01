package org.deeplearning4j.nn.activation;

import static org.jblas.MatrixFunctions.exp;
import static org.jblas.MatrixFunctions.log;

import org.jblas.DoubleMatrix;
/**
 * Rectified linear units:
 * log( 1 + exp(x))
 * 
 * Derivative:
 * exp(x) / 1 + exp(x))
 * 
 * @author Adam Gibson
 *
 */
public class RectifiedLinearActivation implements ActivationFunction {

	/**
	 * 
	 */
	private static final long serialVersionUID = -1843951127276313827L;
	
	@Override
	public DoubleMatrix apply(DoubleMatrix input) {
		DoubleMatrix ret = log(exp(input).add(1));
		return ret;
	}

	@Override
	public DoubleMatrix applyDerivative(DoubleMatrix input) {
		DoubleMatrix exped = exp(input);
		return exped.div(exped.add(1));
	}

	

}
