package org.deeplearning4j.nn.activation;
import static org.deeplearning4j.util.MatrixUtil.sigmoid;
import static org.deeplearning4j.util.MatrixUtil.oneMinus;

import org.deeplearning4j.util.MatrixUtil;
import org.jblas.DoubleMatrix;


public class Sigmoid implements ActivationFunction {

	/**
	 * 
	 */
	private static final long serialVersionUID = -6280602270833101092L;

	@Override
	public DoubleMatrix apply(DoubleMatrix arg0) {
		return sigmoid(arg0);
	}

	@Override
	public DoubleMatrix applyDerivative(DoubleMatrix input) {
		return input.mul(oneMinus(input));
	}

	

}
