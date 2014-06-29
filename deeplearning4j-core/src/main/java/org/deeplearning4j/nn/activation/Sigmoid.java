package org.deeplearning4j.nn.activation;
import static org.deeplearning4j.util.MatrixUtil.sigmoid;
import static org.deeplearning4j.util.MatrixUtil.oneMinus;

import org.deeplearning4j.util.MatrixUtil;
import org.jblas.DoubleMatrix;

/**
 * Sigmoid function
 * @author Adam Gibson
 */
public class Sigmoid extends BaseActivationFunction {

    /**
	 * 
	 */
	private static final long serialVersionUID = -6280602270833101092L;

	@Override
	public DoubleMatrix apply(DoubleMatrix input) {
		return sigmoid(input);
	}

	@Override
	public DoubleMatrix applyDerivative(DoubleMatrix input) {
		return input.mul(oneMinus(input));
	}

	

}
