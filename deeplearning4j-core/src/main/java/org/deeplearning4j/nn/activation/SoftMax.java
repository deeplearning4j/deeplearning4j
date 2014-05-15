package org.deeplearning4j.nn.activation;

import static org.deeplearning4j.util.MatrixUtil.oneMinus;
import static org.deeplearning4j.util.MatrixUtil.softmax;

import org.jblas.DoubleMatrix;
/**
 * Softmax function 
 * @author Adam Gibson
 *
 */
public class SoftMax extends BaseActivationFunction {

    /**
	 * 
	 */
	private static final long serialVersionUID = -3407472284248637360L;

	@Override
	public DoubleMatrix apply(DoubleMatrix input) {
		return softmax(input);
	}

	@Override
	public DoubleMatrix applyDerivative(DoubleMatrix input) {
		return softmax(input).mul(oneMinus(softmax(input)));

	}

}
