package org.deeplearning4j.nn.activation;

import static org.deeplearning4j.util.MatrixUtil.oneMinus;
import static org.deeplearning4j.util.MatrixUtil.softmax;

import org.jblas.DoubleMatrix;
import org.jblas.FloatMatrix;

/**
 * Softmax function 
 * @author Adam Gibson
 *
 */
public class SoftMax extends BaseActivationFunction {

    @Override
    public FloatMatrix apply(FloatMatrix input) {
        return softmax(input);
    }

    @Override
    public FloatMatrix applyDerivative(FloatMatrix input) {
        return softmax(input).mul(oneMinus(softmax(input)));
    }

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
