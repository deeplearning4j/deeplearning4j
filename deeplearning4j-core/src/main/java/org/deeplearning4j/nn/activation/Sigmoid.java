package org.deeplearning4j.nn.activation;
import static org.deeplearning4j.util.MatrixUtil.sigmoid;
import static org.deeplearning4j.util.MatrixUtil.oneMinus;

import org.deeplearning4j.util.MatrixUtil;
import org.jblas.DoubleMatrix;
import org.jblas.FloatMatrix;

/**
 * Sigmoid function
 * @author Adam Gibson
 */
public class Sigmoid extends BaseActivationFunction {

    @Override
    public FloatMatrix apply(FloatMatrix input) {
        return sigmoid(input);
    }

    @Override
    public FloatMatrix applyDerivative(FloatMatrix input) {
        return  input.mul(oneMinus(input));
    }

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
