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
     //whether to take row wise or column wise maxes on softmax calculation
    private boolean rows;

    /**
     * Initialize softmax with whether to use row wise or column wise features
     * @param rows whether to use row wise or column wise features for calculation
     */
    public SoftMax(boolean rows) {
        this.rows = rows;
    }

    /**
     * Initializes softmax with column wise features
     */
    public SoftMax() {
        this(false);
    }

    @Override
    public FloatMatrix apply(FloatMatrix input) {
        return softmax(input,rows);
    }

    @Override
    public FloatMatrix applyDerivative(FloatMatrix input) {
        return softmax(input,rows).mul(oneMinus(softmax(input,rows)));
    }

    /**
	 * 
	 */
	private static final long serialVersionUID = -3407472284248637360L;

	@Override
	public DoubleMatrix apply(DoubleMatrix input) {
		return softmax(input,rows);
	}

	@Override
	public DoubleMatrix applyDerivative(DoubleMatrix input) {
		return softmax(input).mul(oneMinus(softmax(input)));

	}

}
