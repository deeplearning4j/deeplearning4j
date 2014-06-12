package org.deeplearning4j.nn.activation;

import org.deeplearning4j.util.MatrixUtil;
import org.jblas.DoubleMatrix;

/**
 * Rectified Linear
 * max(0,input)
 * @author Adam Gibson
 */
public class RectifiedLinear extends BaseActivationFunction {
    /**
     * Applies the derivative of this function
     *
     * @param input the input to apply it to
     * @return the derivative of this function applied to
     * the input
     */
    @Override
    public DoubleMatrix applyDerivative(DoubleMatrix input) {
        return DoubleMatrix.ones(input.rows,input.columns);
    }


    @Override
    public DoubleMatrix apply(DoubleMatrix input) {
        DoubleMatrix dup = input.dup();
        MatrixUtil.max(0,dup);
        return dup;
    }
}
