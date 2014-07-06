package org.deeplearning4j.nn.activation;

import static org.deeplearning4j.util.MatrixUtil.round;

import org.jblas.DoubleMatrix;
import org.jblas.FloatMatrix;

/**
 * Rounded output
 * @author Adam Gibson
 */
public class RoundedLinear extends BaseActivationFunction {

    @Override
    public FloatMatrix apply(FloatMatrix input) {
        return round(input);
    }

    @Override
    public FloatMatrix applyDerivative(FloatMatrix input) {
        return FloatMatrix.ones(input.rows,input.columns);
    }



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
        return round(input);
    }
}
