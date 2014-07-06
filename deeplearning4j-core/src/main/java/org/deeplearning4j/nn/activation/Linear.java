package org.deeplearning4j.nn.activation;

import org.jblas.DoubleMatrix;
import org.jblas.FloatMatrix;

/**
 * Linear activation function
 * @author Adam Gibson
 */
public class Linear extends BaseActivationFunction {


    @Override
    public FloatMatrix apply(FloatMatrix input) {
        return input.dup();
    }

    @Override
    public FloatMatrix applyDerivative(FloatMatrix input) {
        return  FloatMatrix.ones(input.rows,input.columns);
    }


    /**
     * Name of the function
     *
     * @return the name of the function
     */
    @Override
    public String type() {
        return "linear";
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
        return input.dup();
    }
}
