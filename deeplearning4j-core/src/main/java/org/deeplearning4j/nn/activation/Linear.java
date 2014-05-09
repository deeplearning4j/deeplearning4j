package org.deeplearning4j.nn.activation;
import static org.deeplearning4j.util.MatrixUtil.stabilizeInput;

import org.jblas.DoubleMatrix;

/**
 * Linear activation function
 * @author Adam Gibson
 */
public class Linear implements ActivationFunction {

    /**
     * Applies the derivative of this function
     *
     * @param input the input to apply it to
     * @return the derivative of this function applied to
     * the input
     */
    @Override
    public DoubleMatrix applyDerivative(DoubleMatrix input) {
        return input;
    }

    @Override
    public DoubleMatrix apply(DoubleMatrix input) {
        return stabilizeInput(input,1);
    }
}
