package org.deeplearning4j.nn.activation;


import org.jblas.DoubleMatrix;
import org.jblas.FloatMatrix;

import static org.deeplearning4j.util.MatrixUtil.*;
import static org.jblas.MatrixFunctions.*;

/**
 * Tanh activation function
 *
 * @author Adam Gibson
 */
public class Tanh extends BaseActivationFunction {


    /**
     *
     */
    private static final long serialVersionUID = 4499150153988528321L;

    /**
     * Name of the function
     *
     * @return the name of the function
     */
    @Override
    public String type() {
        return "tanh";
    }

    @Override
    public DoubleMatrix apply(DoubleMatrix arg0) {
        return tanh(arg0);
    }

    @Override
    public DoubleMatrix applyDerivative(DoubleMatrix input) {
        //1 - tanh^2 x
        return oneMinus(pow(input,2));
    }

    @Override
    public FloatMatrix apply(FloatMatrix input) {
        return tanh(input);
    }

    @Override
    public FloatMatrix applyDerivative(FloatMatrix input) {
        return oneMinus(pow(input,2));
    }



}
