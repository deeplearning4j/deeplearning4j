package org.deeplearning4j.nn.activation;

import static org.deeplearning4j.util.MatrixUtil.stabilizeInput;
import static org.deeplearning4j.util.MatrixUtil.exp;

import org.jblas.DoubleMatrix;
import org.jblas.FloatMatrix;

/**
 * The exponential activation function
 * @author Adam Gibson
 */
public class Exp extends BaseActivationFunction {


    @Override
    public FloatMatrix apply(FloatMatrix input) {
        return  exp(stabilizeInput(input,1));
    }

    @Override
    public FloatMatrix applyDerivative(FloatMatrix input) {
        return apply(input);
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
        return apply(input);
    }

    /**
     * Returns the result of applying this function to {@code input}. This method is <i>generally
     * expected</i>, but not absolutely required, to have the following properties:
     * <p/>
     * <ul>
     * <li>Its execution does not cause any observable side effects.
     * <li>The computation is <i>consistent with equals</i>; that is, {@link Objects#equal
     * Objects.equal}{@code (a, b)} implies that {@code Objects.equal(function.apply(a),
     * function.apply(b))}.
     * </ul>
     *
     * @param input
     * @throws NullPointerException if {@code input} is null and this function does not accept null
     *                              arguments
     */
    @Override
    public DoubleMatrix apply(DoubleMatrix input) {
        return exp(stabilizeInput(input,1));
    }
}
