package org.deeplearning4j.linalg.ops.reduceops.scalarops;

import org.apache.commons.math3.stat.StatUtils;
import org.deeplearning4j.linalg.api.ndarray.INDArray;

/**
 * Return the variance of an ndarray
 *
 * @author Adam Gibson
 */
public class Variance extends BaseScalarOp {

    public Variance() {
        super(1);
    }




    public double var(INDArray arr) {
        double mean = new Mean().apply(arr);
        return StatUtils.variance(arr.data(), mean);
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
    public Double apply(INDArray input) {
        return  var(input);
    }

    @Override
    public double accumulate(INDArray arr, int i, double soFar) {
        return 0;
    }
}
