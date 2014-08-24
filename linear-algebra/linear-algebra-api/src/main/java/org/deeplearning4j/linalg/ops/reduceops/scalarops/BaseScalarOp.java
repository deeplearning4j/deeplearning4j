package org.deeplearning4j.linalg.ops.reduceops.scalarops;

import org.deeplearning4j.linalg.api.ndarray.INDArray;

/**
 *
 * Abstract class for scalar operations
 * @author Adam Gibson
 */
public abstract class BaseScalarOp implements ScalarOp {

    protected double startingValue;


    public BaseScalarOp(double startingValue) {
        this.startingValue = startingValue;
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
        INDArray doNDArray = input.isVector() ? input : input.ravel();
        double start = startingValue;
        for(int i = 0; i < doNDArray.length(); i++)
            start = accumulate(doNDArray,i,start);
        return start;
    }

    public abstract double accumulate(INDArray arr,int i,double soFar);


}
