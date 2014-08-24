package org.deeplearning4j.linalg.ops.reduceops.scalarops;

import org.deeplearning4j.linalg.api.ndarray.INDArray;

/**
 * Sum over an ndarray
 *
 * @author Adam Gibson
 */
public class Sum extends BaseScalarOp {

    public Sum() {
        super(0);
    }

    @Override
    public double accumulate(INDArray arr, int i, double soFar) {
        return (double) arr.getScalar(i).element() + soFar;
    }
}
