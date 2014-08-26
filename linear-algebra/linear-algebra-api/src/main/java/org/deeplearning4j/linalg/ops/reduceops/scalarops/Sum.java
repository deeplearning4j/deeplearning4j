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
    public float accumulate(INDArray arr, int i, float soFar) {
        return (float) arr.getScalar(i).element() + soFar;
    }
}
