package org.nd4j.linalg.ops.reduceops.scalarops;

import org.nd4j.linalg.api.ndarray.INDArray;

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
        return arr.get(i) + soFar;
    }
}
