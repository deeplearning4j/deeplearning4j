package org.nd4j.linalg.ops.reduceops.scalarops;

import org.nd4j.linalg.api.ndarray.INDArray;

/**
 *
 * Overall norm1 of an ndarray
 *
 * @author Adam Gibson
 */
public class Norm1 extends BaseScalarOp {
    public Norm1() {
        super(0);
    }

    @Override
    public float accumulate(INDArray arr, int i, float soFar) {
        return soFar + Math.abs(arr.get(i));
    }
}
