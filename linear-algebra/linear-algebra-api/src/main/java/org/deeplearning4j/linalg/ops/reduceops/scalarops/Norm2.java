package org.deeplearning4j.linalg.ops.reduceops.scalarops;

import org.deeplearning4j.linalg.api.ndarray.INDArray;

/**
 * Overall norm2 of an ndarray
 *
 * @author Adam Gibson
 */
public class Norm2 extends BaseScalarOp {
    public Norm2() {
        super(0);
    }

    @Override
    public float accumulate(INDArray arr, int i, float soFar) {
        return soFar + (float) Math.pow((float) arr.getScalar(i).element(),2);
    }
}
