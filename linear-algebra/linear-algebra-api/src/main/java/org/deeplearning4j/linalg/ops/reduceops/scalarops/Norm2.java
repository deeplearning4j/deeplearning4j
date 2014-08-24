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
    public double accumulate(INDArray arr, int i, double soFar) {
        return soFar + Math.pow((double) arr.getScalar(i).element(),2);
    }
}
