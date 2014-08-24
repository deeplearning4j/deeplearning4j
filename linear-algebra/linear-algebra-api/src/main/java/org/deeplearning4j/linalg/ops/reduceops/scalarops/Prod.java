package org.deeplearning4j.linalg.ops.reduceops.scalarops;

import org.deeplearning4j.linalg.api.ndarray.INDArray;

/**
 * Prod operator
 *
 * @author Adam Gibson
 */
public class Prod extends BaseScalarOp {

    public Prod() {
        super(1);
    }

    @Override
    public double accumulate(INDArray arr, int i, double soFar) {
        return soFar * (double) arr.getScalar(i).element();
    }
}
