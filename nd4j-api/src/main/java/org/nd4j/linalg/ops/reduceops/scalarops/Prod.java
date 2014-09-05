package org.nd4j.linalg.ops.reduceops.scalarops;

import org.nd4j.linalg.api.ndarray.INDArray;

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
    public float accumulate(INDArray arr, int i, float soFar) {
        return soFar * (float) arr.get(i);
    }
}
