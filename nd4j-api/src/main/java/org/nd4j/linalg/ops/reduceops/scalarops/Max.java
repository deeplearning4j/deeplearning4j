package org.nd4j.linalg.ops.reduceops.scalarops;

import org.nd4j.linalg.api.ndarray.INDArray;

/**
 *
 * @author Adam Gibson
 */
public class Max extends BaseScalarOp {
    public Max() {
        super(Double.MIN_VALUE);
    }

    @Override
    public double accumulate(INDArray arr, int i, double soFar) {
        double curr = arr.getDouble(i);
        return soFar > curr ? soFar : curr;
    }
}
