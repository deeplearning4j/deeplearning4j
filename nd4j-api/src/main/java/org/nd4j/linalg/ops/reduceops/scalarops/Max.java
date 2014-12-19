package org.nd4j.linalg.ops.reduceops.scalarops;

import org.nd4j.linalg.api.ndarray.INDArray;

/**
 * Returns the max of the ndarray
 * @author Adam Gibson
 */
public class Max extends BaseScalarOp {
    public Max() {
        super(0.0);
    }

    @Override
    public double accumulate(INDArray arr, int i, double soFar) {
        double curr = arr.getDouble(i);
        return soFar > curr ? soFar : curr;
    }
}
