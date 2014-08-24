package org.deeplearning4j.linalg.ops.reduceops.scalarops;

import org.deeplearning4j.linalg.api.ndarray.INDArray;

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
        double curr = (double) arr.getScalar(i).element();
        return soFar > curr ? soFar : curr;
    }
}
