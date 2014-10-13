package org.nd4j.linalg.ops.reduceops.scalarops;

import org.nd4j.linalg.api.ndarray.INDArray;



/**
 *
 * @author Adam Gibson
 */
public class Min extends BaseScalarOp {

    public Min() {
        super(Float.MAX_VALUE);
    }

    @Override
    public double accumulate(INDArray arr, int i, double soFar) {
        double curr = arr.getFloat(i);
        return soFar < curr ? soFar : curr;
    }
}
