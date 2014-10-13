package org.nd4j.linalg.ops.reduceops.scalarops;

import org.nd4j.linalg.api.ndarray.INDArray;

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
        double ret =  soFar + (double) Math.pow(arr.getFloat(i),2);
        if(i == arr.length() - 1)
            return (double) Math.sqrt(ret);
        return ret;
    }
}
