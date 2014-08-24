package org.deeplearning4j.linalg.ops.reduceops.scalarops;

import org.deeplearning4j.linalg.api.ndarray.INDArray;

/**
 *
 * @author Adam Gibson
 */
public class Mean extends BaseScalarOp {

    public Mean() {
        super(0);
    }

    @Override
    public double accumulate(INDArray arr, int i, double soFar) {
       if(i < arr.length() - 1)
           return soFar + (double) arr.getScalar(i).element();
        else {
           soFar += (double) arr.getScalar(i).element();
           return soFar / arr.length();
       }

    }
}
