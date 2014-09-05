package org.nd4j.linalg.ops.reduceops.scalarops;

import org.nd4j.linalg.api.ndarray.INDArray;

/**
 *
 * @author Adam Gibson
 */
public class Mean extends BaseScalarOp {

    public Mean() {
        super(0);
    }

    @Override
    public float accumulate(INDArray arr, int i, float soFar) {
       if(i < arr.length() - 1)
           return soFar + arr.get(i);
        else {
           soFar +=  arr.get(i);
           return soFar / arr.length();
       }

    }
}
