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
    public float accumulate(INDArray arr, int i, float soFar) {
       if(i < arr.length() - 1)
           return soFar + (float) arr.getScalar(i).element();
        else {
           soFar += (float) arr.getScalar(i).element();
           return soFar / arr.length();
       }

    }
}
