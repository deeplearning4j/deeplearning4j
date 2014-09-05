package org.deeplearning4j.linalg.ops.reduceops.scalarops;

import org.deeplearning4j.linalg.api.ndarray.INDArray;

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
    public float accumulate(INDArray arr, int i, float soFar) {
        float ret =  soFar + (float) Math.pow(arr.get(i),2);
        if(i == arr.length() - 1)
            return (float) Math.sqrt(ret);
        return ret;
    }
}
