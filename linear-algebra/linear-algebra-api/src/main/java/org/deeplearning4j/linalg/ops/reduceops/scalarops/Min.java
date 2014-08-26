package org.deeplearning4j.linalg.ops.reduceops.scalarops;

import org.deeplearning4j.linalg.api.ndarray.INDArray;



/**
 *
 * @author Adam Gibson
 */
public class Min extends BaseScalarOp {

    public Min() {
        super(Float.MAX_VALUE);
    }

    @Override
    public float accumulate(INDArray arr, int i, float soFar) {
        float curr = (float) arr.getScalar(i).element();
        return soFar < curr ? soFar : curr;
    }
}
