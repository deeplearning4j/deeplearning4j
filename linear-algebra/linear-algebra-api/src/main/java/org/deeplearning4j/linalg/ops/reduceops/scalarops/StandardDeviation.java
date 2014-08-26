package org.deeplearning4j.linalg.ops.reduceops.scalarops;

import org.deeplearning4j.linalg.api.ndarray.INDArray;
import org.deeplearning4j.linalg.util.ArrayUtil;

/**
 * Return the overall standard deviation of an ndarray
 *
 * @author Adam Gibson
 */
public class StandardDeviation extends BaseScalarOp {
    public StandardDeviation() {
        super(0);
    }

    public float std(INDArray arr) {
        org.apache.commons.math3.stat.descriptive.moment.StandardDeviation dev = new org.apache.commons.math3.stat.descriptive.moment.StandardDeviation();
        float std = (float) dev.evaluate(ArrayUtil.doubleCopyOf(arr.data()));
        return std;
    }

   
    @Override
    public Float apply(INDArray input) {
        return std(input);
    }

    @Override
    public float accumulate(INDArray arr, int i, float soFar) {
        return 0;
    }
}
