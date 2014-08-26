package org.deeplearning4j.linalg.ops.reduceops.scalarops;

import org.apache.commons.math3.stat.StatUtils;
import org.deeplearning4j.linalg.api.ndarray.INDArray;
import org.deeplearning4j.linalg.util.ArrayUtil;

/**
 * Return the variance of an ndarray
 *
 * @author Adam Gibson
 */
public class Variance extends BaseScalarOp {

    public Variance() {
        super(1);
    }




    public float var(INDArray arr) {
        double mean = new Mean().apply(arr);
        return (float) StatUtils.variance(ArrayUtil.doubleCopyOf(arr.data()), mean);
    }



    @Override
    public Float apply(INDArray input) {
        return  var(input);
    }

    @Override
    public float accumulate(INDArray arr, int i, float soFar) {
        return 0;
    }
}
