package org.nd4j.linalg.ops.reduceops.scalarops;

import org.apache.commons.math3.stat.StatUtils;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.util.ArrayUtil;

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
        float temp = 0;
        for(int i = 0; i < arr.length(); i++)
            temp += (mean - arr.get(i))* (mean - arr.get(i));
        return temp / arr.length();

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
