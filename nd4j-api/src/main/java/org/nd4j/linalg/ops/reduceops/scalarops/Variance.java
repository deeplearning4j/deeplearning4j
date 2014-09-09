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


    /**
     * Variance (see: apache commons)
     * @param arr the ndarray to get the variance of
     * @return the variance for this ndarray
     */
    public float var(INDArray arr) {
        float mean = new Mean().apply(arr);
        float accum = 0.0f;
        float dev = 0.0f;
        float accum2 = 0.0f;
        for (int i = 0; i < arr.length(); i++) {
            dev = arr.get(i) - mean;
            accum += dev * dev;
            accum2 += dev;
        }

        float len = arr.length();
        //bias corrected
        return  (accum - (accum2 * accum2 / len)) / (len - 1.0f);


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
