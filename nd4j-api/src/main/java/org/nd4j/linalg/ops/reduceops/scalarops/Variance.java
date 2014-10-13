package org.nd4j.linalg.ops.reduceops.scalarops;

import org.nd4j.linalg.api.ndarray.INDArray;

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
     * @param arr the ndarray to getDouble the variance of
     * @return the variance for this ndarray
     */
    public double var(INDArray arr) {
        double mean = new Mean().apply(arr);
        double accum = 0.0f;
        double dev = 0.0f;
        double accum2 = 0.0f;
        for (int i = 0; i < arr.length(); i++) {
            dev = arr.getDouble(i) - mean;
            accum += dev * dev;
            accum2 += dev;
        }

        double len = arr.length();
        //bias corrected
        return  (accum - (accum2 * accum2 / len)) / (len - 1.0f);


    }



    @Override
    public Double apply(INDArray input) {
        return  var(input);
    }

    @Override
    public double accumulate(INDArray arr, int i, double soFar) {
        return 0;
    }
}
