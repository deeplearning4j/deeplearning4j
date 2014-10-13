package org.nd4j.linalg.ops.reduceops.scalarops;

import org.nd4j.linalg.api.ndarray.INDArray;

/**
 * Return the overall standard deviation of an ndarray
 *
 * @author Adam Gibson
 */
public class StandardDeviation extends BaseScalarOp {
    public StandardDeviation() {
        super(0);
    }

    public double std(INDArray arr) {
        org.apache.commons.math3.stat.descriptive.moment.StandardDeviation dev = new org.apache.commons.math3.stat.descriptive.moment.StandardDeviation();
        double[] test = new double[arr.length()];
        INDArray linear = arr.linearView();
        for(int i = 0; i < linear.length(); i++)
            test[i] = linear.getDouble(i);
        double std =  dev.evaluate(test);
        return std;
    }

   
    @Override
    public Double apply(INDArray input) {
        return std(input);
    }

    @Override
    public double accumulate(INDArray arr, int i, double soFar) {
        return 0;
    }
}
