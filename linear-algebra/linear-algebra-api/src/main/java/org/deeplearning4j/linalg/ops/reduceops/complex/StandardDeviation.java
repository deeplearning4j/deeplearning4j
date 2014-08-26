package org.deeplearning4j.linalg.ops.reduceops.complex;

import org.deeplearning4j.linalg.api.complex.IComplexNDArray;
import org.deeplearning4j.linalg.api.complex.IComplexNumber;
import org.deeplearning4j.linalg.factory.NDArrays;
import org.deeplearning4j.linalg.util.ArrayUtil;

/**
 * Return the overall standard deviation of an ndarray
 *
 * @author Adam Gibson
 */
public class StandardDeviation extends BaseScalarOp {
    public StandardDeviation() {
        super(NDArrays.createDouble(0,0));
    }

    public IComplexNumber std(IComplexNDArray arr) {
        org.apache.commons.math3.stat.descriptive.moment.StandardDeviation dev = new org.apache.commons.math3.stat.descriptive.moment.StandardDeviation();
        double std = dev.evaluate(ArrayUtil.doubleCopyOf(arr.data()));
        return NDArrays.createDouble(std,0);
    }


    @Override
    public IComplexNumber apply(IComplexNDArray input) {
        return std(input);
    }

    @Override
    public IComplexNumber accumulate(IComplexNDArray arr, int i, IComplexNumber soFar) {
        return NDArrays.createDouble(0,0);
    }
}
