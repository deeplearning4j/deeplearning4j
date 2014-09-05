package org.nd4j.linalg.ops.reduceops.complex;

import org.apache.commons.math3.stat.StatUtils;
import org.nd4j.linalg.api.complex.IComplexNDArray;
import org.nd4j.linalg.api.complex.IComplexNumber;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.util.ArrayUtil;

/**
 * Return the variance of an ndarray
 *
 * @author Adam Gibson
 */
public class Variance extends BaseScalarOp {

    public Variance() {
        super(Nd4j.createDouble(0, 0));
    }




    public IComplexNumber var(IComplexNDArray arr) {
        IComplexNumber mean = new Mean().apply(arr);
        return Nd4j.createDouble(StatUtils.variance(ArrayUtil.doubleCopyOf(arr.data()), mean.absoluteValue().floatValue()), 0);
    }



    @Override
    public IComplexNumber apply(IComplexNDArray input) {
        return  var(input);
    }

    @Override
    public IComplexNumber accumulate(IComplexNDArray arr, int i, IComplexNumber soFar) {
        return Nd4j.createDouble(0, 0);
    }
}
