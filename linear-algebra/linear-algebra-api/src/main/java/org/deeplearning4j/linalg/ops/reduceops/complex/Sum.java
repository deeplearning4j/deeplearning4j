package org.deeplearning4j.linalg.ops.reduceops.complex;

import org.deeplearning4j.linalg.api.complex.IComplexNDArray;
import org.deeplearning4j.linalg.api.complex.IComplexNumber;
import org.deeplearning4j.linalg.factory.NDArrays;

/**
 * Sum over an ndarray
 *
 * @author Adam Gibson
 */
public class Sum extends BaseScalarOp {

    public Sum() {
        super(NDArrays.createDouble(0,0));
    }

    @Override
    public IComplexNumber accumulate(IComplexNDArray arr, int i, IComplexNumber soFar) {
        IComplexNumber d = arr.getComplex(i);
        soFar.addi(d);
        return soFar;
    }
}
