package org.deeplearning4j.linalg.ops.reduceops.complex;

import org.deeplearning4j.linalg.api.complex.IComplexNDArray;
import org.deeplearning4j.linalg.api.complex.IComplexNumber;
import org.deeplearning4j.linalg.factory.NDArrays;
import org.deeplearning4j.linalg.util.ComplexUtil;

/**
 * Overall norm2 of an ndarray
 *
 * @author Adam Gibson
 */
public class Norm2 extends BaseScalarOp {
    public Norm2() {
        super(NDArrays.createDouble(0,0));
    }

    @Override
    public IComplexNumber accumulate(IComplexNDArray arr, int i, IComplexNumber soFar) {
        return soFar.addi(ComplexUtil.pow(arr.getComplex(i),2));
    }
}
