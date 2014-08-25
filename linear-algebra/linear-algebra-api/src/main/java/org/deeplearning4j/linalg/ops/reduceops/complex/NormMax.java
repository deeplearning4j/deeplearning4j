package org.deeplearning4j.linalg.ops.reduceops.complex;

import org.deeplearning4j.linalg.api.complex.IComplexNDArray;
import org.deeplearning4j.linalg.api.complex.IComplexNumber;
import org.deeplearning4j.linalg.factory.NDArrays;
import org.deeplearning4j.linalg.util.ComplexUtil;

/**
 *
 * Over all normmax of an ndarray
 * @author Adam Gibson1
 */
public class NormMax extends BaseScalarOp {
    public NormMax() {
        super(NDArrays.createDouble(0,0));
    }

    @Override
    public IComplexNumber accumulate(IComplexNDArray arr, int i, IComplexNumber soFar) {
        IComplexNumber abs = ComplexUtil.abs(arr.getComplex(i));
        return abs.absoluteValue().doubleValue() > soFar.absoluteValue().doubleValue() ? abs : soFar;
    }
}
