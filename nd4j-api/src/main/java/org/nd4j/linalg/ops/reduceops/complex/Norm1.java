package org.nd4j.linalg.ops.reduceops.complex;

import org.nd4j.linalg.api.complex.IComplexNDArray;
import org.nd4j.linalg.api.complex.IComplexNumber;
import org.nd4j.linalg.factory.NDArrays;

/**
 *
 * Overall norm1 of an ndarray
 *
 * @author Adam Gibson
 */
public class Norm1 extends BaseScalarOp {
    public Norm1() {
        super(NDArrays.createDouble(0,0));
    }

    @Override
    public IComplexNumber accumulate(IComplexNDArray arr, int i, IComplexNumber soFar) {
        return soFar.add(arr.getComplex(i).absoluteValue());
    }
}
