package org.nd4j.linalg.ops.reduceops.complex;

import org.nd4j.linalg.api.complex.IComplexNDArray;
import org.nd4j.linalg.api.complex.IComplexNumber;
import org.nd4j.linalg.factory.NDArrays;

/**
 *
 * @author Adam Gibson
 */
public class Max extends BaseScalarOp {
    public Max() {
        super(NDArrays.createDouble(Double.MIN_VALUE,0));
    }

    @Override
    public IComplexNumber accumulate(IComplexNDArray arr, int i, IComplexNumber soFar) {
        IComplexNumber curr = (IComplexNumber) arr.getScalar(i).element();
        return soFar.absoluteValue().doubleValue() > curr.absoluteValue().doubleValue() ? soFar : curr;
    }
}
