package org.nd4j.linalg.ops.reduceops.complex;

import org.nd4j.linalg.api.complex.IComplexNDArray;
import org.nd4j.linalg.api.complex.IComplexNumber;
import org.nd4j.linalg.factory.Nd4j;

/**
 * Prod operator
 *
 * @author Adam Gibson
 */
public class Prod extends BaseScalarOp {

    public Prod() {
        super(Nd4j.createDouble(1, 1));
    }

    @Override
    public IComplexNumber accumulate(IComplexNDArray arr, int i, IComplexNumber soFar) {
        return soFar.mul(arr.getComplex(i));
    }
}
