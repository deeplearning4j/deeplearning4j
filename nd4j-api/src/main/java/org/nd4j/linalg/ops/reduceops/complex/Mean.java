package org.nd4j.linalg.ops.reduceops.complex;

import org.nd4j.linalg.api.complex.IComplexNDArray;
import org.nd4j.linalg.api.complex.IComplexNumber;
import org.nd4j.linalg.factory.Nd4j;

/**
 *
 * @author Adam Gibson
 */
public class Mean extends BaseScalarOp {

    public Mean() {
        super(Nd4j.createDouble(0, 0));
    }

    @Override
    public IComplexNumber accumulate(IComplexNDArray arr, int i, IComplexNumber soFar) {
       if(i < arr.length() - 1)
           return soFar.addi(arr.getComplex(i));
        else {
           soFar.addi(arr.getComplex(i));
           return soFar.div(arr.length());
       }

    }
}
