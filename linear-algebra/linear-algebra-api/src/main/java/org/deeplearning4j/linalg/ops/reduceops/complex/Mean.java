package org.deeplearning4j.linalg.ops.reduceops.complex;

import org.deeplearning4j.linalg.api.complex.IComplexNDArray;
import org.deeplearning4j.linalg.api.complex.IComplexNumber;
import org.deeplearning4j.linalg.factory.NDArrays;

/**
 *
 * @author Adam Gibson
 */
public class Mean extends BaseScalarOp {

    public Mean() {
        super(NDArrays.createDouble(0,0));
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
