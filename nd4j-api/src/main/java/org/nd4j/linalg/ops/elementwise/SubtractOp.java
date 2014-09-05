package org.nd4j.linalg.ops.elementwise;


import org.nd4j.linalg.api.complex.IComplexNumber;
import org.nd4j.linalg.factory.NDArrays;
import org.nd4j.linalg.ops.BaseTwoArrayElementWiseOp;

/**
 * Add a scalar or a matrix
 *
 * @author Adam Gibson
 */
public class SubtractOp extends BaseTwoArrayElementWiseOp {


    @Override
    protected IComplexNumber complexComplex(IComplexNumber num1, IComplexNumber num2) {
        return num1.sub(num2);
    }

    @Override
    protected IComplexNumber realComplex(float real, IComplexNumber other) {
        return NDArrays.createDouble(real - other.realComponent().doubleValue(),other.imaginaryComponent().doubleValue());
    }

    @Override
    protected IComplexNumber complexReal(IComplexNumber origin, float secondValue) {
        return origin.sub(secondValue);
    }

    @Override
    protected float realReal(float firstElement, float secondElement) {
        return firstElement - secondElement;
    }
}
