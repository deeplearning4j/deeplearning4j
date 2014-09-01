package org.deeplearning4j.linalg.ops.elementwise;

import org.deeplearning4j.linalg.api.complex.IComplexNumber;
import org.deeplearning4j.linalg.factory.NDArrays;
import org.deeplearning4j.linalg.ops.BaseTwoArrayElementWiseOp;

/**
 * Multiply a scalar or a matrix
 *
 * @author Adam Gibson
 */
public class MultiplyOp extends BaseTwoArrayElementWiseOp {


    @Override
    protected IComplexNumber complexComplex(IComplexNumber num1, IComplexNumber num2) {
        return num1.mul(num2);
    }

    @Override
    protected IComplexNumber realComplex(float real, IComplexNumber other) {
        return NDArrays.createDouble(real * other.realComponent().doubleValue(),other.complexArgument().doubleValue());
    }

    @Override
    protected IComplexNumber complexReal(IComplexNumber origin, float secondValue) {
        return origin.mul(secondValue);
    }

    @Override
    protected float realReal(float firstElement, float secondElement) {
        return firstElement * secondElement;
    }
}
