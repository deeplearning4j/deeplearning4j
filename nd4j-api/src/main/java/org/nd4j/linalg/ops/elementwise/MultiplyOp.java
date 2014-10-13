package org.nd4j.linalg.ops.elementwise;

import org.nd4j.linalg.api.complex.IComplexNumber;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.BaseTwoArrayElementWiseOp;

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
    protected IComplexNumber realComplex(double real, IComplexNumber other) {
        return Nd4j.createDouble(real * other.realComponent().doubleValue(), other.complexArgument().doubleValue());
    }

    @Override
    protected IComplexNumber complexReal(IComplexNumber origin, double secondValue) {
        return origin.mul(secondValue);
    }

    @Override
    protected double realReal(double firstElement, double secondElement) {
        return firstElement * secondElement;
    }
}
