package org.nd4j.linalg.ops.elementwise;


import org.nd4j.linalg.api.complex.IComplexNumber;
import org.nd4j.linalg.ops.BaseTwoArrayElementWiseOp;

/**
 * Add a scalar or a matrix
 *
 * @author Adam Gibson
 */
public class AddOp extends BaseTwoArrayElementWiseOp {


    @Override
    protected IComplexNumber complexComplex(IComplexNumber num1, IComplexNumber num2) {
        return num1.add(num2);
    }

    @Override
    protected IComplexNumber realComplex(double real, IComplexNumber other) {
        return other.add(real);
    }

    @Override
    protected IComplexNumber complexReal(IComplexNumber origin, double secondValue) {
        return origin.add(secondValue);
    }

    @Override
    protected double realReal(double firstElement, double secondElement) {
        return firstElement + secondElement;
    }
}
