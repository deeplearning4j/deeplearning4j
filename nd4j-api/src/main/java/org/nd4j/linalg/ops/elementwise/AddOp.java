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
    protected IComplexNumber realComplex(float real, IComplexNumber other) {
        return other.add(real);
    }

    @Override
    protected IComplexNumber complexReal(IComplexNumber origin, float secondValue) {
        return origin.add(secondValue);
    }

    @Override
    protected float realReal(float firstElement, float secondElement) {
        return firstElement + secondElement;
    }
}
