package org.nd4j.linalg.ops.elementwise;


import org.nd4j.linalg.api.complex.IComplexNumber;

import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.BaseTwoArrayElementWiseOp;

/**
 * Divide a scalar or a matrix
 *
 * @author Adam Gibson
 */
public class DivideOp extends BaseTwoArrayElementWiseOp {


    @Override
    protected IComplexNumber complexComplex(IComplexNumber num1, IComplexNumber num2) {
        return num1.div(num2);
    }

    @Override
    protected IComplexNumber realComplex(double real, IComplexNumber other) {
        return Nd4j.createDouble(real / other.asFloat().realComponent(), other.asFloat().imaginaryComponent());
    }

    @Override
    protected IComplexNumber complexReal(IComplexNumber origin, double secondValue) {
        return origin.div(secondValue);
    }

    @Override
    protected double realReal(double firstElement, double secondElement) {
        return firstElement / secondElement;
    }
}
