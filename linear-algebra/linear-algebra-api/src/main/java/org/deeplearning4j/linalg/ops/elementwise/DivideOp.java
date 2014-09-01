package org.deeplearning4j.linalg.ops.elementwise;


import org.deeplearning4j.linalg.api.complex.IComplexNumber;

import org.deeplearning4j.linalg.factory.NDArrays;
import org.deeplearning4j.linalg.ops.BaseTwoArrayElementWiseOp;

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
    protected IComplexNumber realComplex(float real, IComplexNumber other) {
        return NDArrays.createDouble(real / other.asFloat().realComponent(),other.asFloat().imaginaryComponent());
    }

    @Override
    protected IComplexNumber complexReal(IComplexNumber origin, float secondValue) {
        return origin.div(secondValue);
    }

    @Override
    protected float realReal(float firstElement, float secondElement) {
        return firstElement / secondElement;
    }
}
