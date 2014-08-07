package org.deeplearning4j.nn.linalg.elementwise.complex.ops;

import org.deeplearning4j.nn.linalg.ComplexNDArray;
import org.deeplearning4j.nn.linalg.elementwise.complex.BaseComplexTwoArrayElementWiseOp;
import org.jblas.ComplexDouble;

/**
 * Add a scalar or a matrix
 *
 * @author Adam Gibson
 */
public class MultiplyOp extends BaseComplexTwoArrayElementWiseOp {
    /**
     * This is for scalar operations with an alternative destination matrix
     *
     * @param from        the origin matrix
     * @param to          the destination matrix
     * @param other
     * @param scalarValue the scalar value to apply
     */
    public MultiplyOp(ComplexNDArray from, ComplexNDArray to, ComplexNDArray other, ComplexDouble scalarValue) {
        super(from, to, other, scalarValue);
    }

    /**
     * This is for matrix <-> matrix operations
     *
     * @param from  the origin matrix
     * @param to    the destination matrix
     * @param other
     */
    public MultiplyOp(ComplexNDArray from, ComplexNDArray to, ComplexNDArray other) {
        super(from, to, other);
    }

    /**
     * This is for scalar operations with an alternative destination matrix
     *
     * @param from        the origin matrix
     * @param to          the destination matrix
     * @param scalarValue the scalar value to apply
     */
    public MultiplyOp(ComplexNDArray from, ComplexNDArray to, ComplexDouble scalarValue) {
        super(from, to, scalarValue);
    }

    public MultiplyOp(ComplexNDArray from) {
        super(from);
    }

    /**
     * Add two ComplexNDArrays
     * @param from the element
     * @param to
     */
    public MultiplyOp(ComplexNDArray from, ComplexNDArray to) {
        super(from, to);
    }

    public MultiplyOp(ComplexNDArray from, ComplexDouble scalarValue) {
        super(from, scalarValue);
    }

    /**
     * The transformation for a given value
     *
     * @param value the value to applyTransformToOrigin
     * @return the transformed value based on the input
     */
    @Override
    public ComplexDouble apply(ComplexDouble value,int i) {
        return value.mul(getFromOrigin(i));
    }


}
