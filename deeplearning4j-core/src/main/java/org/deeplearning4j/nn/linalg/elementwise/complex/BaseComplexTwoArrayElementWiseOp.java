package org.deeplearning4j.nn.linalg.elementwise.complex;

import org.deeplearning4j.nn.linalg.ComplexNDArray;
import org.deeplearning4j.nn.linalg.NDArray;
import org.jblas.ComplexDouble;

/**
 * Base class for complex nd array element wise ops
 */
public abstract class BaseComplexTwoArrayElementWiseOp extends BaseComplexElementWiseOp implements ComplexTwoArrayElementWiseOp {


    protected ComplexNDArray to,other;




    /**
     * This is for scalar operations with an alternative destination matrix
     * @param from the origin matrix
     * @param to the destination matrix
     * @param scalarValue the scalar value to apply
     */
    public BaseComplexTwoArrayElementWiseOp(ComplexNDArray from, ComplexNDArray to,ComplexNDArray other,ComplexDouble scalarValue) {
        super(from,scalarValue);
        this.to = to;
        this.other = other;
        assert from.length == to.length : "From and to must be the same length";
    }

    /**
     * This is for matrix <-> matrix operations
     * @param from the origin matrix
     * @param to the destination matrix
     */
    public BaseComplexTwoArrayElementWiseOp(ComplexNDArray from, ComplexNDArray to,ComplexNDArray other) {
        super(from);
        this.to = to;
        this.other = other;
        assert from.length == to.length : "From and to must be the same length";
    }


    /**
     * This is for scalar operations with an alternative destination matrix
     * @param from the origin matrix
     * @param to the destination matrix
     * @param scalarValue the scalar value to apply
     */
    public BaseComplexTwoArrayElementWiseOp(ComplexNDArray from, ComplexNDArray to,ComplexDouble scalarValue) {
        super(from,scalarValue);
        this.to = to;
        this.other = to;
        assert from.length == to.length : "From and to must be the same length";
    }

    /**
     * This is for matrix <-> matrix operations
     * @param from the origin matrix
     * @param to the destination matrix
     */
    public BaseComplexTwoArrayElementWiseOp(ComplexNDArray from, ComplexNDArray to) {
        super(from);
        this.to = to;
        this.other = to;
        assert from.length == to.length : "From and to must be the same length";
    }

    public BaseComplexTwoArrayElementWiseOp(ComplexNDArray from) {
        super(from);
    }


    protected BaseComplexTwoArrayElementWiseOp(ComplexNDArray from, ComplexDouble scalarValue) {
        super(from, scalarValue);
    }

    /**
     * Apply the function from to the specified index
     * in to. The value from to is passed in to apply
     * and then a transform of the matching elements in
     * both from and to are used for a transformation.
     *
     * If a scalar is specified, this will apply a scalar wise operation
     * based on the scalar and the origin matrix instead
     * @param i the index to apply to
     */
    @Override
    public void applyTransformToDestination(int i) {
        if(Double.isInfinite(scalarValue.real())) {
            ComplexDouble ret =  apply(getFromDestination(i),i);
            to.data[to.unSafeLinearIndex(i)] = ret.real();
            to.data[to.unSafeLinearIndex(i) + 1] = ret.imag();

        }
        else {
            ComplexDouble c = apply(scalarValue,i);
            to.data[to.unSafeLinearIndex(i)] = c.real();
            to.data[to.unSafeLinearIndex(i) + 1] = c.imag();


        }
    }

    /**
     * Executes the operation
     * across the matrix
     */
    @Override
    public void exec() {
        if(to == null) {
            if(!Double.isInfinite(scalarValue.real()))
                for(int i = 0; i < from.length; i++)
                    if(Double.isInfinite(scalarValue.real()))
                        applyTransformToOrigin(i);
                    else applyTransformToOrigin(i,scalarValue);
        }
        else {
            for(int i = 0; i < to.length; i++) {
                applyTransformToDestination(i);
            }
        }

    }

    /**
     * Returns the element
     * in destination at index i
     *
     * @param i the index of the element to retrieve
     * @return the element at index i
     */
    @Override
    public ComplexDouble getFromDestination(int i) {
        return new ComplexDouble(other.data[other.unSafeLinearIndex(i)],other.data[other.unSafeLinearIndex(i) + 1]);
    }

    /**
     * The output matrix
     *
     * @return
     */
    @Override
    public ComplexNDArray to() {
        return to;
    }

}
