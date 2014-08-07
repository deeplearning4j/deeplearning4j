package org.deeplearning4j.nn.linalg.elementwise;

import org.deeplearning4j.nn.linalg.NDArray;

/**
 * Baseline element wise operation so only applyTransformToOrigin has to be implemented.
 * This also handles the ability to perform scalar wise operations vs just
 * a functional transformation
 *
 * @author Adam Gibson
 */

public abstract class BaseElementWiseOp implements ElementWiseOp {

    protected NDArray from;
    //this is for operations like adding or multiplying a scalar over the from array
    protected double scalarValue = Double.NEGATIVE_INFINITY;

    protected BaseElementWiseOp(NDArray from) {
        this.from = from;
    }

    protected BaseElementWiseOp(NDArray from,double scalarValue) {
        this.from = from;
        this.scalarValue = scalarValue;
    }


    /**
     * Apply the transformation at from[i]
     *
     * @param i the index of the element to applyTransformToOrigin
     */
    @Override
    public void applyTransformToOrigin(int i) {
        from.data[from.unSafeLinearIndex(i)] = apply(getFromOrigin(i),i);
    }

    /**
     * Apply the transformation at from[i] using the supplied value
     *
     * @param i            the index of the element to applyTransformToOrigin
     * @param valueToApply the value to apply to the given index
     */
    @Override
    public void applyTransformToOrigin(int i, double valueToApply) {
        from.data[from.unSafeLinearIndex(i)] = apply(valueToApply,i);

    }

    @Override
    public double getFromOrigin(int i) {
        return from.data[from.unSafeLinearIndex(i)];
    }

    /**
     * The input matrix
     *
     * @return
     */
    @Override
    public NDArray from() {
        return from;
    }
}
