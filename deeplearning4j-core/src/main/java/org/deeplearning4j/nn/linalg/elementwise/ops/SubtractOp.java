package org.deeplearning4j.nn.linalg.elementwise.ops;

import org.deeplearning4j.nn.linalg.NDArray;
import org.deeplearning4j.nn.linalg.elementwise.BaseTwoArrayElementWiseOp;

/**
 * Add a scalar or a matrix
 *
 * @author Adam Gibson
 */
public class SubtractOp extends BaseTwoArrayElementWiseOp {

    /**
     * This is for scalar operations with an alternative destination matrix
     *
     * @param from        the origin matrix
     * @param to          the destination matrix
     * @param other
     * @param scalarValue the scalar value to apply
     */
    public SubtractOp(NDArray from, NDArray to, NDArray other, double scalarValue) {
        super(from, to, other, scalarValue);
    }

    /**
     * This is for matrix <-> matrix operations
     *
     * @param from  the origin matrix
     * @param to    the destination matrix
     * @param other
     */
    public SubtractOp(NDArray from, NDArray to, NDArray other) {
        super(from, to, other);
    }

    /**
     * This is for scalar operations with an alternative destination matrix
     *
     * @param from        the origin matrix
     * @param to          the destination matrix
     * @param scalarValue the scalar value to apply
     */
    public SubtractOp(NDArray from, NDArray to, double scalarValue) {
        super(from, to, scalarValue);
    }

    public SubtractOp(NDArray from) {
        super(from);
    }

    /**
     * Add two ndarrays
     * @param from the element
     * @param to
     */
    public SubtractOp(NDArray from, NDArray to) {
        super(from, to);
    }

    public SubtractOp(NDArray from, double scalarValue) {
        super(from, scalarValue);
    }

    /**
     * The transformation for a given value
     *
     * @param value the value to applyTransformToOrigin
     * @return the transformed value based on the input
     */
    @Override
    public double apply(double value,int i) {
        return getFromOrigin(i) - value;
    }


}
