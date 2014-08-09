package org.deeplearning4j.nn.linalg.elementwise;

import org.deeplearning4j.nn.linalg.NDArray;

/**
 * Apply an operation and save it to a resulting matrix
 *
 * @author Adam Gibson
 */
public abstract  class BaseTwoArrayElementWiseOp extends BaseElementWiseOp implements TwoArrayElementWiseOp {


    protected NDArray to,other;




    /**
     * This is for scalar operations with an alternative destination matrix
     * @param from the origin matrix
     * @param to the destination matrix
     * @param scalarValue the scalar value to apply
     */
    public BaseTwoArrayElementWiseOp(NDArray from, NDArray to,NDArray other,double scalarValue) {
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
    public BaseTwoArrayElementWiseOp(NDArray from, NDArray to,NDArray other) {
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
    public BaseTwoArrayElementWiseOp(NDArray from, NDArray to,double scalarValue) {
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
    public BaseTwoArrayElementWiseOp(NDArray from, NDArray to) {
        super(from);
        this.to = to;
        this.other = to;
        assert from.length == to.length : "From and to must be the same length";
    }

    public BaseTwoArrayElementWiseOp(NDArray from) {
        super(from);
    }


    protected BaseTwoArrayElementWiseOp(NDArray from, double scalarValue) {
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
        if(Double.isInfinite(scalarValue)) {
            to.put(i,apply(getFromDestination(i),i));

        }
        else {
            to.put(i,apply(scalarValue,i));

        }
    }

    /**
     * Executes the operation
     * across the matrix
     */
    @Override
    public void exec() {
        if(to == null) {
            if(!Double.isInfinite(scalarValue))
                for(int i = 0; i < from.length; i++)
                    if(Double.isInfinite(scalarValue))
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
    public double getFromDestination(int i) {
        return other.get(i);
    }

    /**
     * The output matrix
     *
     * @return
     */
    @Override
    public NDArray to() {
        return to;
    }
}
