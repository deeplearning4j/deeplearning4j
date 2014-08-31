package org.deeplearning4j.linalg.ops;


import org.deeplearning4j.linalg.api.ndarray.INDArray;
import org.deeplearning4j.linalg.util.Shape;

import java.util.concurrent.CountDownLatch;

/**
 * Apply an operation and save it to a resulting matrix
 *
 * @author Adam Gibson
 */
public abstract  class BaseTwoArrayElementWiseOp extends BaseElementWiseOp implements TwoArrayElementWiseOp {


    protected INDArray to,other;
    protected INDArray currTo,currOther;


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
        if(scalarValue == null)
            currTo.put(i,apply(getFromDestination(i),i));


        else {
            currTo.put(i,apply(scalarValue,i));

        }
    }

    /**
     * Executes the operation
     * across the matrix
     */
    @Override
    public void exec() {
        if(from != null && to != null && !from.isScalar() && !to.isScalar())
            assert Shape.shapeEquals(from.shape(),to.shape()) : "From and to must be same length";
        if(from != null && other != null && !from.isScalar() && !to.isScalar())
            assert from.length() == other.length() : "From and other must be the same length";

        if(to == null) {
            if(scalarValue != null)
                for(int i = 0; i < from.length(); i++)
                    if(scalarValue != null)
                        applyTransformToOrigin(i);
                    else
                        applyTransformToOrigin(i,scalarValue);
        }
        else {
            assert from.length() == to.length() : "From and to must be same length";

            for(int i = 0; i < from.vectorsAlongDimension(0); i++) {
                INDArray curr = to.vectorAlongDimension(i,0);
                INDArray currOther = other != null ? other.vectorAlongDimension(i,0) : null;
                INDArray fromCurr = from != null ? from.vectorAlongDimension(i,0) : null;
                currTo = curr;
                this.currOther = currOther;
                currVector = fromCurr;
                for(int j = 0; j < fromCurr.length(); j++) {
                    applyTransformToDestination(j);
                }
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
    public INDArray getFromDestination(int i) {
        return currOther.getScalar(i);
    }

    /**
     * The output matrix
     *
     * @return
     */
    @Override
    public INDArray to() {
        return to;
    }
}
