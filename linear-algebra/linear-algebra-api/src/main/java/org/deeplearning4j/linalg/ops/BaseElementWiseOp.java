package org.deeplearning4j.linalg.ops;


import org.deeplearning4j.linalg.api.ndarray.INDArray;

/**
 * Baseline element wise operation so only applyTransformToOrigin has to be implemented.
 * This also handles the ability to perform scalar wise operations vs just
 * a functional transformation
 *
 * @author Adam Gibson
 */

public abstract class BaseElementWiseOp implements ElementWiseOp {

    protected INDArray from;
    //this is for operations like adding or multiplying a scalar over the from array
    protected INDArray scalarValue;
    protected INDArray currVector;


    /**
     * Apply the transformation at from[i]
     *
     * @param i the index of the element to applyTransformToOrigin
     */
    @Override
    public void applyTransformToOrigin(int i) {
        currVector.put(i,apply(getFromOrigin(i),i));
    }

    /**
     * Apply the transformation at from[i] using the supplied value
     *
     * @param i            the index of the element to applyTransformToOrigin
     * @param valueToApply the value to apply to the given index
     */
    @Override
    public void applyTransformToOrigin(int i, INDArray valueToApply) {
        currVector.put(i,apply(valueToApply,i));

    }

    @Override
    public INDArray getFromOrigin(int i) {
        return currVector.getScalar(i);
    }

    /**
     * The input matrix
     *
     * @return
     */
    @Override
    public INDArray from() {
        return from;
    }

    /**
     * Apply the transformation
     */
    @Override
    public void exec() {
        for(int i = 0; i < from.vectorsAlongDimension(0); i++) {
            INDArray vectorAlongDim = from.vectorAlongDimension(i,0);
            currVector = vectorAlongDim;
            for(int j = 0; j < vectorAlongDim.length(); j++) {
                 vectorAlongDim.put(j,apply(vectorAlongDim.getScalar(j),j));
            }
        }



    }
}
