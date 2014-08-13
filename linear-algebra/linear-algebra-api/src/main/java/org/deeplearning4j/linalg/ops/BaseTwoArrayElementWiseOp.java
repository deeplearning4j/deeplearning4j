package org.deeplearning4j.linalg.ops;


import org.deeplearning4j.linalg.api.ndarray.INDArray;

/**
 * Apply an operation and save it to a resulting matrix
 *
 * @author Adam Gibson
 */
public abstract  class BaseTwoArrayElementWiseOp extends BaseElementWiseOp implements TwoArrayElementWiseOp {


    protected INDArray to,other;



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
        if(scalarValue != null) {
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
            if(scalarValue != null)
                for(int i = 0; i < from.length(); i++)
                    if(scalarValue != null)
                        applyTransformToOrigin(i);
                    else applyTransformToOrigin(i,scalarValue);
        }
        else {
            for(int i = 0; i < to.length(); i++) {
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
    public INDArray getFromDestination(int i) {
        return other.getScalar(i);
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
