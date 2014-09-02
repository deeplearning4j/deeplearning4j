package org.deeplearning4j.linalg.ops;


import org.deeplearning4j.linalg.api.ndarray.INDArray;

/**
 *
 * An extension of the element wise operations applying
 * the element transform to a destination matrix
 * with respect to an origin matrix.
 *
 * @author Adam Gibson
 */
public interface TwoArrayElementWiseOp extends ElementWiseOp {

    /**
     * The output matrix
     * @return
     */
    public INDArray to();


    /**
     * Returns the element
     * in destination at index i
     * @param destination the destination ndarray
     * @param i the index of the element to retrieve
     * @return the element at index i
     */
    public <E> E getOther(INDArray destination, int i);


    /**
     *
     * Apply a transform
     * based on the passed in ndarray to other
     * @param from the origin ndarray
     * @param destination the destination ndarray
     * @param other the other ndarray
     * @param i the index of the element to retrieve
     */
    void applyTransformToDestination(INDArray from,INDArray destination,INDArray other,int i);

    /**
     * Executes the operation
     * across the matrix
     */
    @Override
    void exec();
}
