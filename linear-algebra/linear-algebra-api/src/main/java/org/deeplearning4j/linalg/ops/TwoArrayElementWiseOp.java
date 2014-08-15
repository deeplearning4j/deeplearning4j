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
     * @param i the index of the element to retrieve
     * @return the element at index i
     */
    public INDArray getFromDestination(int i);


    /**
     * Apply the function from to the

     */
    void applyTransformToDestination(int i);

    /**
     * Executes the operation
     * across the matrix
     */
    @Override
    void exec();
}
