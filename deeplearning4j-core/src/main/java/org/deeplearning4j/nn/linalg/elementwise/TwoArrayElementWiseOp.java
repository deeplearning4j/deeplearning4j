package org.deeplearning4j.nn.linalg.elementwise;

import org.deeplearning4j.nn.linalg.NDArray;

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
    public NDArray to();


    /**
     * Returns the element
     * in destination at index i
     * @param i the index of the element to retrieve
     * @return the element at index i
     */
    public double getFromDestination(int i);


    /**
     * Apply the function from to the

     */
    void applyTransformToDestination(int i);

    /**
     * Executes the operation
     * across the matrix
     */
    void exec();
}
