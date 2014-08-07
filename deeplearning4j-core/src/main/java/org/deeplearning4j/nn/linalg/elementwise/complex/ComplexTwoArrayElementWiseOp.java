package org.deeplearning4j.nn.linalg.elementwise.complex;

import org.deeplearning4j.nn.linalg.ComplexNDArray;
import org.jblas.ComplexDouble;

/**
 * Created by agibsonccc on 8/6/14.
 */
public interface ComplexTwoArrayElementWiseOp extends  ComplexElementWiseOp {



    /**
     * The output matrix
     * @return
     */
    public ComplexNDArray to();


    /**
     * Returns the element
     * in destination at index i
     * @param i the index of the element to retrieve
     * @return the element at index i
     */
    public ComplexDouble getFromDestination(int i);


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
