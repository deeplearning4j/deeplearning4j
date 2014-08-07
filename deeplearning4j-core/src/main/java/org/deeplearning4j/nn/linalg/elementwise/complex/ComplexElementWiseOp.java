package org.deeplearning4j.nn.linalg.elementwise.complex;

import com.google.common.base.Function;
import org.deeplearning4j.nn.linalg.ComplexNDArray;
import org.jblas.ComplexDouble;

/**
 * A linear element wise operation over a complex ndarray
 */
public interface ComplexElementWiseOp {


    /**
     * The input matrix
     * @return
     */
    public ComplexNDArray from();


    /**
     * Apply the transformation at from[i]
     * @param i the index of the element to applyTransformToOrigin
     */
    void applyTransformToOrigin(int i);


    /**
     * Apply the transformation at from[i] using the supplied value
     * @param i the index of the element to applyTransformToOrigin
     * @param valueToApply the value to apply to the given index
     */
    void applyTransformToOrigin(int i, ComplexDouble valueToApply);

    /**
     * Get the element at from
     * at index i
     * @param i the index of the element to retrieve
     * @return the element at index i
     */
    ComplexDouble getFromOrigin(int i);

    /**
     * The transformation for a given value
     * @param value the value to applyTransformToOrigin
     *  @param i the index of the element being acted upon
     * @return the transformed value based on the input
     */

    ComplexDouble apply(ComplexDouble value,int i);

}
