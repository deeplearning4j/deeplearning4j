package org.deeplearning4j.linalg.ops;


import org.deeplearning4j.linalg.api.ndarray.INDArray;

/**
 * An element wise operation over an ndarray.
 *
 *
 *
 * @author Adam Gibson
 */
public interface ElementWiseOp {



    /**
     * The input matrix
     * @return
     */
    public INDArray from();


    /**
     * Apply the transformation at from[i]
     * @param i the index of the element to applyTransformToOrigin
     */
    void applyTransformToOrigin(int i);


    /**
     * Apply the transformation at from[i] using the supplied value (a scalar ndarray)
     * @param i the index of the element to applyTransformToOrigin
     * @param valueToApply the value to apply to the given index
     */
    void applyTransformToOrigin(int i, INDArray valueToApply);

    /**
     * Get the element at from
     * at index i
     * @param i the index of the element to retrieve
     * @return the element at index i
     */
    INDArray getFromOrigin(int i);

    /**
     * The transformation for a given value (a scalar ndarray)
      * @param value the value to applyTransformToOrigin (a scalar ndarray)
     *  @param i the index of the element being acted upon
     * @return the transformed value based on the input
     */

    INDArray apply(INDArray value, int i);

    /**
     * Apply the transformation
     */
    void exec();


}
