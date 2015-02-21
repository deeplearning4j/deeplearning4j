/*
 * Copyright 2015 Skymind,Inc.
 *
 *    Licensed under the Apache License, Version 2.0 (the "License");
 *    you may not use this file except in compliance with the License.
 *    You may obtain a copy of the License at
 *
 *        http://www.apache.org/licenses/LICENSE-2.0
 *
 *    Unless required by applicable law or agreed to in writing, software
 *    distributed under the License is distributed on an "AS IS" BASIS,
 *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *    See the License for the specific language governing permissions and
 *    limitations under the License.
 */

package org.nd4j.linalg.ops;


import org.nd4j.linalg.api.ndarray.INDArray;

/**
 * An element wise operation over an ndarray.
 *
 * @author Adam Gibson
 */
public interface ElementWiseOp {


    /**
     * Setter for extra arguments
     * @param args the extra arguments to set
     */
    void setExtraArgs(Object[] args);

    /**
     * Extra arguments to the element wise operation
     * @return the operation's extra arguments
     */
    Object[] extraArgs();

    /**
     *
     * The name of the function
     * @return the name of the function
     */
    String name();

    /**
     * The input matrix
     *
     * @return
     */
    public INDArray from();


    /**
     * Apply the transformation at from[i]
     *
     * @param origin the origin ndarray
     * @param i      the index of the element to applyTransformToOrigin
     */
    void applyTransformToOrigin(INDArray origin, int i);


    /**
     * Apply the transformation at from[i] using the supplied value (a scalar ndarray)
     *
     * @param origin       the origin ndarray
     * @param i            the index of the element to applyTransformToOrigin
     * @param valueToApply the value to apply to the given index
     */
    void applyTransformToOrigin(INDArray origin, int i, Object valueToApply);

    /**
     * Get the element at from
     * at index i
     *
     * @param origin the origin ndarray
     * @param i      the index of the element to retrieve
     * @return the element at index i
     */
    <E> E getFromOrigin(INDArray origin, int i);

    /**
     * The transformation for a given value (a scalar)
     *
     * @param origin the origin ndarray
     * @param value  the value to apply (a scalar)
     * @param i      the index of the element being acted upon
     * @return the transformed value based on the input
     */

    <E> E apply(INDArray origin, Object value, int i);


    /**
     * Apply the transformation
     */
    void exec();


}
