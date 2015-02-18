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
 * An extension of the element wise operations applying
 * the element transform to a destination matrix
 * with respect to an origin matrix.
 *
 * @author Adam Gibson
 */
public interface TwoArrayElementWiseOp extends ElementWiseOp {

    /**
     * The output matrix
     *
     * @return
     */
    public INDArray to();


    /**
     * Returns the element
     * in destination at index i
     *
     * @param destination the destination ndarray
     * @param i           the index of the element to retrieve
     * @return the element at index i
     */
    public <E> E getOther(INDArray destination, int i);


    /**
     * Apply a transform
     * based on the passed in ndarray to other
     *
     * @param from        the origin ndarray
     * @param destination the destination ndarray
     * @param other       the other ndarray
     * @param i           the index of the element to retrieve
     */
    void applyTransformToDestination(INDArray from, INDArray destination, INDArray other, int i);

    /**
     * Executes the operation
     * across the matrix
     */
    @Override
    void exec();
}
