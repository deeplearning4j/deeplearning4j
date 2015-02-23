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

package org.nd4j.linalg.api.ops;

import org.nd4j.linalg.api.complex.IComplexNumber;
import org.nd4j.linalg.api.ndarray.INDArray;

/**
 * An op is defined as follows:
 * name: name of the operation
 * x: the origin ndarray
 * y: the ndarray to parse in parallel
 * z: the resulting buffer
 * n: the number of elements to iterate over
 * where x is the origin ndarray,
 * y, is a pairwise op
 * over n elements in the ndarray
 * stored in result z
 *
 * This is followed from the standard template for a BLAS operation
 * such that given a linear buffer, a function defines 3 buffers (x,y,z)
 * and the associated strides and offsets (handled by the ndarrays in this case)
 *
 * @author Adam Gibson
 *
 */
public interface Op {

    /**
     * The name of this operation
     * @return the name of this operation
     */
    String name();

    /**
     * The origin ndarray
     * @return the origin ndarray
     */
    INDArray x();

    /**
     * The pairwise op ndarray
     * @return the pairwise op ndarray
     */
    INDArray y();



    /**
     * The number of elements to do a op over
     * @return the op
     */
    int n();

    /**
     * Pairwise op (applicable with an individual element in y)
     * @param origin the origin number
     * @param other the other number
     * @return the transformed output
     */
    IComplexNumber op(IComplexNumber origin, double other);
    /**
     * Pairwise op (applicable with an individual element in y)
     * @param origin the origin number
     * @param other the other number
     * @return the transformed output
     */
    IComplexNumber op(IComplexNumber origin, float other);

    /**
     * Pairwise op (applicable with an individual element in y)
     * @param origin the origin number
     * @param other the other number
     * @return the transformed output
     */
    IComplexNumber op(IComplexNumber origin, IComplexNumber other);

    /**
     * Pairwise op (applicable with an individual element in y)
     * @param origin the origin number
     * @param other the other number
     * @return the transformed output
     */
    float op(float origin, float other);

    /**
     * Pairwise op (applicable with an individual element in y)
     * @param origin the origin number
     * @param other the other number
     * @return the transformed output
     */
    double op(double origin, double other);

    /**
     * Transform an individual element
     * @param origin the origin element
     * @return the new element
     */
    double op(double origin);

    /**
     * Transform an individual element
     * @param origin the origin element
     * @return the new element
     */
    float op(float origin);

    /**
     * Transform an individual element
     * @param origin the origin element
     * @return the new element
     */
    IComplexNumber op(IComplexNumber origin);




    /**
     * Pairwise op (applicable with an individual element in y)
     * @param origin the origin number
     * @param other the other number
     * @return the transformed output
     */
    IComplexNumber op(IComplexNumber origin, double other,Object[] extraArgs);
    /**
     * Pairwise op (applicable with an individual element in y)
     * @param origin the origin number
     * @param other the other number
     * @return the transformed output
     */
    IComplexNumber op(IComplexNumber origin, float other,Object[] extraArgs);

    /**
     * Pairwise op (applicable with an individual element in y)
     * @param origin the origin number
     * @param other the other number
     * @return the transformed output
     */
    IComplexNumber op(IComplexNumber origin, IComplexNumber other,Object[] extraArgs);

    /**
     * Pairwise op (applicable with an individual element in y)
     * @param origin the origin number
     * @param other the other number
     * @return the transformed output
     */
    float op(float origin, float other, Object[] extraArgs);

    /**
     * Pairwise op (applicable with an individual element in y)
     * @param origin the origin number
     * @param other the other number
     * @return the transformed output
     */
    double op(double origin, double other,Object[] extraArgs);

    /**
     * Transform an individual element
     * @param origin the origin element
     * @return the new element
     */
    double op(double origin,Object[] extraArgs);

    /**
     * Transform an individual element
     * @param origin the origin element
     * @return the new element
     */
    float op(float origin,Object[] extraArgs);

    /**
     * Transform an individual element
     * @param origin the origin element
     * @return the new element
     */
    IComplexNumber op(IComplexNumber origin,Object[] extraArgs);
}
