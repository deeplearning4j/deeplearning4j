/*
 *
 *  * Copyright 2015 Skymind,Inc.
 *  *
 *  *    Licensed under the Apache License, Version 2.0 (the "License");
 *  *    you may not use this file except in compliance with the License.
 *  *    You may obtain a copy of the License at
 *  *
 *  *        http://www.apache.org/licenses/LICENSE-2.0
 *  *
 *  *    Unless required by applicable law or agreed to in writing, software
 *  *    distributed under the License is distributed on an "AS IS" BASIS,
 *  *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  *    See the License for the specific language governing permissions and
 *  *    limitations under the License.
 *
 *
 */

package org.nd4j.linalg.api.ops;

import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.complex.IComplexNumber;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.nio.Buffer;

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
 * <p/>
 * This is followed from the standard template for a BLAS operation
 * such that given a linear buffer, a function defines 3 buffers (x,y,z)
 * and the associated strides and offsets (handled by the ndarrays in this case)
 *
 * @author Adam Gibson
 */
public interface Op {

    /**
     * Whether the executioner
     * needs to do a special call or not
     * @return true if the executioner needs to do a special
     * call or not false otherwise
     */
    boolean isExecSpecial();

    /**
     * Returns the extra args as a data buffer
     * @return
     */
    DataBuffer extraArgsDataBuff();
    /**
     * Returns a buffer of either float
     * or double
     * of the extra args for this buffer
     * @return  a buffer of either type float or double
     * representing the extra args for this op
     */
    Buffer extraArgsBuff();
    /**
     * An op number
     * @return
     */
    int opNum();

    /**
     * The name of this operation
     *
     * @return the name of this operation
     */
    String name();

    /**
     * The origin ndarray
     *
     * @return the origin ndarray
     */
    INDArray x();

    /**
     * The pairwise op ndarray
     *
     * @return the pairwise op ndarray
     */
    INDArray y();

    /**
     * The resulting ndarray
     *
     * @return the resulting ndarray
     */
    INDArray z();



    /**
     * The number of elements to do a op over
     *
     * @return the op
     */
    int n();

    /**
     * Pairwise op (applicable with an individual element in y)
     *
     * @param origin the origin number
     * @param other  the other number
     * @return the transformed output
     */
    IComplexNumber op(IComplexNumber origin, double other);

    /**
     * Pairwise op (applicable with an individual element in y)
     *
     * @param origin the origin number
     * @param other  the other number
     * @return the transformed output
     */
    IComplexNumber op(IComplexNumber origin, float other);

    /**
     * Pairwise op (applicable with an individual element in y)
     *
     * @param origin the origin number
     * @param other  the other number
     * @return the transformed output
     */
    IComplexNumber op(IComplexNumber origin, IComplexNumber other);

    /**
     * Pairwise op (applicable with an individual element in y)
     *
     * @param origin the origin number
     * @param other  the other number
     * @return the transformed output
     */
    float op(float origin, float other);

    /**
     * Pairwise op (applicable with an individual element in y)
     *
     * @param origin the origin number
     * @param other  the other number
     * @return the transformed output
     */
    double op(double origin, double other);

    /**
     * Transform an individual element
     *
     * @param origin the origin element
     * @return the new element
     */
    double op(double origin);

    /**
     * Transform an individual element
     *
     * @param origin the origin element
     * @return the new element
     */
    float op(float origin);

    /**
     * Transform an individual element
     *
     * @param origin the origin element
     * @return the new element
     */
    IComplexNumber op(IComplexNumber origin);


    /**
     * A copy of this operation for a particular dimension of the input
     *
     * @param index     the index of the op to iterate over
     * @param dimension the dimension to ge the input for
     * @return the operation for that dimension
     */
    Op opForDimension(int index, int dimension);

    /**
     * A copy of this operation for a particular dimension of the input
     *
     * @param index     the index of the op to iterate over
     * @param dimension the dimension to ge the input for
     * @return the operation for that dimension
     */
    Op opForDimension(int index, int...dimension);

    /**
     * Initialize the operation based on the parameters
     *
     * @param x the input
     * @param y the pairwise transform ndarray
     * @param z the resulting ndarray
     * @param n the number of elements
     */
    void init(INDArray x, INDArray y, INDArray z, int n);

    /**
     * Number processed
     *
     * @return the number of elements accumulated
     */
    int numProcessed();

    /**
     * Extra arguments
     *
     * @return the extra arguments
     */
    Object[] extraArgs();


    /**
     * set x (the input ndarray)
     * @param x
     */
    void setX(INDArray x);

    /**
     * set z (the solution ndarray)
     * @param z
     */
    void setZ(INDArray z);

    /**
     * set y(the pairwise ndarray)
     * @param y
     */
    void setY(INDArray y);

    /**
     * Returns whether the op should be executed or not (through the executioner)
     *
     * @return true if the op is pass through false otherwise
     */
    boolean isPassThrough();

    /**
     * Execute the op if its pass through (not needed most of the time)
     */
    void exec();

    /**
     * Exec along each dimension
     * @param dimensions the dimensions to execute on
     */
    void exec(int...dimensions);

    /**
     * Change n
     * @param n
     */
    void setN(int n);

}
