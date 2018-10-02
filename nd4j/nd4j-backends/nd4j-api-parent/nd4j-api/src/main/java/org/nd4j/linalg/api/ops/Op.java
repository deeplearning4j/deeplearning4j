/*******************************************************************************
 * Copyright (c) 2015-2018 Skymind, Inc.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

package org.nd4j.linalg.api.ops;

import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.nio.Buffer;

/**
 * An op is defined as follows:
 * opName: opName of the operation
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
    enum Type {
        SCALAR,
        TRANSFORM_SAME,
        TRANSFORM_FLOAT,
        TRANSFORM_BOOL,
        TRANSFORM_STRICT,
        PAIRWISE,
        SPECIAL,
        BROADCAST,
        REDUCE_LONG,
        REDUCE_SAME,
        REDUCE_FLOAT,
        REDUCE_BOOL,
        INDEXREDUCE,
        VARIANCE,
        REDUCE3,
        GRID,
        META,
        AGGREGATION,
        CUSTOM,
        GRADIENT,
        SHAPE,
        CONDITIONAL,
        LOOP,
        LOOP_COND,
        IF,
        RETURN,
        ENTER,
        EXIT,
        NEXT_ITERATION,
        RANDOM,
        MERGE,
        SUMMARYSTATS,
    }

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
     * @return  a buffer of either opType float or double
     * representing the extra args for this op
     */
    Buffer extraArgsBuff();

    /**
     * An op number
     * @return
     */
    int opNum();

    /**
     * The opName of this operation
     *
     * @return the opName of this operation
     */
    String opName();

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
    long n();



    /**
     * Initialize the operation based on the parameters
     *
     * @param x the input
     * @param y the pairwise transform ndarray
     * @param z the resulting ndarray
     * @param n the number of elements
     */
    void init(INDArray x, INDArray y, INDArray z, long n);

    /**
     * Number processed
     *
     * @return the number of elements accumulated
     */
    long numProcessed();

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
    void exec(int... dimensions);

    /**
     * Change n
     * @param n
     */
    void setN(long n);

    /**
     *
     * @param extraArgs
     */
    void setExtraArgs(Object[] extraArgs);

    /**
     * Converts this op to be a {@link CustomOp}
     * A {@link CustomOp} is a more flexible op
     * meant for multiple inputs and outputs.
     * The default implementation in {@link BaseOp}
     * converts a simple op to a multi input/output operation
     * by mapping the x and y on to inputs , the op opName
     * and the z on to outputs.
     * @return the equivalent {@link CustomOp}
     */
    CustomOp toCustomOp();

}
