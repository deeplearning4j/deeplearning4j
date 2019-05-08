/*******************************************************************************
 * Copyright (c) 2015-2019 Skymind, Inc.
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
import org.nd4j.linalg.api.buffer.DataType;
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
        SCALAR_BOOL,
        TRANSFORM_SAME,
        TRANSFORM_FLOAT,
        TRANSFORM_ANY,
        TRANSFORM_BOOL,
        TRANSFORM_STRICT,
        PAIRWISE,
        PAIRWISE_BOOL,
        SPECIAL,
        BROADCAST,
        BROADCAST_BOOL,
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
     * Returns the extra args as a data buffer
     * @return
     */
    DataBuffer extraArgsDataBuff(DataType bufferType);

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
