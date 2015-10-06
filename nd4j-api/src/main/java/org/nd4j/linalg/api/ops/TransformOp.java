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
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.parallel.bufferops.TransformDataBufferAction;

/**
 * Transform operation:
 * stores the result in an ndarray
 *
 * @author Adam Gibson
 */
public interface TransformOp extends Op {

    /**
     * The derivative operation for this op
     *
     * @return the derivative operation
     * for this op's inputs
     */
    TransformOp derivative();

    /** Get a TransformDataBufferAction. Used (internally) for parallelizing Transform operations.
     * Note that most TransformOp implementations can simply inherit BaseTransformOp.getTransformOpDataBufferTask,
     * however this can be overridden in certain cases for performance reasons
     * @param parallelThreshold The parallelism threshold for breaking an TransformDataBufferAction into smaller
     *                          sub-tasks for parallel execution
     * @param n The number of operations / length
     * @param x x DataBuffer
     * @param y y DataBuffer (may be null for x=Op(x) or z=Op(x) transform ops)
     * @param z z DataBuffer (may be identical to x)
     * @param offsetX Offset of the first element to be processed in the X databuffer
     * @param offsetY Offset of the first element to be processed in the Y databuffer (not used if y is null)
     * @param offsetZ Offset of the first element to be processed in the Z databuffer
     * @param incrX Increment between elements in the X databuffer
     * @param incrY Increment between elements in the Y databuffer (not used if y is null)
     * @param incrZ Increment between elements in the Z databuffer
     * @return The TransformDataBufferAction
     */
    TransformDataBufferAction getTransformOpDataBufferAction( int parallelThreshold, int n, DataBuffer x, DataBuffer y,
                                                                   DataBuffer z, int offsetX, int offsetY, int offsetZ,
                                                                   int incrX, int incrY, int incrZ);

    /**Get a TransformDataBufferAction. Used (internall) for parallelizing Transform operations.
     * Unlike the other getTransformOpDataBufferAction, the resulting TransformDataBufferAction should
     * calculate a 1d tensor first.
     * Note that most TransformOp implementations can simply inherit BaseTransformOp.getTransformOpDataBufferTask,
     * however this can be overridden in certain cases for performance reasons
     * @param tensorNum The number/index of the 1d tensor that should be calculated (i.e., x.tensorAlongDimension(tensorNum,tensorDim))
     * @param tensorDim The dimension that the 1d tensor should be calculated along
     * @param parallelThreshold The threshold for parallelism (for breaking an TrasformOp into smaller ops to be executed
     *                          in parallel)
     * @param x x INDArray
     * @param y y INDArray (may be null for x=Op(x) and z=Op(x) TransformOps)
     * @param z z INDArray
     * @return The TransformDataBufferAction
     */
    TransformDataBufferAction getTransformOpDataBufferAction( int tensorNum, int tensorDim, int parallelThreshold,
                                                              INDArray x, INDArray y, INDArray z);
}
