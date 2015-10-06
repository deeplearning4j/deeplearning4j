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
import org.nd4j.linalg.api.parallel.bufferops.ScalarDataBufferAction;

/**
 * @author Adam Gibson
 */
public interface ScalarOp extends Op {

    /**The normal scalar
     *@return the scalar
     */
    Number scalar();

    /**The complex scalar
     *@return
     */
    IComplexNumber complexScalar();

    /**Get a ScalarDataBufferAction. Used (internally) for parallelizing Scalar operations.
     * Most ScalarOp implementation scan simply inherit BaseScalarOp.getScalarOpDataBufferAction
     * @param parallelThreshold The parallelism threshold for breaking a ScalarDataBufferAction into smaller sub-tasks
     *                          for parallel execution
     * @param n The number of operations / length
     * @param x x DataBuffer
     * @param z z DataBuffer (may be identical to x, for x=Op(x), or different for z=Op(x))
     * @param offsetX Offset of the first element to be processed in the X databuffer
     * @param offsetZ Offset of the first element to be processed in the Z databuffer
     * @param incrX Increment between elements in the X DataBuffer
     * @param incrZ Increment between elements in the Z DataBuffer
     * @return The ScalarDataBufferAction
     */
    ScalarDataBufferAction getScalarOpDataBufferAction(int parallelThreshold, int n, DataBuffer x, DataBuffer z,
                                                       int offsetX, int offsetZ, int incrX, int incrZ);

    /**Get a ScalarDataBufferAction. Used (internally) for parallelizing Scalar operations
     * Unlike the other getScalarOpDataBufferAction, the resulting ScalarDataBufferAction should
     * calculate a 1d tensor first
     * Most ScalarOp implementation scan simply inherit BaseScalarOp.getScalarOpDataBufferAction
     * @param tensorNum The number/index of the 1d tensor that should be calculated (i.e., x.tensorAlongDimension(tensorNum,tensorDim))
     * @param tensorDim The dimension that the 1d tensor should be calculated along
     * @param parallelThreshold The threshold for parallelism (for breaking an ScalarOp into smaller parallel ops) for
     *                          parallel execution
     * @param x x INDArray
     * @param z z INDArray (result, may be identical to x for x=Op(x), or different for z=Op(x))
     * @return The ScalarDataBufferAction
     */
    ScalarDataBufferAction getScalarOpDataBufferAction(int tensorNum, int tensorDim, int parallelThreshold,
                                                       INDArray x, INDArray z);
}
