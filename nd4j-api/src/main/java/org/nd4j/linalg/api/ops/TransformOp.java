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

    TransformDataBufferAction getTransformOpDataBufferAction( int parallelThreshold, int n, DataBuffer x, DataBuffer y,
                                                                   DataBuffer z, int offsetX, int offsetY, int offsetZ,
                                                                   int incrX, int incrY, int incrZ);

    TransformDataBufferAction getTransformOpDataBufferAction( int tensorNum, int tensorDim, int parallelThreshold,
                                                              INDArray x, INDArray y, INDArray z);

}
