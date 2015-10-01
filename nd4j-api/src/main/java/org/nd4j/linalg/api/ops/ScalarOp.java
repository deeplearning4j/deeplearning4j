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

    /**
     * The normal scalar
     *
     * @return the scalar
     */
    Number scalar();

    /**
     * The complex sscalar
     *
     * @return
     */
    IComplexNumber complexScalar();

    ScalarDataBufferAction getScalarOpDataBufferAction(int parallelThreshold, int n, DataBuffer x, DataBuffer z,
                                                       int offsetX, int offsetZ, int incrX, int incrZ);

    ScalarDataBufferAction getScalarOpDataBufferAction(int tensorNum, int tensorDim, int parallelThreshold,
                                                       INDArray x, INDArray z);
}
