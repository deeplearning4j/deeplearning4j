/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  *  See the NOTICE file distributed with this work for additional
 *  *  information regarding copyright ownership.
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

package org.nd4j.linalg.api.ops;

import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;

public interface ReduceOp extends Op {

    /**
     * Returns the no op version
     * of the input
     * Basically when a reduce can't happen (eg: sum(0) on a row vector)
     * you have a no op state for a given reduction.
     * For most accumulations, this should return x
     * but certain transformations should return say: the absolute value
     *
     *
     * @return the no op version of the input
     */
    INDArray noOp();

    /**
     * This method returns dimensions for this op
     * @return
     */
    INDArray dimensions();

    @Deprecated
    boolean isComplexAccumulation();

    Type getOpType();

    /**
     * This method returns TRUE if we're going to keep axis, FALSE otherwise
     *
     * @return
     */
    boolean isKeepDims();

    /**
     * This method returns datatype for result array wrt given inputs
     * @return
     */
    DataType resultType();

    DataType resultType(OpContext oc);

    boolean validateDataTypes(OpContext oc);

    Number getFinalResult();

    void setDimensions(int... dimensions);
}
