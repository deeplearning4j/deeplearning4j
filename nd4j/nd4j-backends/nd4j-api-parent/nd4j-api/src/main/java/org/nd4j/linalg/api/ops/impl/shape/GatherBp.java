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

package org.nd4j.linalg.api.ops.impl.shape;

import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.DynamicCustomOp;

import java.util.Collections;
import java.util.List;

/**
 * Native backward pass for the Gather op.
 *
 * <p>Contract:
 * <ul>
 *   <li>inputs[0] = {@code inputShapeVec} – 1D integer array whose VALUES are the
 *       forward input's shape dims (e.g. from {@code sameDiff.shape(forwardInput)} or
 *       {@code forwardInput.shape()}).  Drives the value-dependent output shape.</li>
 *   <li>inputs[1] = {@code indices}       – integer gather indices</li>
 *   <li>inputs[2] = {@code gradOut}       – upstream gradient (same shape as gather's output)</li>
 *   <li>IArgs[0]  = {@code axis}          – gather axis (may be negative; normalised in C++)</li>
 *   <li>output[0] = {@code dInput}        – gradient w.r.t. input;
 *       shape = inputShapeVec values, dtype = gradOut dtype</li>
 * </ul>
 *
 * <p>The output shape is determined at runtime from the VALUES of {@code inputShapeVec},
 * so this op works for any input shape — including fully dynamic shapes that are not
 * statically known at graph-build time.
 *
 * <p>Duplicate gather indices accumulate (sum) — correct gradient semantics.
 * On CUDA accumulation uses {@code sd_atomicAdd}; on CPU it is single-threaded.
 *
 * <p>This is a backward primitive: {@code doDiff} is not implemented.
 */
public class GatherBp extends DynamicCustomOp {

    /** No-arg constructor required for op-registry reflection. */
    public GatherBp() {}

    /**
     * SameDiff (symbolic) constructor.
     *
     * @param sd             the SameDiff graph
     * @param inputShapeVec  1D SDVariable whose runtime VALUES are the forward input's shape
     *                       (obtain via {@code forwardInput.shape()} or
     *                       {@code sameDiff.shape(forwardInput)})
     * @param indices        gather-index variable (integer dtype)
     * @param gradOut        upstream gradient variable
     * @param axis           gather axis (may be negative; normalised in C++)
     */
    public GatherBp(SameDiff sd, SDVariable inputShapeVec, SDVariable indices,
                    SDVariable gradOut, long axis) {
        super(sd, new SDVariable[]{inputShapeVec, indices, gradOut});
        addIArgument(axis);
    }

    /**
     * Eager (INDArray) constructor.
     *
     * @param inputShapeVec  1D integer array whose values are the forward input's shape
     * @param indices        gather-index array (integer dtype)
     * @param gradOut        upstream gradient array
     * @param axis           gather axis
     */
    public GatherBp(INDArray inputShapeVec, INDArray indices, INDArray gradOut, long axis) {
        super(new INDArray[]{inputShapeVec, indices, gradOut}, null);
        addIArgument(axis);
    }

    @Override
    public String opName() {
        return "gather_bp";
    }

    /**
     * Output dtype = dtype of {@code gradOut} (inputs[2]).
     */
    @Override
    public List<DataType> calculateOutputDataTypes(List<DataType> dataTypes) {
        // dataTypes[0] = inputShapeVec dtype (int)
        // dataTypes[1] = indices dtype (int)
        // dataTypes[2] = gradOut dtype (float)
        return Collections.singletonList(dataTypes.get(2));
    }
}
