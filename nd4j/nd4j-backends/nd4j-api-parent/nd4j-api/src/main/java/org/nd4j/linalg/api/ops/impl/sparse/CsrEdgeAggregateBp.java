/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  * See the NOTICE file distributed with this work for additional
 *  * information regarding copyright ownership.
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

package org.nd4j.linalg.api.ops.impl.sparse;

import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.DynamicCustomOp;

import java.util.Collections;
import java.util.List;

/**
 * Backward pass for {@link CsrEdgeAggregate}: route per-node gradients back to edges.
 *
 * <p>C++ op name: {@code csr_edge_aggregate_bp}
 * <ul>
 *   <li>Inputs:  rowPtr[rows+1] (INT32), edgeMsg[nnz, F] (forward input),
 *                gradOut[rows, F] (upstream gradient)</li>
 *   <li>IArgs:   rows, mode</li>
 *   <li>Output:  dEdgeMsg[nnz, F] — gradient w.r.t. edgeMsg</li>
 * </ul>
 *
 * <p>Gradient routing per mode (mirrors the forward semantics):
 * <ul>
 *   <li>mode 0 (SUM):  {@code dEdgeMsg[e, f] = gradOut[i, f]} for all e in row i</li>
 *   <li>mode 1 (MEAN): {@code dEdgeMsg[e, f] = gradOut[i, f] / degree(i)}</li>
 *   <li>mode 2 (MAX):  gradient routed to the unique argmax edge per (i, f);
 *                       non-argmax edges receive 0 (handled by pre-zero in native)</li>
 * </ul>
 *
 * <p>This is a backward primitive: {@code doDiff} is not implemented.
 */
public class CsrEdgeAggregateBp extends DynamicCustomOp {

    /** No-arg constructor required for op-registry reflection. */
    public CsrEdgeAggregateBp() {}

    /**
     * Eager (INDArray) constructor.
     *
     * @param rowPtr   1D [rows+1] INT32 CSR row pointers
     * @param edgeMsg  2D [nnz, F] forward edge messages (for MAX argmax recompute)
     * @param gradOut  2D [rows, F] upstream gradient
     * @param rows     number of target nodes
     * @param mode     0=SUM, 1=MEAN, 2=MAX (must match forward)
     */
    public CsrEdgeAggregateBp(INDArray rowPtr, INDArray edgeMsg, INDArray gradOut,
                               long rows, int mode) {
        super(new INDArray[]{rowPtr, edgeMsg, gradOut}, null);
        addIArgument(rows, mode);
    }

    /**
     * SameDiff (symbolic) constructor.
     *
     * @param sd       the SameDiff graph
     * @param rowPtr   SDVariable [rows+1] INT32 row pointers
     * @param edgeMsg  SDVariable [nnz, F] forward edge messages
     * @param gradOut  SDVariable [rows, F] upstream gradient
     * @param rows     number of target nodes
     * @param mode     0=SUM, 1=MEAN, 2=MAX
     */
    public CsrEdgeAggregateBp(SameDiff sd, SDVariable rowPtr, SDVariable edgeMsg,
                               SDVariable gradOut, long rows, int mode) {
        super(sd, new SDVariable[]{rowPtr, edgeMsg, gradOut});
        addIArgument(rows, mode);
    }

    @Override
    public String opName() { return "csr_edge_aggregate_bp"; }

    @Override
    public List<DataType> calculateOutputDataTypes(List<DataType> dataTypes) {
        // dEdgeMsg dtype = edgeMsg dtype (input[1])
        return Collections.singletonList(dataTypes.get(1));
    }
}
