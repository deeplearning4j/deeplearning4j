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

import java.util.Arrays;
import java.util.Collections;
import java.util.List;

/**
 * Segment scatter-reduce: aggregate per-edge messages to per-node outputs.
 *
 * <p>This is the N-step (aggregation) of MPNN / edge-conditioned GNNs: for each
 * target node {@code i} (CSR row) and feature dimension {@code f}, reduce the
 * incoming edge messages {@code edgeMsg[e, f]} over all edges {@code e} in row
 * {@code i} according to the chosen reduction mode:
 *
 * <ul>
 *   <li>mode 0 — SUM:  {@code out[i, f] = Σ_{e} edgeMsg[e, f]}</li>
 *   <li>mode 1 — MEAN: {@code out[i, f] = Σ_{e} edgeMsg[e, f] / degree(i)}
 *                       (empty row → 0)</li>
 *   <li>mode 2 — MAX:  {@code out[i, f] = max_{e} edgeMsg[e, f]}
 *                       (empty row → 0)</li>
 * </ul>
 *
 * <p>C++ op name: {@code csr_edge_aggregate}
 * <ul>
 *   <li>Inputs:  rowPtr[rows+1] (INT32), edgeMsg[nnz, F] (floating)</li>
 *   <li>IArgs:   rows, mode</li>
 *   <li>Output:  out[rows, F]</li>
 * </ul>
 *
 * <p>{@code rowPtr} is structural and receives zero gradient.
 * {@code edgeMsg} is differentiable; backward pass is {@link CsrEdgeAggregateBp}.
 */
public class CsrEdgeAggregate extends DynamicCustomOp {

    /** No-arg constructor required for op-registry reflection. */
    public CsrEdgeAggregate() {}

    /**
     * Eager (INDArray) constructor.
     *
     * @param rowPtr   1D [rows+1] INT32 CSR row pointers
     * @param edgeMsg  2D [nnz, F] per-edge message vectors (floating dtype)
     * @param rows     number of target nodes (output rows)
     * @param mode     0=SUM, 1=MEAN, 2=MAX
     */
    public CsrEdgeAggregate(INDArray rowPtr, INDArray edgeMsg, long rows, int mode) {
        super(new INDArray[]{rowPtr, edgeMsg}, null);
        addIArgument(rows, mode);
    }

    /**
     * SameDiff (symbolic) constructor.
     *
     * @param sd       the SameDiff graph
     * @param rowPtr   SDVariable [rows+1] INT32 row pointers
     * @param edgeMsg  SDVariable [nnz, F] per-edge messages
     * @param rows     number of target nodes
     * @param mode     0=SUM, 1=MEAN, 2=MAX
     */
    public CsrEdgeAggregate(SameDiff sd, SDVariable rowPtr, SDVariable edgeMsg,
                            long rows, int mode) {
        super(sd, new SDVariable[]{rowPtr, edgeMsg});
        addIArgument(rows, mode);
    }

    @Override
    public String opName() { return "csr_edge_aggregate"; }

    @Override
    public List<DataType> calculateOutputDataTypes(List<DataType> dataTypes) {
        // out dtype = edgeMsg dtype (input[1])
        return Collections.singletonList(dataTypes.get(1));
    }

    /**
     * Backward pass for csr_edge_aggregate — delegates to {@link CsrEdgeAggregateBp}.
     *
     * <p>Gradient routing per mode:
     * <ul>
     *   <li>SUM:  {@code dEdgeMsg[e, f] = gradOut[i, f]} for all e in row i</li>
     *   <li>MEAN: {@code dEdgeMsg[e, f] = gradOut[i, f] / degree(i)}</li>
     *   <li>MAX:  gradient goes to the unique argmax edge per (i, f)</li>
     * </ul>
     *
     * @param grads upstream gradients; {@code grads.get(0)} is d(loss)/d(out[rows, F])
     * @return gradients for [rowPtr, edgeMsg]
     */
    @Override
    public List<SDVariable> doDiff(List<SDVariable> grads) {
        SDVariable g = grads.get(0);  // [rows, F]

        long rows = iArguments.get(0);
        long mode = iArguments.get(1);

        SDVariable rowPtrArg  = arg(0);
        SDVariable edgeMsgArg = arg(1);

        SDVariable dEdgeMsg = new CsrEdgeAggregateBp(
                sameDiff, rowPtrArg, edgeMsgArg, g, rows, (int) mode)
                .outputVariable();

        return Arrays.asList(
                sameDiff.zerosLike(rowPtrArg),
                dEdgeMsg);
    }
}
