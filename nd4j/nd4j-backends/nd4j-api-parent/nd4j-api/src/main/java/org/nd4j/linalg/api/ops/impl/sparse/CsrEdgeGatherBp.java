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
 * Backward pass for {@link CsrEdgeGather}: scatter-add edge gradients to node features.
 *
 * <p>C++ op name: {@code csr_edge_gather_bp}
 * <ul>
 *   <li>Inputs:  colIdx[nnz] (INT), X[n, F] (node features — shape only, values unused),
 *                gradEdge[nnz, F] (upstream gradient)</li>
 *   <li>IArgs:   none — n and F are read from X's shape</li>
 *   <li>Output:  dX[n, F] — gradient w.r.t. node features X (same shape as X)</li>
 * </ul>
 *
 * <p>Gradient semantics:
 * <pre>
 *   dX[colIdx[e], f] += gradEdge[e, f]   for all e in [0, nnz), f in [0, F)
 * </pre>
 * Multiple edges may point to the same source node; their contributions are summed.
 * The output dX is pre-zeroed in the native kernel before accumulation.
 * X values are NOT read — X is passed purely so the C++ shape function can derive n and F
 * without relying on an IArg (which was unreliable in the SameDiff path).
 *
 * <p>This is a backward primitive: {@code doDiff} is not implemented.
 */
public class CsrEdgeGatherBp extends DynamicCustomOp {

    /** No-arg constructor required for op-registry reflection. */
    public CsrEdgeGatherBp() {}

    /**
     * Eager (INDArray) constructor.
     *
     * @param colIdx    1D [nnz] INT source-node ids
     * @param x         2D [n, F] node-feature matrix (shape source; values unused)
     * @param gradEdge  2D [nnz, F] upstream gradient w.r.t. edgeFeat
     */
    public CsrEdgeGatherBp(INDArray colIdx, INDArray x, INDArray gradEdge) {
        super(new INDArray[]{colIdx, x, gradEdge}, null);
        // No IArgs: n and F are read from x's shape in C++
    }

    /**
     * SameDiff (symbolic) constructor.
     *
     * @param sd        the SameDiff graph
     * @param colIdx    SDVariable [nnz] INT source-node ids
     * @param x         SDVariable [n, F] node features (shape source; values unused)
     * @param gradEdge  SDVariable [nnz, F] upstream gradient
     */
    public CsrEdgeGatherBp(SameDiff sd, SDVariable colIdx, SDVariable x, SDVariable gradEdge) {
        super(sd, new SDVariable[]{colIdx, x, gradEdge});
        // No IArgs: n and F are read from x's shape in C++
    }

    @Override
    public String opName() { return "csr_edge_gather_bp"; }

    @Override
    public List<DataType> calculateOutputDataTypes(List<DataType> dataTypes) {
        // Inputs: [0]=colIdx(INT), [1]=X(FLOAT), [2]=gradEdge(FLOAT)
        // dX dtype = X dtype (input[1])
        return Collections.singletonList(dataTypes.get(1));
    }
}
