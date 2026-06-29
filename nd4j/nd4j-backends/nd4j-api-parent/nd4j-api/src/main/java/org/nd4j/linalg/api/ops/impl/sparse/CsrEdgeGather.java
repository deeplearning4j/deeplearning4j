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
 * Edge-gather primitive: pull node-feature vectors onto edges.
 *
 * <p>For each edge {@code e} and feature dimension {@code f}:
 * <pre>
 *   edgeFeat[e, f] = X[colIdx[e], f]
 * </pre>
 *
 * <p>This is the E-step (gather) in MPNN / edge-conditioned GNNs: each edge acquires
 * the source-node embedding so that a per-edge message function can operate entirely
 * on edge tensors.
 *
 * <p>C++ op name: {@code csr_edge_gather}
 * <ul>
 *   <li>Inputs:  colIdx[nnz] (INT32), X[n, F] (floating)</li>
 *   <li>IArgs:   none (n, F read from X in C++)</li>
 *   <li>Output:  edgeFeat[nnz, F]</li>
 * </ul>
 *
 * <p>{@code colIdx} is structural and receives zero gradient.
 * Only {@code X} is differentiable; the backward pass is {@link CsrEdgeGatherBp}.
 */
public class CsrEdgeGather extends DynamicCustomOp {

    /**
     * Number of nodes in X (= X.shape[0]).  Stored for {@link #doDiff} so the
     * backward op can construct the correct output shape [n, F] for dX.
     */
    private final long n;

    /** No-arg constructor required for op-registry reflection. */
    public CsrEdgeGather() { this.n = -1; }

    /**
     * Eager (INDArray) constructor.
     *
     * @param colIdx  1D [nnz] INT32 source-node ids for each edge
     * @param X       2D [n, F] dense node-feature matrix (floating dtype)
     */
    public CsrEdgeGather(INDArray colIdx, INDArray X) {
        super(new INDArray[]{colIdx, X}, null);
        this.n = X.size(0);
        // No IArgs: n and F are read from X's shape in C++
    }

    /**
     * Eager (INDArray) constructor with an explicit node count — accepted for API
     * symmetry with the SameDiff ctor and the generated NDSparse namespace. {@code n}
     * is stored for {@link #doDiff}; the C++ op still reads n/F from X's shape.
     *
     * @param colIdx 1D [nnz] INT32 source-node ids
     * @param X      2D [n, F] dense node features
     * @param n      number of nodes (= X.shape[0])
     */
    public CsrEdgeGather(INDArray colIdx, INDArray X, long n) {
        super(new INDArray[]{colIdx, X}, null);
        this.n = n;
    }

    /**
     * SameDiff (symbolic) constructor.
     *
     * @param sd      the SameDiff graph
     * @param colIdx  SDVariable [nnz] INT32 source-node ids
     * @param X       SDVariable [n, F] dense node features
     * @param n       number of nodes (= X.shape[0]) — stored for doDiff
     */
    public CsrEdgeGather(SameDiff sd, SDVariable colIdx, SDVariable X, long n) {
        super(sd, new SDVariable[]{colIdx, X});
        this.n = n;
        // No IArgs for forward op
    }

    @Override
    public String opName() { return "csr_edge_gather"; }

    @Override
    public List<DataType> calculateOutputDataTypes(List<DataType> dataTypes) {
        // edgeFeat dtype = X dtype (input[1])
        return Collections.singletonList(dataTypes.get(1));
    }

    /**
     * Backward pass for csr_edge_gather — delegates to {@link CsrEdgeGatherBp}.
     *
     * <p>The gradient of the gather is a scatter-add:
     * {@code dX[colIdx[e], f] += gradEdge[e, f]} summed over all edges.
     * Multiple edges targeting the same node accumulate into dX.
     *
     * <p>X is passed to the backward op purely so the C++ shape function can determine
     * the output shape [n, F] from X's shapeInfo; X's values are not read in the bp op.
     *
     * @param grads upstream gradients; {@code grads.get(0)} is d(loss)/d(edgeFeat[nnz, F])
     * @return gradients for [colIdx, X]
     */
    @Override
    public List<SDVariable> doDiff(List<SDVariable> grads) {
        SDVariable g = grads.get(0);  // [nnz, F]

        SDVariable colIdxArg = arg(0);  // [nnz]  INT
        SDVariable xArg      = arg(1);  // [n, F] FLOAT — shape source for dX

        // Pass X so the C++ shape function reads n from X.shape[0]; no IArg needed.
        SDVariable dX = new CsrEdgeGatherBp(sameDiff, colIdxArg, xArg, g)
                .outputVariable();

        return Arrays.asList(
                sameDiff.zerosLike(colIdxArg),
                dX);
    }
}
