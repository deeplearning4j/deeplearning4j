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
import java.util.List;

/**
 * Extract the induced subgraph CSR for a selected set of K nodes (TopK/SAGPool).
 *
 * <p>Given a CSR graph with N nodes and a sorted list of K selected node ids
 * ({@code nodeIdx}), extracts the subgraph containing only edges {@code (i->j)}
 * where BOTH {@code i} and {@code j} appear in {@code nodeIdx}. Both endpoints
 * are remapped to their 0-based position in {@code nodeIdx}.
 *
 * <p>C++ op name: {@code csr_subgraph_extract}
 * <ul>
 *   <li>Inputs:  values[nnz] (float), colIdx[nnz] (INT), rowPtr[N+1] (INT),
 *                nodeIdx[K] (INT, sorted ascending)</li>
 *   <li>IArgs:   N (original node count), K (selected count)</li>
 *   <li>Outputs: newValues[nnz'] (float), newColIdx[nnz'] (INT32), newRowPtr[K+1] (INT32)</li>
 *   <li>nnz' is data-dependent and computed exactly in the native shape function.</li>
 * </ul>
 *
 * <p>Only {@code values} is differentiable; the backward pass is handled by
 * {@link CsrSubgraphExtractBp}. Index arrays (colIdx, rowPtr, nodeIdx) are
 * structural and receive zero gradients.
 */
public class CsrSubgraphExtract extends DynamicCustomOp {

    /** Original node count. Stored so doDiff can forward it to the _bp op. */
    private long N;
    /** Selected node count. */
    private long K;

    /** No-arg constructor required for op-registry reflection. */
    public CsrSubgraphExtract() {}

    /**
     * Eager (INDArray) constructor.
     *
     * @param values   1D [nnz] float — edge weights of the original graph
     * @param colIdx   1D [nnz] INT — column indices (destination node ids)
     * @param rowPtr   1D [N+1] INT — row pointer array
     * @param nodeIdx  1D [K]   INT — SORTED ascending selected node ids
     * @param N        original node count
     * @param K        number of selected nodes
     */
    public CsrSubgraphExtract(INDArray values, INDArray colIdx, INDArray rowPtr,
                               INDArray nodeIdx, long N, long K) {
        super(new INDArray[]{values, colIdx, rowPtr, nodeIdx}, null);
        this.N = N;
        this.K = K;
        addIArgument(N, K);
    }

    /**
     * SameDiff (symbolic) constructor.
     *
     * @param sd       the SameDiff graph
     * @param values   SDVariable [nnz] float — edge weights
     * @param colIdx   SDVariable [nnz] INT — column indices
     * @param rowPtr   SDVariable [N+1] INT — row pointers
     * @param nodeIdx  SDVariable [K]   INT — selected node ids (sorted)
     * @param N        original node count
     * @param K        number of selected nodes
     */
    public CsrSubgraphExtract(SameDiff sd, SDVariable values, SDVariable colIdx,
                               SDVariable rowPtr, SDVariable nodeIdx, long N, long K) {
        super(sd, new SDVariable[]{values, colIdx, rowPtr, nodeIdx});
        this.N = N;
        this.K = K;
        addIArgument(N, K);
    }

    @Override
    public String opName() { return "csr_subgraph_extract"; }

    @Override
    public List<DataType> calculateOutputDataTypes(List<DataType> dataTypes) {
        // out[0] newValues: same float dtype as values (input[0])
        // out[1] newColIdx: INT32
        // out[2] newRowPtr: INT32
        return Arrays.asList(dataTypes.get(0), DataType.INT32, DataType.INT32);
    }

    /**
     * Backward pass for csr_subgraph_extract.
     *
     * <p>Gradient flows only through {@code values} (edge weights). For each kept
     * edge {@code e -> e'}, {@code dValues[e] = dNewValues[e']}. Dropped edges get 0.
     * Structural inputs (colIdx, rowPtr, nodeIdx) get zerosLike.
     *
     * <p>The upstream gradient {@code grads.get(0)} is w.r.t. {@code newValues[nnz']}.
     * {@code grads.get(1)} and {@code grads.get(2)} are for the INT32 index outputs
     * (structural, not used).
     *
     * @param grads upstream gradients for the 3 outputs [newValues, newColIdx, newRowPtr]
     * @return gradients for [values, colIdx, rowPtr, nodeIdx]
     */
    @Override
    public List<SDVariable> doDiff(List<SDVariable> grads) {
        SDVariable dNewValues = grads.get(0);  // upstream grad w.r.t. newValues

        long nArg = iArguments.size() > 0 ? iArguments.get(0) : this.N;
        long kArg = iArguments.size() > 1 ? iArguments.get(1) : this.K;

        SDVariable valuesArg  = arg(0);
        SDVariable colIdxArg  = arg(1);
        SDVariable rowPtrArg  = arg(2);
        SDVariable nodeIdxArg = arg(3);

        SDVariable dValues = new CsrSubgraphExtractBp(
                sameDiff, valuesArg, colIdxArg, rowPtrArg, nodeIdxArg, dNewValues, nArg, kArg)
                .outputVariable();

        return Arrays.asList(
                dValues,
                sameDiff.zerosLike(colIdxArg),
                sameDiff.zerosLike(rowPtrArg),
                sameDiff.zerosLike(nodeIdxArg));
    }
}
