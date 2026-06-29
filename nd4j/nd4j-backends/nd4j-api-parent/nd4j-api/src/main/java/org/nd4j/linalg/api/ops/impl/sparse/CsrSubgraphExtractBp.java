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
 * Backward pass for {@link CsrSubgraphExtract}: gradient of the induced-subgraph extraction.
 *
 * <p>C++ op name: {@code csr_subgraph_extract_bp}
 * <ul>
 *   <li>Inputs:  values[nnz] (float, forward input),
 *                colIdx[nnz] (INT), rowPtr[N+1] (INT), nodeIdx[K] (INT),
 *                dNewValues[nnz'] (float, upstream gradient w.r.t. newValues)</li>
 *   <li>IArgs:   N (original node count), K (selected count)</li>
 *   <li>Output:  dValues[nnz] (float) — gradient w.r.t. values; zero for dropped edges</li>
 * </ul>
 *
 * <p>Gradient semantics: for each kept edge {@code e} (original index) that maps to
 * extracted position {@code e'}, {@code dValues[e] = dNewValues[e']}. Dropped edges
 * (not in the induced subgraph) receive zero gradient.
 *
 * <p>This is a backward primitive; {@code doDiff} is not implemented.
 */
public class CsrSubgraphExtractBp extends DynamicCustomOp {

    /** No-arg constructor required for op-registry reflection. */
    public CsrSubgraphExtractBp() {}

    /**
     * Eager (INDArray) constructor.
     *
     * @param values      1D [nnz] float — forward input edge weights
     * @param colIdx      1D [nnz] INT — column indices
     * @param rowPtr      1D [N+1] INT — row pointers
     * @param nodeIdx     1D [K]   INT — selected node ids (sorted ascending)
     * @param dNewValues  1D [nnz'] float — upstream gradient w.r.t. newValues
     * @param N           original node count
     * @param K           number of selected nodes
     */
    public CsrSubgraphExtractBp(INDArray values, INDArray colIdx, INDArray rowPtr,
                                 INDArray nodeIdx, INDArray dNewValues, long N, long K) {
        super(new INDArray[]{values, colIdx, rowPtr, nodeIdx, dNewValues}, null);
        addIArgument(N, K);
    }

    /**
     * SameDiff (symbolic) constructor.
     *
     * @param sd          the SameDiff graph
     * @param values      SDVariable [nnz] float — forward input edge weights
     * @param colIdx      SDVariable [nnz] INT — column indices
     * @param rowPtr      SDVariable [N+1] INT — row pointers
     * @param nodeIdx     SDVariable [K]   INT — selected node ids (sorted)
     * @param dNewValues  SDVariable [nnz'] float — upstream gradient
     * @param N           original node count
     * @param K           number of selected nodes
     */
    public CsrSubgraphExtractBp(SameDiff sd, SDVariable values, SDVariable colIdx,
                                 SDVariable rowPtr, SDVariable nodeIdx, SDVariable dNewValues,
                                 long N, long K) {
        super(sd, new SDVariable[]{values, colIdx, rowPtr, nodeIdx, dNewValues});
        addIArgument(N, K);
    }

    @Override
    public String opName() { return "csr_subgraph_extract_bp"; }

    @Override
    public List<DataType> calculateOutputDataTypes(List<DataType> dataTypes) {
        // dValues: same float dtype as values (input[0])
        return Collections.singletonList(dataTypes.get(0));
    }
}
