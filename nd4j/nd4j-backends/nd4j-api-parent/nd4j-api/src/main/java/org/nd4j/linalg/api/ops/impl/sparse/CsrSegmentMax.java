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
 * Neighbourhood max-aggregation over a graph in CSR format: the GraphSAGE-max primitive.
 *
 * <p>For each target node {@code i} (row), collects the feature vectors of its source
 * neighbours {@code j} (given by {@code colIdx} entries in row {@code i}), then takes the
 * element-wise maximum:
 * <pre>
 *   out[i, f] = max_{k: rowPtr[i] <= k < rowPtr[i+1]} X[colIdx[k], f]
 * </pre>
 * This is the max-aggregator in GraphSAGE (Hamilton et al. 2017) and is used for
 * inductive representation learning on large graphs.  Rows with no neighbours
 * (empty adjacency row) produce {@code -Infinity} (or the backend's corresponding
 * sentinel) in the output.
 *
 * <p>C++ op name: {@code csr_segment_max}
 * <ul>
 *   <li>Inputs:  colIdx[nnz] (INT32), rowPtr[rows+1] (INT32), X[n, f] (floating)</li>
 *   <li>IArgs:   rows</li>
 *   <li>Output:  out[rows, f] — per-segment (per-row) feature max</li>
 * </ul>
 *
 * <p>{@code colIdx} and {@code rowPtr} are integer/structural and receive zero gradients.
 * Only {@code X} is differentiable; the backward pass is handled by {@link CsrSegmentMaxBp}.
 */
public class CsrSegmentMax extends DynamicCustomOp {

    /** Number of rows (segments / target nodes). Stored so doDiff can use it. */
    private long rows;

    /** No-arg constructor required for op-registry reflection. */
    public CsrSegmentMax() {}

    /**
     * Eager (INDArray) constructor.
     *
     * @param colIdx  1D [nnz] INT32 column (source-node) indices
     * @param rowPtr  1D [rows+1] INT32 row (segment) pointers
     * @param X       2D [n, f] dense node feature matrix (floating dtype)
     * @param rows    number of target-node rows (segments)
     */
    public CsrSegmentMax(INDArray colIdx, INDArray rowPtr, INDArray X, long rows) {
        super(new INDArray[]{colIdx, rowPtr, X}, null);
        this.rows = rows;
        addIArgument(rows);
    }

    /**
     * SameDiff (symbolic) constructor.
     *
     * @param sd      the SameDiff graph
     * @param colIdx  SDVariable [nnz] INT32 column indices
     * @param rowPtr  SDVariable [rows+1] INT32 row pointers
     * @param X       SDVariable [n, f] dense node features
     * @param rows    number of rows (segments)
     */
    public CsrSegmentMax(SameDiff sd, SDVariable colIdx, SDVariable rowPtr,
                         SDVariable X, long rows) {
        super(sd, new SDVariable[]{colIdx, rowPtr, X});
        this.rows = rows;
        addIArgument(rows);
    }

    @Override
    public String opName() { return "csr_segment_max"; }

    @Override
    public List<DataType> calculateOutputDataTypes(List<DataType> dataTypes) {
        // Output dtype matches X (input[2])
        return Collections.singletonList(dataTypes.get(2));
    }

    /**
     * Backward pass for csr_segment_max — delegates to {@link CsrSegmentMaxBp}.
     *
     * <p>The gradient is a masked scatter: {@code dX[j, f] = gradOut[i, f]} if node
     * {@code j} was the max-winning neighbour of row {@code i} at feature dimension
     * {@code f}, or zero otherwise.  Ties are broken by the native op (typically the
     * first occurrence wins).
     *
     * @param grads upstream gradients; {@code grads.get(0)} is d(loss)/d(out[rows, f])
     * @return gradients for [colIdx, rowPtr, X]
     */
    @Override
    public List<SDVariable> doDiff(List<SDVariable> grads) {
        SDVariable g = grads.get(0);

        long r = iArguments.size() > 0 ? iArguments.get(0) : this.rows;

        SDVariable colIdxArg = arg(0);
        SDVariable rowPtrArg = arg(1);
        SDVariable xArg      = arg(2);

        SDVariable dX = new CsrSegmentMaxBp(sameDiff, colIdxArg, rowPtrArg, xArg, g, r)
                .outputVariable();

        return Arrays.asList(
                sameDiff.zerosLike(colIdxArg),
                sameDiff.zerosLike(rowPtrArg),
                dX);
    }
}
