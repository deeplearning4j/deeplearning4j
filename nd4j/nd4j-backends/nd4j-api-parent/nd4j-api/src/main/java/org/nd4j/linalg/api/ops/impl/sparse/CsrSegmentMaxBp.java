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
 * Backward pass for {@link CsrSegmentMax}: gradient of the neighbourhood max-aggregation.
 *
 * <p>C++ op name: {@code csr_segment_max_bp}
 * <ul>
 *   <li>Inputs:  colIdx[nnz] (INT32), rowPtr[rows+1] (INT32),
 *                X[n, f] (forward input node features),
 *                gradOut[rows, f] (upstream gradient)</li>
 *   <li>IArgs:   rows</li>
 *   <li>Output:  dX[n, f] — gradient w.r.t. the node feature matrix X</li>
 * </ul>
 *
 * <p>Gradient semantics: for each row {@code i} and feature {@code f}, the gradient
 * {@code gradOut[i, f]} is routed back to the unique neighbour {@code j*} that achieved
 * the maximum {@code X[j*, f]} in the forward pass.  If a node wins the max in multiple
 * rows, its gradient accumulates (sum).  Ties are broken by the native op.
 *
 * <p>This is a backward primitive: {@code doDiff} is not implemented.
 */
public class CsrSegmentMaxBp extends DynamicCustomOp {

    /** No-arg constructor required for op-registry reflection. */
    public CsrSegmentMaxBp() {}

    /**
     * Eager (INDArray) constructor.
     *
     * @param colIdx   1D [nnz] INT32 column (source-node) indices
     * @param rowPtr   1D [rows+1] INT32 row (segment) pointers
     * @param X        2D [n, f] dense node feature matrix from the forward pass
     * @param gradOut  2D [rows, f] upstream gradient
     * @param rows     number of rows (segments)
     */
    public CsrSegmentMaxBp(INDArray colIdx, INDArray rowPtr, INDArray X,
                           INDArray gradOut, long rows) {
        super(new INDArray[]{colIdx, rowPtr, X, gradOut}, null);
        addIArgument(rows);
    }

    /**
     * SameDiff (symbolic) constructor.
     *
     * @param sd       the SameDiff graph
     * @param colIdx   SDVariable [nnz] INT32 column indices
     * @param rowPtr   SDVariable [rows+1] INT32 row pointers
     * @param X        SDVariable [n, f] forward node features
     * @param gradOut  SDVariable [rows, f] upstream gradient
     * @param rows     number of rows
     */
    public CsrSegmentMaxBp(SameDiff sd, SDVariable colIdx, SDVariable rowPtr,
                           SDVariable X, SDVariable gradOut, long rows) {
        super(sd, new SDVariable[]{colIdx, rowPtr, X, gradOut});
        addIArgument(rows);
    }

    @Override
    public String opName() { return "csr_segment_max_bp"; }

    @Override
    public List<DataType> calculateOutputDataTypes(List<DataType> dataTypes) {
        // dX: same dtype as X (input[2])
        return Collections.singletonList(dataTypes.get(2));
    }
}
