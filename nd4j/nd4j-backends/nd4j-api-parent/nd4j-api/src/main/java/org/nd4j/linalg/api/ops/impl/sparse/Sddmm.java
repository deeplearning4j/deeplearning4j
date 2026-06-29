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
 * Sampled Dense-Dense Matrix Multiplication (SDDMM).
 *
 * <p>For each non-zero position {@code k} at row {@code i}, column {@code j} in the sparsity
 * pattern defined by (rowPtr, colIdx), computes:
 * <pre>
 *   out[k] = sum_l  D1[i, l] * D2[j, l]
 * </pre>
 *
 * <p>C++ op name: {@code sddmm}
 * <ul>
 *   <li>Inputs:  rowPtr[rows+1 INT32], colIdx[nnz INT32], D1[rows, p], D2[cols, p]</li>
 *   <li>IArgs:   rows, cols</li>
 *   <li>Output:  values[nnz]  (same dtype as D1)</li>
 * </ul>
 *
 * <p>SDDMM is used as both a primary op and a backward primitive for
 * {@link CsrSpmv} and {@link CsrSpmm}.  First-order gradients w.r.t. D1 and D2
 * are computed by {@link SddmmBp}.
 */
public class Sddmm extends DynamicCustomOp {

    /** Number of rows in the sparse sparsity pattern. */
    private long rows;
    /** Number of columns in the sparse sparsity pattern. */
    private long cols;

    /** No-arg constructor required for op-registry reflection. */
    public Sddmm() {}

    /**
     * Eager (INDArray) constructor.
     *
     * @param rowPtr 1D [rows+1] INT32 row pointers
     * @param colIdx 1D [nnz] INT32 column indices
     * @param D1     2D [rows, p] left dense factor
     * @param D2     2D [cols, p] right dense factor
     * @param rows   number of rows
     * @param cols   number of columns
     */
    public Sddmm(INDArray rowPtr, INDArray colIdx, INDArray D1, INDArray D2,
                 long rows, long cols) {
        super(new INDArray[]{rowPtr, colIdx, D1, D2}, null);
        this.rows = rows;
        this.cols = cols;
        addIArgument(rows, cols);
    }

    /**
     * SameDiff (symbolic) constructor.
     *
     * @param sd     the SameDiff graph
     * @param rowPtr SDVariable [rows+1] INT32 row pointers
     * @param colIdx SDVariable [nnz] INT32 column indices
     * @param D1     SDVariable [rows, p] left dense factor
     * @param D2     SDVariable [cols, p] right dense factor
     * @param rows   number of rows
     * @param cols   number of columns
     */
    public Sddmm(SameDiff sd, SDVariable rowPtr, SDVariable colIdx,
                 SDVariable D1, SDVariable D2, long rows, long cols) {
        super(sd, new SDVariable[]{rowPtr, colIdx, D1, D2});
        this.rows = rows;
        this.cols = cols;
        addIArgument(rows, cols);
    }

    @Override
    public String opName() { return "sddmm"; }

    @Override
    public List<DataType> calculateOutputDataTypes(List<DataType> dataTypes) {
        // Output dtype matches D1 (input[2]) dtype
        return Collections.singletonList(dataTypes.get(2));
    }

    /**
     * Backward pass for sddmm — delegates to the dedicated {@code sddmm_bp} op.
     *
     * <p>Outputs of the backward op:
     * <ul>
     *   <li>[0] dD1[rows, p]: gradient w.r.t. the left dense factor D1</li>
     *   <li>[1] dD2[cols, p]: gradient w.r.t. the right dense factor D2</li>
     * </ul>
     * rowPtr and colIdx are integer/structural and receive zero gradients.
     */
    @Override
    public List<SDVariable> doDiff(List<SDVariable> grads) {
        SDVariable gradOut = grads.get(0);

        long r = iArguments.size() > 0 ? iArguments.get(0) : this.rows;
        long c = iArguments.size() > 1 ? iArguments.get(1) : this.cols;

        SDVariable rowPtrVar = arg(0);
        SDVariable colIdxVar = arg(1);
        SDVariable D1Var     = arg(2);
        SDVariable D2Var     = arg(3);

        // Delegate to dedicated backward op: outputs are [dD1, dD2]
        SDVariable[] bpOut = new SddmmBp(sameDiff, rowPtrVar, colIdxVar, D1Var, D2Var, gradOut, r, c)
                .outputVariables();

        return Arrays.asList(sameDiff.zerosLike(rowPtrVar), sameDiff.zerosLike(colIdxVar),
                bpOut[0], bpOut[1]);
    }
}
