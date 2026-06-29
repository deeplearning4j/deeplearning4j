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
 * Sparse matrix-vector product in CSR format: y = A * x  (or Aᵀ * x).
 *
 * <p>C++ op name: {@code csr_spmv}
 * <ul>
 *   <li>Inputs:  values[nnz], colIdx[nnz INT32], rowPtr[rows+1 INT32], x[cols] (x[rows] if transpose)</li>
 *   <li>IArgs:   rows, cols, transposeA (0 or 1)</li>
 *   <li>Output:  y[rows] (y[cols] if transpose)</li>
 * </ul>
 *
 * <p>Only {@code values} and {@code x} are differentiable; {@code colIdx} and {@code rowPtr}
 * are integer/structural and receive zero gradients.
 */
public class CsrSpmv extends DynamicCustomOp {

    /** Number of rows in the sparse matrix. Stored so doDiff can use it. */
    private long rows;
    /** Number of columns in the sparse matrix. */
    private long cols;
    /** Whether to multiply by Aᵀ instead of A. */
    private boolean transposeA;

    /** No-arg constructor required for op-registry reflection. */
    public CsrSpmv() {}

    /**
     * Eager (INDArray) constructor.
     *
     * @param values     1D [nnz] non-zero values (floating dtype)
     * @param colIdx     1D [nnz] INT32 column indices
     * @param rowPtr     1D [rows+1] INT32 row pointers
     * @param x          1D dense vector: length = cols (non-transpose) or rows (transpose)
     * @param rows       number of rows in the logical dense matrix
     * @param cols       number of columns in the logical dense matrix
     * @param transposeA if true compute Aᵀ · x instead of A · x
     */
    public CsrSpmv(INDArray values, INDArray colIdx, INDArray rowPtr, INDArray x,
                   long rows, long cols, boolean transposeA) {
        super(new INDArray[]{values, colIdx, rowPtr, x}, null);
        this.rows = rows;
        this.cols = cols;
        this.transposeA = transposeA;
        addIArgument(rows, cols, transposeA ? 1L : 0L);
    }

    /**
     * SameDiff (symbolic) constructor.
     *
     * @param sd         the SameDiff graph
     * @param values     SDVariable [nnz] non-zero values
     * @param colIdx     SDVariable [nnz] INT32 column indices
     * @param rowPtr     SDVariable [rows+1] INT32 row pointers
     * @param x          SDVariable dense vector
     * @param rows       number of rows
     * @param cols       number of columns
     * @param transposeA if true compute Aᵀ · x
     */
    public CsrSpmv(SameDiff sd, SDVariable values, SDVariable colIdx, SDVariable rowPtr,
                   SDVariable x, long rows, long cols, boolean transposeA) {
        super(sd, new SDVariable[]{values, colIdx, rowPtr, x});
        this.rows = rows;
        this.cols = cols;
        this.transposeA = transposeA;
        addIArgument(rows, cols, transposeA ? 1L : 0L);
    }

    @Override
    public String opName() { return "csr_spmv"; }

    @Override
    public List<DataType> calculateOutputDataTypes(List<DataType> dataTypes) {
        // Output dtype matches values (input[0]) dtype
        return Collections.singletonList(dataTypes.get(0));
    }

    /**
     * Backward pass for csr_spmv — delegates to the dedicated {@code csr_spmv_bp} op.
     *
     * <p>Outputs of the backward op:
     * <ul>
     *   <li>[0] dAValues[nnz]: gradient w.r.t. CSR non-zero values</li>
     *   <li>[1] dx:            gradient w.r.t. the dense input vector</li>
     * </ul>
     * colIdx and rowPtr are integer/structural and receive zero gradients.
     */
    @Override
    public List<SDVariable> doDiff(List<SDVariable> grads) {
        SDVariable dy = grads.get(0);

        long r = iArguments.size() > 0 ? iArguments.get(0) : this.rows;
        long c = iArguments.size() > 1 ? iArguments.get(1) : this.cols;
        boolean t = iArguments.size() > 2 ? (iArguments.get(2) != 0L) : this.transposeA;

        SDVariable valuesVar = arg(0);
        SDVariable colIdxVar = arg(1);
        SDVariable rowPtrVar = arg(2);
        SDVariable xVar      = arg(3);

        // Delegate to dedicated backward op: outputs are [dAValues, dx]
        SDVariable[] bpOut = new CsrSpmvBp(sameDiff, valuesVar, colIdxVar, rowPtrVar, xVar, dy, r, c, t)
                .outputVariables();

        return Arrays.asList(bpOut[0], sameDiff.zerosLike(colIdxVar),
                sameDiff.zerosLike(rowPtrVar), bpOut[1]);
    }
}
