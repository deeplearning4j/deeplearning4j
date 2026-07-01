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
 * Forward-only CSR SpMV with a user-selected algebraic semiring.
 * Graph algorithms are not differentiable; {@link #doDiff} is intentionally absent.
 *
 * <p>C++ op name: {@code csr_spmv_semiring}
 * <ul>
 *   <li>Inputs:  values[nnz], colIdx[nnz INT32], rowPtr[rows+1 INT32], x[cols]</li>
 *   <li>IArgs:   rows, cols, semiring (integer code from {@link Semiring#code()})</li>
 *   <li>Output:  y[rows]</li>
 * </ul>
 *
 * <p>For each row i, the output is computed as:
 * <pre>
 *   y[i] = REDUCE_{k in row i}( add(acc, mul(values[k], x[colIdx[k]])) )
 * </pre>
 * where {@code mul} and {@code add} are determined by the chosen {@link Semiring}.
 *
 * @see Semiring
 * @see CsrSpmv
 */
public class CsrSpmvSemiring extends DynamicCustomOp {

    /** Number of rows in the sparse matrix. */
    private long rows;
    /** Number of columns in the sparse matrix. */
    private long cols;
    /** The algebraic semiring used for the multiply-reduce. */
    private Semiring semiring;

    /** No-arg constructor required for op-registry reflection. */
    public CsrSpmvSemiring() {}

    /**
     * Eager (INDArray) constructor.
     *
     * @param values   1D [nnz] non-zero values (floating dtype)
     * @param colIdx   1D [nnz] INT32 column indices
     * @param rowPtr   1D [rows+1] INT32 row pointers
     * @param x        1D dense vector of length {@code cols}
     * @param rows     number of rows in the logical dense matrix
     * @param cols     number of columns in the logical dense matrix
     * @param semiring the algebraic semiring to use for the multiply-reduce
     */
    public CsrSpmvSemiring(INDArray values, INDArray colIdx, INDArray rowPtr, INDArray x,
                            long rows, long cols, Semiring semiring) {
        super(new INDArray[]{values, colIdx, rowPtr, x}, null);
        this.rows = rows;
        this.cols = cols;
        this.semiring = semiring;
        addIArgument(rows, cols, semiring.code());
    }

    /**
     * SameDiff (symbolic) constructor.
     *
     * @param sd       the SameDiff graph
     * @param values   SDVariable [nnz] non-zero values
     * @param colIdx   SDVariable [nnz] INT32 column indices
     * @param rowPtr   SDVariable [rows+1] INT32 row pointers
     * @param x        SDVariable dense vector of length {@code cols}
     * @param rows     number of rows
     * @param cols     number of columns
     * @param semiring the algebraic semiring to use for the multiply-reduce
     */
    public CsrSpmvSemiring(SameDiff sd, SDVariable values, SDVariable colIdx, SDVariable rowPtr,
                            SDVariable x, long rows, long cols, Semiring semiring) {
        super(sd, new SDVariable[]{values, colIdx, rowPtr, x});
        this.rows = rows;
        this.cols = cols;
        this.semiring = semiring;
        addIArgument(rows, cols, semiring.code());
    }

    @Override
    public String opName() { return "csr_spmv_semiring"; }

    @Override
    public List<DataType> calculateOutputDataTypes(List<DataType> dataTypes) {
        // Output dtype matches values (input[0]) dtype
        return Collections.singletonList(dataTypes.get(0));
    }

    @Override
    public List<SDVariable> doDiff(List<SDVariable> grads) {
        throw new UnsupportedOperationException(
            "csr_spmv_semiring is forward-only: semiring ops (min-plus / max-plus / or-and) " +
            "are non-differentiable and cannot be used in automatic differentiation.");
    }
}
