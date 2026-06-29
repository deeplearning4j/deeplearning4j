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
 * Diagonal-scaled sparse matrix product: {@code out[e] = dl[i] * aValues[e] * dr[j]}
 * for each non-zero entry {@code e} at position {@code (i, j)} in the CSR matrix A.
 *
 * <p>Equivalently, this computes the non-zero values of the product {@code Dl · A · Dr}
 * where {@code Dl = diag(dl)} and {@code Dr = diag(dr)}, while keeping the sparsity
 * structure (column indices and row pointers) unchanged.
 *
 * <p>C++ op name: {@code csr_diag_mm}
 *
 * <p><b>Inputs (5 arrays):</b>
 * <ol>
 *   <li>{@code aValues}  – 1D [nnz], floating dtype — non-zero values of A</li>
 *   <li>{@code aColIdx}  – 1D [nnz], INT32 — column indices of A</li>
 *   <li>{@code aRowPtr}  – 1D [rows+1], INT32 — row pointers of A</li>
 *   <li>{@code dl}       – 1D [rows], floating dtype — left diagonal vector</li>
 *   <li>{@code dr}       – 1D [cols], floating dtype — right diagonal vector</li>
 * </ol>
 *
 * <p><b>Integer arguments (IArgs):</b> {@code rows}, {@code cols}.
 *
 * <p><b>Output (1 array):</b>
 * <ol>
 *   <li>{@code outValues} – 1D [nnz], same dtype as {@code aValues} —
 *       the scaled non-zero values; the sparsity structure is unchanged</li>
 * </ol>
 *
 * <p><b>Autodiff:</b>
 * <ul>
 *   <li><b>d(aValues)</b>: exact — {@code dA[e] = g[e] * dl[i_e] * dr[j_e]}</li>
 *   <li><b>d(dl)</b>: exact — {@code ddl[i] = sum_{e in row i} g[e] * aValues[e] * dr[j_e]}</li>
 *   <li><b>d(dr)</b>: exact — {@code ddr[j] = sum_{e: col j_e==j} g[e] * aValues[e] * dl[i_e]}</li>
 *   <li><b>d(aColIdx)</b>, <b>d(aRowPtr)</b>: zero (integer/structural inputs).</li>
 * </ul>
 * All three floating gradients are computed in a single {@link CsrDiagMmBp} op.
 */
public class CsrDiagMm extends DynamicCustomOp {

    /** Number of rows in the sparse matrix. Stored so doDiff can use it. */
    private long rows;
    /** Number of columns in the sparse matrix. */
    private long cols;

    /** No-arg constructor required for op-registry reflection. */
    public CsrDiagMm() {}

    /**
     * Eager (INDArray) constructor.
     *
     * @param aValues  1D [nnz] non-zero values of A (floating dtype)
     * @param aColIdx  1D [nnz] INT32 column indices of A
     * @param aRowPtr  1D [rows+1] INT32 row pointers of A
     * @param dl       1D [rows] left diagonal vector (floating dtype)
     * @param dr       1D [cols] right diagonal vector (floating dtype)
     * @param rows     number of rows in the logical dense matrix
     * @param cols     number of columns in the logical dense matrix
     */
    public CsrDiagMm(INDArray aValues, INDArray aColIdx, INDArray aRowPtr,
                     INDArray dl, INDArray dr,
                     long rows, long cols) {
        super(new INDArray[]{aValues, aColIdx, aRowPtr, dl, dr}, null);
        this.rows = rows;
        this.cols = cols;
        addIArgument(rows, cols);
    }

    /**
     * SameDiff (symbolic) constructor.
     *
     * @param sd       the SameDiff graph
     * @param aValues  SDVariable [nnz] non-zero values of A
     * @param aColIdx  SDVariable [nnz] INT32 column indices of A
     * @param aRowPtr  SDVariable [rows+1] INT32 row pointers of A
     * @param dl       SDVariable [rows] left diagonal vector
     * @param dr       SDVariable [cols] right diagonal vector
     * @param rows     number of rows
     * @param cols     number of columns
     */
    public CsrDiagMm(SameDiff sd,
                     SDVariable aValues, SDVariable aColIdx, SDVariable aRowPtr,
                     SDVariable dl, SDVariable dr,
                     long rows, long cols) {
        super(sd, new SDVariable[]{aValues, aColIdx, aRowPtr, dl, dr});
        this.rows = rows;
        this.cols = cols;
        addIArgument(rows, cols);
    }

    @Override
    public String opName() {
        return "csr_diag_mm";
    }

    /**
     * Output data type: {@code outValues} has the same floating dtype as {@code aValues}
     * (input index 0).
     */
    @Override
    public List<DataType> calculateOutputDataTypes(List<DataType> dataTypes) {
        return Collections.singletonList(dataTypes.get(0));
    }

    /**
     * Backward pass for csr_diag_mm — delegates to {@link CsrDiagMmBp}.
     *
     * <p>Let {@code g = grads.get(0)} be the upstream gradient w.r.t. {@code outValues} [nnz].
     * The exact gradients are:
     * <pre>
     *   dAValues[e]  = g[e] * dl[i_e] * dr[j_e]
     *   ddl[i]       = sum_{e in row i}    g[e] * aValues[e] * dr[j_e]
     *   ddr[j]       = sum_{e: col j_e==j} g[e] * aValues[e] * dl[i_e]
     * </pre>
     *
     * @param grads upstream gradients; {@code grads.get(0)} is d(loss)/d(outValues)
     * @return gradients for [aValues, aColIdx, aRowPtr, dl, dr]
     */
    @Override
    public List<SDVariable> doDiff(List<SDVariable> grads) {
        SDVariable g = grads.get(0);

        // Read rows/cols from persisted iArguments (safe after no-arg construction too)
        long r = iArguments.size() > 0 ? iArguments.get(0) : this.rows;
        long c = iArguments.size() > 1 ? iArguments.get(1) : this.cols;

        SDVariable aValuesVar = arg(0);
        SDVariable aColIdxVar = arg(1);
        SDVariable aRowPtrVar = arg(2);
        SDVariable dlVar      = arg(3);
        SDVariable drVar      = arg(4);

        // Compute all three floating gradients in one native op
        SDVariable[] bpOuts = new CsrDiagMmBp(sameDiff,
                aValuesVar, aColIdxVar, aRowPtrVar, dlVar, drVar, g, r, c)
                .outputVariables();

        // bpOuts[0] = dAValues, bpOuts[1] = ddl, bpOuts[2] = ddr
        return Arrays.asList(
                bpOuts[0],                           // dAValues — exact
                sameDiff.zerosLike(aColIdxVar),      // aColIdx — integer/structural
                sameDiff.zerosLike(aRowPtrVar),      // aRowPtr — integer/structural
                bpOuts[1],                           // ddl — exact
                bpOuts[2]                            // ddr — exact
        );
    }
}
