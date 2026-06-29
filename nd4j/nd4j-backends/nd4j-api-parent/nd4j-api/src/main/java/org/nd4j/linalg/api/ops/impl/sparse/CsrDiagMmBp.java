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
 * Exact backward pass for {@link CsrDiagMm}: {@code out[e] = dl[i_e] * aValues[e] * dr[j_e]}.
 *
 * <p>Given the upstream gradient {@code gradOut[nnz]} w.r.t. the forward output values
 * and the original forward inputs, computes:
 * <pre>
 *   dAValues[e]  = gradOut[e] * dl[i_e] * dr[j_e]
 *   ddl[i]       = sum_{e in row i}    gradOut[e] * aValues[e] * dr[j_e]
 *   ddr[j]       = sum_{e: col j_e==j} gradOut[e] * aValues[e] * dl[i_e]
 * </pre>
 *
 * <p>C++ op name: {@code csr_diag_mm_bp}
 *
 * <p><b>Inputs (6 arrays):</b>
 * <ol>
 *   <li>{@code aValues}  – 1D [nnz],    floating dtype — original non-zero values of A</li>
 *   <li>{@code aColIdx}  – 1D [nnz],    INT32 — column indices of A</li>
 *   <li>{@code aRowPtr}  – 1D [rows+1], INT32 — row pointers of A</li>
 *   <li>{@code dl}       – 1D [rows],   floating dtype — left diagonal</li>
 *   <li>{@code dr}       – 1D [cols],   floating dtype — right diagonal</li>
 *   <li>{@code gradOut}  – 1D [nnz],    floating dtype — upstream gradient w.r.t. outValues</li>
 * </ol>
 *
 * <p><b>Integer arguments (IArgs):</b> {@code rows}, {@code cols}.
 *
 * <p><b>Outputs (3 arrays):</b>
 * <ol>
 *   <li>{@code dAValues} – 1D [nnz],  same floating dtype as {@code aValues}</li>
 *   <li>{@code ddl}      – 1D [rows], same floating dtype as {@code dl}</li>
 *   <li>{@code ddr}      – 1D [cols], same floating dtype as {@code dr}</li>
 * </ol>
 *
 * <p>This op is forward-only (no {@code doDiff}); it is the gradient leaf created by
 * {@link CsrDiagMm#doDiff}.
 */
public class CsrDiagMmBp extends DynamicCustomOp {

    /** No-arg constructor required for op-registry reflection. */
    public CsrDiagMmBp() {}

    /**
     * Eager (INDArray) constructor.
     *
     * @param aValues  1D [nnz]    original non-zero values of A (floating dtype)
     * @param aColIdx  1D [nnz]    INT32 column indices of A
     * @param aRowPtr  1D [rows+1] INT32 row pointers of A
     * @param dl       1D [rows]   left diagonal vector (floating dtype)
     * @param dr       1D [cols]   right diagonal vector (floating dtype)
     * @param gradOut  1D [nnz]    upstream gradient w.r.t. outValues (floating dtype)
     * @param rows     number of rows in the sparse matrix
     * @param cols     number of columns in the sparse matrix
     */
    public CsrDiagMmBp(INDArray aValues, INDArray aColIdx, INDArray aRowPtr,
                        INDArray dl, INDArray dr, INDArray gradOut,
                        long rows, long cols) {
        super(new INDArray[]{aValues, aColIdx, aRowPtr, dl, dr, gradOut}, null);
        addIArgument(rows, cols);
    }

    /**
     * SameDiff (symbolic) constructor.
     *
     * @param sd       the SameDiff graph
     * @param aValues  SDVariable [nnz]    original non-zero values of A
     * @param aColIdx  SDVariable [nnz]    INT32 column indices of A
     * @param aRowPtr  SDVariable [rows+1] INT32 row pointers of A
     * @param dl       SDVariable [rows]   left diagonal vector
     * @param dr       SDVariable [cols]   right diagonal vector
     * @param gradOut  SDVariable [nnz]    upstream gradient w.r.t. outValues
     * @param rows     number of rows
     * @param cols     number of columns
     */
    public CsrDiagMmBp(SameDiff sd,
                        SDVariable aValues, SDVariable aColIdx, SDVariable aRowPtr,
                        SDVariable dl, SDVariable dr, SDVariable gradOut,
                        long rows, long cols) {
        super(sd, new SDVariable[]{aValues, aColIdx, aRowPtr, dl, dr, gradOut});
        addIArgument(rows, cols);
    }

    @Override
    public String opName() {
        return "csr_diag_mm_bp";
    }

    /**
     * Output data types:
     * <ol>
     *   <li>{@code dAValues} – same floating dtype as {@code aValues} (input index 0)</li>
     *   <li>{@code ddl}      – same floating dtype as {@code dl}      (input index 3)</li>
     *   <li>{@code ddr}      – same floating dtype as {@code dr}      (input index 4)</li>
     * </ol>
     */
    @Override
    public List<DataType> calculateOutputDataTypes(List<DataType> dataTypes) {
        return Arrays.asList(dataTypes.get(0), dataTypes.get(3), dataTypes.get(4));
    }
}
