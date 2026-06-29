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
 * Backward pass for {@link CsrToCsc}: apply the inverse value-permutation to produce
 * the gradient of the CSR values input.
 *
 * <p>C++ op name: {@code csr_to_csc_bp}
 *
 * <p><b>Background:</b> The forward {@code csr_to_csc} applies a counting-sort permutation
 * {@code P} that maps CSR entry positions to CSC entry positions.  The gradient of
 * {@code cscValues = P(csrValues)} with respect to {@code csrValues} is simply:
 * {@code dCsrValues[e] = gradCscValues[P(e)]}.  The inverse permutation {@code P⁻¹} is
 * computed device-side by re-applying {@code csr_to_csc} to the gradient arrays with the
 * matrix dimensions swapped (treating the CSC of A as the CSR of A^T).
 *
 * <p><b>Inputs (5 arrays):</b>
 * <ol>
 *   <li>{@code csrColIdx}     – 1D [nnz], INT32/INT64 — CSR column indices (forward input[1])</li>
 *   <li>{@code csrRowPtr}     – 1D [rows+1], same INT — CSR row pointers (forward input[2])</li>
 *   <li>{@code cscRowIdx}     – 1D [nnz], INT32 — CSC row indices (forward output[1])</li>
 *   <li>{@code cscColPtr}     – 1D [cols+1], INT32 — CSC column pointers (forward output[2])</li>
 *   <li>{@code gradCscValues} – 1D [nnz], float — upstream gradient w.r.t. cscValues output</li>
 * </ol>
 *
 * <p><b>Integer arguments (IArgs):</b> {@code rows}, {@code cols}
 *
 * <p><b>Output:</b>
 * <ol>
 *   <li>{@code dAValues} – 1D [nnz], same float dtype as {@code gradCscValues} —
 *       gradient w.r.t. the CSR values input</li>
 * </ol>
 *
 * <p>This op is forward-only (no {@code doDiff}); it is the gradient leaf for
 * {@link CsrToCsc#doDiff}.
 */
public class CsrToCscBp extends DynamicCustomOp {

    /** No-arg constructor required for ImportClassMapping reflection. */
    public CsrToCscBp() {}

    /**
     * Eager (INDArray) constructor.
     *
     * @param csrColIdx     1D [nnz] INT32/INT64 — CSR column indices (forward input[1])
     * @param csrRowPtr     1D [rows+1] INT32/INT64 — CSR row pointers (forward input[2])
     * @param cscRowIdx     1D [nnz] INT32 — CSC row indices (forward output[1])
     * @param cscColPtr     1D [cols+1] INT32 — CSC column pointers (forward output[2])
     * @param gradCscValues 1D [nnz] float — upstream gradient w.r.t. cscValues output
     * @param rows          number of rows in the original matrix A
     * @param cols          number of columns in the original matrix A
     */
    public CsrToCscBp(INDArray csrColIdx, INDArray csrRowPtr,
                       INDArray cscRowIdx, INDArray cscColPtr,
                       INDArray gradCscValues, long rows, long cols) {
        super(new INDArray[]{csrColIdx, csrRowPtr, cscRowIdx, cscColPtr, gradCscValues}, null);
        addIArgument(rows, cols);
    }

    /**
     * SameDiff (symbolic) constructor.
     *
     * @param sd            SameDiff instance
     * @param csrColIdx     SDVariable [nnz] INT32/INT64 — CSR column indices
     * @param csrRowPtr     SDVariable [rows+1] INT32/INT64 — CSR row pointers
     * @param cscRowIdx     SDVariable [nnz] INT32 — CSC row indices
     * @param cscColPtr     SDVariable [cols+1] INT32 — CSC column pointers
     * @param gradCscValues SDVariable [nnz] float — upstream gradient w.r.t. cscValues
     * @param rows          number of rows
     * @param cols          number of columns
     */
    public CsrToCscBp(SameDiff sd,
                       SDVariable csrColIdx, SDVariable csrRowPtr,
                       SDVariable cscRowIdx, SDVariable cscColPtr,
                       SDVariable gradCscValues, long rows, long cols) {
        super(sd, new SDVariable[]{csrColIdx, csrRowPtr, cscRowIdx, cscColPtr, gradCscValues});
        addIArgument(rows, cols);
    }

    @Override
    public String opName() {
        return "csr_to_csc_bp";
    }

    /**
     * Output data type: same float dtype as {@code gradCscValues} (input index 4).
     */
    @Override
    public List<DataType> calculateOutputDataTypes(List<DataType> dataTypes) {
        return Collections.singletonList(dataTypes.get(4));
    }
}
