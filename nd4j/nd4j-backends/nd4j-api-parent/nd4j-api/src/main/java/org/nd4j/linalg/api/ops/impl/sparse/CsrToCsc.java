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
 * Converts a CSR (Compressed Sparse Row) sparse matrix to CSC (Compressed Sparse Column) format.
 *
 * <p>C++ op name: {@code csr_to_csc}
 *
 * <p>Inputs:
 * <ol>
 *   <li>values  – 1D [nnz], floating dtype — the non-zero values in row-major order</li>
 *   <li>colIdx  – 1D [nnz], INT32 — column indices (CSR column-index array)</li>
 *   <li>rowPtr  – 1D [rows+1], INT32 — row pointers (CSR row-pointer array)</li>
 * </ol>
 * Integer args: rows, cols
 *
 * <p>Outputs:
 * <ol>
 *   <li>cscValues  – 1D [nnz], same dtype as values — non-zero values in column-major order</li>
 *   <li>cscRowIdx  – 1D [nnz], INT32 — row index for each non-zero in the CSC ordering</li>
 *   <li>cscColPtr  – 1D [cols+1], INT32 — column pointers</li>
 * </ol>
 *
 * <p><b>Algebraic identity:</b> The CSC representation of a matrix A is identical to the
 * CSR representation of Aᵀ.  Callers can therefore obtain the sparse transpose of A at no
 * additional cost by reinterpreting the outputs (swap shape dimensions, use cscColPtr as the
 * row-pointer array and cscRowIdx as the column-index array).
 */
public class CsrToCsc extends DynamicCustomOp {

    /** No-arg constructor required for ImportClassMapping reflection. */
    public CsrToCsc() {}

    /**
     * Construct from pre-built CSR component arrays.
     *
     * @param values  1D [nnz] non-zero values
     * @param colIdx  1D [nnz] CSR column indices (INT32/INT64)
     * @param rowPtr  1D [rows+1] CSR row pointers (INT32/INT64)
     * @param rows    number of rows in the logical matrix
     * @param cols    number of columns in the logical matrix
     */
    public CsrToCsc(INDArray values, INDArray colIdx, INDArray rowPtr, long rows, long cols) {
        super(new INDArray[]{values, colIdx, rowPtr}, null);
        addIArgument(rows, cols);
    }

    /**
     * SameDiff constructor.
     *
     * @param sd      SameDiff instance
     * @param values  SD variable for non-zero values
     * @param colIdx  SD variable for CSR column indices
     * @param rowPtr  SD variable for CSR row pointers
     * @param rows    number of rows
     * @param cols    number of columns
     */
    public CsrToCsc(SameDiff sd, SDVariable values, SDVariable colIdx, SDVariable rowPtr,
                    long rows, long cols) {
        super(sd, new SDVariable[]{values, colIdx, rowPtr});
        addIArgument(rows, cols);
    }

    @Override
    public String opName() {
        return "csr_to_csc";
    }

    @Override
    public List<DataType> calculateOutputDataTypes(List<DataType> dataTypes) {
        // Output 0 (cscValues) matches the values input dtype (index 0)
        // Output 1 (cscRowIdx) is INT32; output 2 (cscColPtr) is INT32
        return Arrays.asList(dataTypes.get(0), DataType.INT32, DataType.INT32);
    }

    /**
     * Backward pass for {@code csr_to_csc}.
     *
     * <p>The forward op applies a counting-sort value-permutation {@code P} mapping
     * CSR entry positions to CSC positions.  The gradient of
     * {@code cscValues = P(csrValues)} is {@code dCsrValues[e] = gradCscValues[P(e)]},
     * i.e., the inverse permutation applied to the upstream gradient.
     *
     * <p>This is computed by {@link CsrToCscBp} ({@code csr_to_csc_bp}) entirely
     * on-device: it re-applies {@code csr_to_csc} with dimensions swapped
     * (treating the CSC of A as the CSR of A^T) to route gradients back to their
     * original CSR positions.
     *
     * <p>The structural inputs {@code csrColIdx} and {@code csrRowPtr} receive zero
     * gradients.
     *
     * @param grads three-element list (one per forward output):
     *              {@code grads.get(0)} is the upstream gradient for {@code cscValues}
     * @return gradients for [csrValues, csrColIdx, csrRowPtr]
     */
    @Override
    public List<SDVariable> doDiff(List<SDVariable> grads) {
        SDVariable gradCscValues = grads.get(0);  // gradient w.r.t. cscValues output

        long rows = iArguments.get(0);
        long cols = iArguments.get(1);

        SDVariable csrColIdx = arg(1);
        SDVariable csrRowPtr = arg(2);

        // cscRowIdx (output[1]) and cscColPtr (output[2]) from the forward pass
        SDVariable[] fwdOuts = outputVariables();
        SDVariable cscRowIdx = fwdOuts[1];
        SDVariable cscColPtr = fwdOuts[2];

        SDVariable dAValues = new CsrToCscBp(sameDiff,
                csrColIdx, csrRowPtr,
                cscRowIdx, cscColPtr,
                gradCscValues, rows, cols).outputVariables()[0];

        // csrColIdx and csrRowPtr are structural INT arrays — zero gradient
        return Arrays.asList(
                dAValues,
                sameDiff.zerosLike(csrColIdx),
                sameDiff.zerosLike(csrRowPtr));
    }
}
