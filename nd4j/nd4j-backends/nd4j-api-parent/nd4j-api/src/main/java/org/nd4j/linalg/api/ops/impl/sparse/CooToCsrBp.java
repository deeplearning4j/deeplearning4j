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
 * Native backward pass for {@link CooToCsr}: distributes the upstream gradient
 * for the CSR values back to the original COO values input.
 *
 * <p>C++ op name: {@code coo_to_csr_bp}
 *
 * <p>Forward recap: {@code (cooIndices[coo_nnz,2], cooValues[coo_nnz]) →
 * (csrValues[csr_nnz], csrColIdx[csr_nnz], csrRowPtr[rows+1])}.
 * The forward op may coalesce duplicate {@code (row, col)} entries by summing
 * their values, so {@code csr_nnz ≤ coo_nnz}.
 *
 * <p><b>Inputs (4 arrays):</b>
 * <ol>
 *   <li>{@code cooIndices}    – 2D [coo_nnz, 2], INT — row/col index pairs (forward input[0])</li>
 *   <li>{@code csrColIdx}     – 1D [csr_nnz], INT32 — column indices (forward output[1])</li>
 *   <li>{@code csrRowPtr}     – 1D [rows+1], INT32 — row pointers (forward output[2])</li>
 *   <li>{@code gradCsrValues} – 1D [csr_nnz], float — upstream gradient w.r.t. the CSR values</li>
 * </ol>
 *
 * <p><b>Integer arguments (IArgs):</b> {@code rows}, {@code cols}.
 *
 * <p><b>Output (1 array):</b>
 * <ol>
 *   <li>{@code dCooValues} – 1D [coo_nnz], same float dtype as {@code gradCsrValues}</li>
 * </ol>
 *
 * <p>Math: for each COO entry {@code k} with {@code (i, j) = cooIndices[k]},
 * binary-search {@code csrColIdx[csrRowPtr[i] .. csrRowPtr[i+1])} for column {@code j};
 * {@code dCooValues[k] = gradCsrValues[foundPos]}.
 * No atomics: each COO entry maps to exactly one CSR slot.
 *
 * <p>This op is forward-only (no {@code doDiff}); it is the gradient leaf for
 * {@link CooToCsr#doDiff}.
 */
public class CooToCsrBp extends DynamicCustomOp {

    /** No-arg constructor required for op-registry reflection. */
    public CooToCsrBp() {}

    /**
     * Eager (INDArray) constructor.
     *
     * @param cooIndices    2D [coo_nnz, 2] INT — row/col index pairs of the original COO input
     * @param csrColIdx     1D [csr_nnz] INT32 — column indices of the CSR forward output
     * @param csrRowPtr     1D [rows+1] INT32 — row pointers of the CSR forward output
     * @param gradCsrValues 1D [csr_nnz] float — upstream gradient w.r.t. the CSR values output
     * @param rows          number of rows in the logical sparse matrix
     * @param cols          number of columns in the logical sparse matrix
     */
    public CooToCsrBp(INDArray cooIndices, INDArray csrColIdx, INDArray csrRowPtr,
                      INDArray gradCsrValues, long rows, long cols) {
        super(new INDArray[]{cooIndices, csrColIdx, csrRowPtr, gradCsrValues}, null);
        addIArgument(rows, cols);
    }

    /**
     * SameDiff (symbolic) constructor.
     *
     * @param sd            the SameDiff graph
     * @param cooIndices    SDVariable [coo_nnz, 2] INT — row/col index pairs (forward input[0])
     * @param csrColIdx     SDVariable [csr_nnz] INT32 — column indices (forward output[1])
     * @param csrRowPtr     SDVariable [rows+1] INT32 — row pointers (forward output[2])
     * @param gradCsrValues SDVariable [csr_nnz] float — upstream gradient w.r.t. CSR values
     * @param rows          number of rows in the logical sparse matrix
     * @param cols          number of columns in the logical sparse matrix
     */
    public CooToCsrBp(SameDiff sd,
                      SDVariable cooIndices, SDVariable csrColIdx, SDVariable csrRowPtr,
                      SDVariable gradCsrValues,
                      long rows, long cols) {
        super(sd, new SDVariable[]{cooIndices, csrColIdx, csrRowPtr, gradCsrValues});
        addIArgument(rows, cols);
    }

    @Override
    public String opName() {
        return "coo_to_csr_bp";
    }

    /**
     * Output data type: {@code dCooValues} has the same float dtype as
     * {@code gradCsrValues} (input index 3).
     */
    @Override
    public List<DataType> calculateOutputDataTypes(List<DataType> dataTypes) {
        return Collections.singletonList(dataTypes.get(3));
    }
}
