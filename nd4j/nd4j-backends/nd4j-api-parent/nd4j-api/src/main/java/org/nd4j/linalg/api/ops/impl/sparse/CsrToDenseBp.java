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
 * Backward pass for {@link CsrToDense}: gather gradient values from the dense upstream gradient
 * at the CSR sparsity pattern.
 *
 * <p>C++ op name: {@code csr_to_dense_bp}
 *
 * <p><b>Inputs (3 arrays):</b>
 * <ol>
 *   <li>{@code colIdx}    – 1D [nnz], INT32/INT64 — CSR column indices (forward input[1])</li>
 *   <li>{@code rowPtr}    – 1D [rows+1], same INT dtype — CSR row pointers (forward input[2])</li>
 *   <li>{@code gradDense} – 2D [rows, cols], float — upstream gradient w.r.t. dense output</li>
 * </ol>
 *
 * <p><b>Integer arguments (IArgs):</b> {@code rows}, {@code cols}
 *
 * <p><b>Output:</b>
 * <ol>
 *   <li>{@code dValues} – 1D [nnz], same float dtype as {@code gradDense}</li>
 * </ol>
 *
 * <p>Math: {@code dValues[e] = gradDense[i, colIdx[e]]} where {@code i} is the row of
 * entry {@code e} in the CSR pattern.  Pure gather — no atomics.
 *
 * <p>This op is forward-only (no {@code doDiff}); it is the gradient leaf for
 * {@link CsrToDense#doDiff}.
 */
public class CsrToDenseBp extends DynamicCustomOp {

    /** No-arg constructor required for ImportClassMapping reflection. */
    public CsrToDenseBp() {}

    /**
     * Eager (INDArray) constructor.
     *
     * @param colIdx    1D [nnz] INT32/INT64 — CSR column indices
     * @param rowPtr    1D [rows+1] INT32/INT64 — CSR row pointers
     * @param gradDense 2D [rows, cols] float — upstream gradient w.r.t. dense output
     * @param rows      number of rows
     * @param cols      number of columns
     */
    public CsrToDenseBp(INDArray colIdx, INDArray rowPtr, INDArray gradDense, long rows, long cols) {
        super(new INDArray[]{colIdx, rowPtr, gradDense}, null);
        addIArgument(rows, cols);
    }

    /**
     * SameDiff (symbolic) constructor.
     *
     * @param sd        SameDiff instance
     * @param colIdx    SDVariable [nnz] INT32/INT64 — CSR column indices
     * @param rowPtr    SDVariable [rows+1] INT32/INT64 — CSR row pointers
     * @param gradDense SDVariable [rows, cols] float — upstream gradient w.r.t. dense output
     * @param rows      number of rows
     * @param cols      number of columns
     */
    public CsrToDenseBp(SameDiff sd, SDVariable colIdx, SDVariable rowPtr,
                         SDVariable gradDense, long rows, long cols) {
        super(sd, new SDVariable[]{colIdx, rowPtr, gradDense});
        addIArgument(rows, cols);
    }

    @Override
    public String opName() {
        return "csr_to_dense_bp";
    }

    /**
     * Output data type: same float dtype as {@code gradDense} (input index 2).
     */
    @Override
    public List<DataType> calculateOutputDataTypes(List<DataType> dataTypes) {
        return Collections.singletonList(dataTypes.get(2));
    }
}
