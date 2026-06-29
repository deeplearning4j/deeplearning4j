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
 * Backward pass for {@link DenseToCsr}: scatter upstream gradients from the CSR values output
 * back into the dense gradient matrix.
 *
 * <p>C++ op name: {@code dense_to_csr_bp}
 *
 * <p><b>Inputs (3 arrays):</b>
 * <ol>
 *   <li>{@code colIdx}     – 1D [nnz], INT32 — CSR column indices (forward output[1])</li>
 *   <li>{@code rowPtr}     – 1D [rows+1], INT32 — CSR row pointers (forward output[2])</li>
 *   <li>{@code gradValues} – 1D [nnz], float — upstream gradient w.r.t. the values output</li>
 * </ol>
 *
 * <p><b>Integer arguments (IArgs):</b> {@code rows}, {@code cols}
 *
 * <p><b>Output:</b>
 * <ol>
 *   <li>{@code dDense} – 2D [rows, cols], same float dtype as {@code gradValues} —
 *       gradient w.r.t. the dense input (pre-zeroed by the framework)</li>
 * </ol>
 *
 * <p>Math: {@code dDense[i, colIdx[e]] = gradValues[e]} for each entry {@code e}.
 * In a valid CSR pattern every (row, col) pair is unique, so no atomics are needed.
 *
 * <p>This op is forward-only (no {@code doDiff}); it is the gradient leaf for
 * {@link DenseToCsr#doDiff}.
 */
public class DenseToCsrBp extends DynamicCustomOp {

    /** No-arg constructor required for ImportClassMapping reflection. */
    public DenseToCsrBp() {}

    /**
     * Eager (INDArray) constructor.
     *
     * @param colIdx     1D [nnz] INT32 — CSR column indices (from forward output)
     * @param rowPtr     1D [rows+1] INT32 — CSR row pointers (from forward output)
     * @param gradValues 1D [nnz] float — upstream gradient w.r.t. the values output
     * @param rows       number of rows in the original dense matrix
     * @param cols       number of columns in the original dense matrix
     */
    public DenseToCsrBp(INDArray colIdx, INDArray rowPtr, INDArray gradValues,
                         long rows, long cols) {
        super(new INDArray[]{colIdx, rowPtr, gradValues}, null);
        addIArgument(rows, cols);
    }

    /**
     * SameDiff (symbolic) constructor.
     *
     * @param sd         SameDiff instance
     * @param colIdx     SDVariable [nnz] INT32 — CSR column indices
     * @param rowPtr     SDVariable [rows+1] INT32 — CSR row pointers
     * @param gradValues SDVariable [nnz] float — upstream gradient w.r.t. the values output
     * @param rows       number of rows
     * @param cols       number of columns
     */
    public DenseToCsrBp(SameDiff sd, SDVariable colIdx, SDVariable rowPtr,
                         SDVariable gradValues, long rows, long cols) {
        super(sd, new SDVariable[]{colIdx, rowPtr, gradValues});
        addIArgument(rows, cols);
    }

    @Override
    public String opName() {
        return "dense_to_csr_bp";
    }

    /**
     * Output data type: same float dtype as {@code gradValues} (input index 2).
     */
    @Override
    public List<DataType> calculateOutputDataTypes(List<DataType> dataTypes) {
        return Collections.singletonList(dataTypes.get(2));
    }
}
