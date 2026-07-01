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
 * Backward pass for {@link CsrToBsr}: scatters the upstream BSR block gradient back into CSR values.
 *
 * <p>C++ op name: {@code csr_to_bsr_bp}
 *
 * <p><b>Inputs (5 arrays):</b>
 * <ol>
 *   <li>{@code csrColIdx}    – 1D [nnz], INT — CSR column indices</li>
 *   <li>{@code csrRowPtr}    – 1D [rows+1], INT — CSR row pointers</li>
 *   <li>{@code bsrColIdx}    – 1D [nnzb], INT — BSR block column indices (forward output[1])</li>
 *   <li>{@code bsrRowPtr}    – 1D [mb+1], INT — BSR block row pointers (forward output[2])</li>
 *   <li>{@code gradBsrValues}– 1D [nnzb*blockDim*blockDim], float — upstream gradient w.r.t. BSR values</li>
 * </ol>
 *
 * <p><b>Integer arguments (IArgs):</b> {@code rows}, {@code cols}, {@code blockDim}
 *
 * <p><b>Output (1 array):</b>
 * <ol>
 *   <li>{@code dCsrValues} – 1D [nnz], same float dtype as {@code gradBsrValues}</li>
 * </ol>
 *
 * <p>This op is a backward primitive: {@code doDiff} is not implemented.
 */
public class CsrToBsrBp extends DynamicCustomOp {

    /** No-arg constructor required for ImportClassMapping reflection. */
    public CsrToBsrBp() {}

    /**
     * Eager (INDArray) constructor.
     *
     * @param csrColIdx     1D [nnz] INT CSR column indices
     * @param csrRowPtr     1D [rows+1] INT CSR row pointers
     * @param bsrColIdx     1D [nnzb] INT BSR block column indices
     * @param bsrRowPtr     1D [mb+1] INT BSR block row pointers
     * @param gradBsrValues 1D [nnzb*blockDim*blockDim] float upstream gradient w.r.t. BSR values
     * @param rows          number of rows (must be a multiple of blockDim)
     * @param cols          number of columns (must be a multiple of blockDim)
     * @param blockDim      the square block size
     */
    public CsrToBsrBp(INDArray csrColIdx, INDArray csrRowPtr,
                      INDArray bsrColIdx, INDArray bsrRowPtr,
                      INDArray gradBsrValues,
                      long rows, long cols, long blockDim) {
        super(new INDArray[]{csrColIdx, csrRowPtr, bsrColIdx, bsrRowPtr, gradBsrValues}, null);
        addIArgument(rows, cols, blockDim);
    }

    /**
     * SameDiff (symbolic) constructor.
     *
     * @param sd            the SameDiff graph
     * @param csrColIdx     SDVariable [nnz] INT CSR column indices
     * @param csrRowPtr     SDVariable [rows+1] INT CSR row pointers
     * @param bsrColIdx     SDVariable [nnzb] INT BSR block column indices
     * @param bsrRowPtr     SDVariable [mb+1] INT BSR block row pointers
     * @param gradBsrValues SDVariable [nnzb*blockDim*blockDim] float upstream gradient w.r.t. BSR values
     * @param rows          number of rows (must be a multiple of blockDim)
     * @param cols          number of columns (must be a multiple of blockDim)
     * @param blockDim      the square block size
     */
    public CsrToBsrBp(SameDiff sd,
                      SDVariable csrColIdx, SDVariable csrRowPtr,
                      SDVariable bsrColIdx, SDVariable bsrRowPtr,
                      SDVariable gradBsrValues,
                      long rows, long cols, long blockDim) {
        super(sd, new SDVariable[]{csrColIdx, csrRowPtr, bsrColIdx, bsrRowPtr, gradBsrValues});
        addIArgument(rows, cols, blockDim);
    }

    @Override
    public String opName() {
        return "csr_to_bsr_bp";
    }

    @Override
    public int getNumOutputs() {
        return 1;
    }

    /**
     * Output data type: same float dtype as {@code gradBsrValues} (input index 4).
     */
    @Override
    public List<DataType> calculateOutputDataTypes(List<DataType> dataTypes) {
        return Collections.singletonList(dataTypes.get(4));
    }
}
