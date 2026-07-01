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
 * Converts a CSR (Compressed Sparse Row) sparse matrix to BSR (Block Sparse Row) format.
 *
 * <p>C++ op name: {@code csr_to_bsr}
 *
 * <p>Inputs:
 * <ol>
 *   <li>csrValues – 1D [nnz], floating dtype — non-zero values in CSR row-major order</li>
 *   <li>csrColIdx – 1D [nnz], INT32 — CSR column indices</li>
 *   <li>csrRowPtr – 1D [rows+1], INT32 — CSR row pointers</li>
 * </ol>
 * Integer args: rows, cols, blockDim
 *
 * <p>Preconditions (enforced by native op):
 * <ul>
 *   <li>{@code rows % blockDim == 0} — rows must be an integer multiple of the block dimension</li>
 *   <li>{@code cols % blockDim == 0} — cols must be an integer multiple of the block dimension</li>
 * </ul>
 *
 * <p>Outputs:
 * <ol>
 *   <li>bsrValues – 1D [nnzb * blockDim * blockDim], same dtype as csrValues —
 *       the non-zero blocks stored in row-major block order, each block stored in
 *       row-major element order</li>
 *   <li>bsrColIdx – 1D [nnzb], INT32 — block-column index for each stored block;
 *       nnzb is data-dependent (native shape function)</li>
 *   <li>bsrRowPtr – 1D [mb+1], INT32 — block-row pointers where mb = rows / blockDim</li>
 * </ol>
 *
 * <p>A block-row {@code br} stores blocks {@code bsrRowPtr[br]} through
 * {@code bsrRowPtr[br+1]-1} (inclusive). Block {@code k} occupies element rows
 * {@code br*blockDim .. (br+1)*blockDim-1} and column block {@code bsrColIdx[k]},
 * i.e., columns {@code bsrColIdx[k]*blockDim .. (bsrColIdx[k]+1)*blockDim-1}.
 */
public class CsrToBsr extends DynamicCustomOp {

    /** No-arg constructor required for ImportClassMapping reflection. */
    public CsrToBsr() {}

    /**
     * Construct from pre-built CSR component arrays.
     *
     * @param csrValues  1D [nnz] non-zero values
     * @param csrColIdx  1D [nnz] CSR column indices (INT32)
     * @param csrRowPtr  1D [rows+1] CSR row pointers (INT32)
     * @param rows       number of rows in the logical matrix (must be a multiple of blockDim)
     * @param cols       number of columns in the logical matrix (must be a multiple of blockDim)
     * @param blockDim   the square block size; both rows and cols must be exact multiples
     */
    public CsrToBsr(INDArray csrValues, INDArray csrColIdx, INDArray csrRowPtr,
                    long rows, long cols, long blockDim) {
        super(new INDArray[]{csrValues, csrColIdx, csrRowPtr}, null);
        addIArgument(rows, cols, blockDim);
    }

    /**
     * SameDiff constructor.
     *
     * @param sd         SameDiff instance
     * @param csrValues  SD variable for CSR non-zero values
     * @param csrColIdx  SD variable for CSR column indices
     * @param csrRowPtr  SD variable for CSR row pointers
     * @param rows       number of rows (must be a multiple of blockDim)
     * @param cols       number of columns (must be a multiple of blockDim)
     * @param blockDim   square block size
     */
    public CsrToBsr(SameDiff sd, SDVariable csrValues, SDVariable csrColIdx, SDVariable csrRowPtr,
                    long rows, long cols, long blockDim) {
        super(sd, new SDVariable[]{csrValues, csrColIdx, csrRowPtr});
        addIArgument(rows, cols, blockDim);
    }

    @Override
    public String opName() {
        return "csr_to_bsr";
    }

    @Override
    public List<DataType> calculateOutputDataTypes(List<DataType> dataTypes) {
        // Output 0 (bsrValues) matches the csrValues input dtype (index 0)
        // Output 1 (bsrColIdx) is INT32; output 2 (bsrRowPtr) is INT32
        return Arrays.asList(dataTypes.get(0), DataType.INT32, DataType.INT32);
    }

    @Override
    public List<SDVariable> doDiff(List<SDVariable> grads) {
        SDVariable gradBsrValues = grads.get(0);
        long rows     = iArguments.get(0);
        long cols     = iArguments.get(1);
        long blockDim = iArguments.get(2);
        SDVariable csrColIdx = arg(1);
        SDVariable csrRowPtr = arg(2);
        SDVariable[] fwdOuts = outputVariables();
        SDVariable bsrColIdx = fwdOuts[1];
        SDVariable bsrRowPtr = fwdOuts[2];
        SDVariable dCsrValues = new CsrToBsrBp(sameDiff, csrColIdx, csrRowPtr, bsrColIdx, bsrRowPtr, gradBsrValues, rows, cols, blockDim)
                .outputVariables()[0];
        return Arrays.asList(dCsrValues, sameDiff.zerosLike(csrColIdx), sameDiff.zerosLike(csrRowPtr));
    }
}
