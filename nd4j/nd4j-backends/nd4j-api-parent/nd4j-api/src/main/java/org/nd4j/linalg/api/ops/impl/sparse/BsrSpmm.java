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
 * Sparse matrix-matrix multiplication in BSR (Block Sparse Row) format:
 * {@code C = A_bsr · B}, where A is stored in BSR and B is a dense matrix.
 *
 * <p>C++ op name: {@code bsr_spmm}
 *
 * <p>Inputs:
 * <ol>
 *   <li>bsrValues – 1D [nnzb * blockDim * blockDim], floating dtype —
 *       non-zero block values of A in row-major block order</li>
 *   <li>bsrColIdx – 1D [nnzb], INT32 — block-column index for each stored block of A</li>
 *   <li>bsrRowPtr – 1D [mb+1], INT32 — block-row pointers of A, mb = rows / blockDim</li>
 *   <li>B – 2D [cols, n], floating dtype — the dense right-hand matrix</li>
 * </ol>
 * Integer args: rows, cols, blockDim
 *
 * <p>Output:
 * <ol>
 *   <li>C – 2D [rows, n], same dtype as bsrValues — the dense result matrix</li>
 * </ol>
 *
 * <p>The computation is equivalent to {@code toDense(A_bsr).mmul(B)} but uses the
 * sparse BSR structure to skip zero blocks.
 */
public class BsrSpmm extends DynamicCustomOp {

    /** No-arg constructor required for ImportClassMapping reflection. */
    public BsrSpmm() {}

    /**
     * Construct from pre-built BSR component arrays and a dense right-hand matrix.
     *
     * @param bsrValues  1D [nnzb * blockDim * blockDim] non-zero block values of A
     * @param bsrColIdx  1D [nnzb] block-column indices (INT32)
     * @param bsrRowPtr  1D [mb+1] block-row pointers (INT32), mb = rows / blockDim
     * @param B          2D [cols, n] dense right-hand matrix
     * @param rows       number of rows in A (and rows of output C)
     * @param cols       number of columns in A (and rows of B)
     * @param blockDim   the square block size used in the BSR representation
     */
    public BsrSpmm(INDArray bsrValues, INDArray bsrColIdx, INDArray bsrRowPtr, INDArray B,
                   long rows, long cols, long blockDim) {
        super(new INDArray[]{bsrValues, bsrColIdx, bsrRowPtr, B}, null);
        addIArgument(rows, cols, blockDim);
    }

    /**
     * SameDiff constructor.
     *
     * @param sd         SameDiff instance
     * @param bsrValues  SD variable for BSR non-zero block values
     * @param bsrColIdx  SD variable for BSR block-column indices
     * @param bsrRowPtr  SD variable for BSR block-row pointers
     * @param B          SD variable for the dense right-hand matrix [cols, n]
     * @param rows       number of rows in A
     * @param cols       number of columns in A
     * @param blockDim   the square block size
     */
    public BsrSpmm(SameDiff sd, SDVariable bsrValues, SDVariable bsrColIdx, SDVariable bsrRowPtr,
                   SDVariable B, long rows, long cols, long blockDim) {
        super(sd, new SDVariable[]{bsrValues, bsrColIdx, bsrRowPtr, B});
        addIArgument(rows, cols, blockDim);
    }

    @Override
    public String opName() {
        return "bsr_spmm";
    }

    @Override
    public List<DataType> calculateOutputDataTypes(List<DataType> dataTypes) {
        // Single dense output C matching the bsrValues input dtype (index 0)
        return Collections.singletonList(dataTypes.get(0));
    }

    @Override
    public List<SDVariable> doDiff(List<SDVariable> grads) {
        SDVariable dC = grads.get(0);
        long rows     = iArguments.get(0);
        long cols     = iArguments.get(1);
        long blockDim = iArguments.get(2);
        SDVariable bsrValues = arg(0);
        SDVariable bsrColIdx = arg(1);
        SDVariable bsrRowPtr = arg(2);
        SDVariable B         = arg(3);
        SDVariable[] bpOut = new BsrSpmmBp(sameDiff, bsrValues, bsrColIdx, bsrRowPtr, B, dC, rows, cols, blockDim)
                .outputVariables();
        return Arrays.asList(bpOut[0], sameDiff.zerosLike(bsrColIdx), sameDiff.zerosLike(bsrRowPtr), bpOut[1]);
    }
}
