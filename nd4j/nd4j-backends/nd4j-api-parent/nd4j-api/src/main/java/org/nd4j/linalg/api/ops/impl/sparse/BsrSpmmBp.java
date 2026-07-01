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
 * Backward pass for {@link BsrSpmm}: C = A_bsr · B.
 *
 * <p>C++ op name: {@code bsr_spmm_bp}
 *
 * <p><b>Inputs (5 arrays):</b>
 * <ol>
 *   <li>{@code bsrValues} – 1D [nnzb*blockDim*blockDim], float — BSR non-zero block values</li>
 *   <li>{@code bsrColIdx} – 1D [nnzb], INT — block column indices</li>
 *   <li>{@code bsrRowPtr} – 1D [mb+1], INT — block row pointers</li>
 *   <li>{@code B}         – 2D [cols, n], float — dense right-hand matrix</li>
 *   <li>{@code gradC}     – 2D [rows, n], float — upstream gradient w.r.t. C</li>
 * </ol>
 *
 * <p><b>Integer arguments (IArgs):</b> {@code rows}, {@code cols}, {@code blockDim}
 *
 * <p><b>Outputs (2 arrays):</b>
 * <ol>
 *   <li>{@code dBsrValues} – 1D [nnzb*blockDim*blockDim], same dtype as {@code bsrValues}</li>
 *   <li>{@code dB}         – 2D [cols, n], same dtype as {@code B}</li>
 * </ol>
 *
 * <p>This op is a backward primitive: {@code doDiff} is not implemented.
 */
public class BsrSpmmBp extends DynamicCustomOp {

    /** No-arg constructor required for ImportClassMapping reflection. */
    public BsrSpmmBp() {}

    /**
     * Eager (INDArray) constructor.
     *
     * @param bsrValues 1D [nnzb*blockDim*blockDim] non-zero block values of A
     * @param bsrColIdx 1D [nnzb] INT block column indices
     * @param bsrRowPtr 1D [mb+1] INT block row pointers
     * @param B         2D [cols, n] dense right-hand matrix
     * @param gradC     2D [rows, n] upstream gradient w.r.t. C
     * @param rows      number of rows in A
     * @param cols      number of columns in A
     * @param blockDim  the square block size
     */
    public BsrSpmmBp(INDArray bsrValues, INDArray bsrColIdx, INDArray bsrRowPtr,
                     INDArray B, INDArray gradC,
                     long rows, long cols, long blockDim) {
        super(new INDArray[]{bsrValues, bsrColIdx, bsrRowPtr, B, gradC}, null);
        addIArgument(rows, cols, blockDim);
    }

    /**
     * SameDiff (symbolic) constructor.
     *
     * @param sd        the SameDiff graph
     * @param bsrValues SDVariable [nnzb*blockDim*blockDim] BSR non-zero block values
     * @param bsrColIdx SDVariable [nnzb] INT block column indices
     * @param bsrRowPtr SDVariable [mb+1] INT block row pointers
     * @param B         SDVariable [cols, n] dense right-hand matrix
     * @param gradC     SDVariable [rows, n] upstream gradient w.r.t. C
     * @param rows      number of rows in A
     * @param cols      number of columns in A
     * @param blockDim  the square block size
     */
    public BsrSpmmBp(SameDiff sd,
                     SDVariable bsrValues, SDVariable bsrColIdx, SDVariable bsrRowPtr,
                     SDVariable B, SDVariable gradC,
                     long rows, long cols, long blockDim) {
        super(sd, new SDVariable[]{bsrValues, bsrColIdx, bsrRowPtr, B, gradC});
        addIArgument(rows, cols, blockDim);
    }

    @Override
    public String opName() {
        return "bsr_spmm_bp";
    }

    @Override
    public int getNumOutputs() {
        return 2;
    }

    /**
     * Output data types:
     * <ol>
     *   <li>dBsrValues – same dtype as {@code bsrValues} (input index 0)</li>
     *   <li>dB         – same dtype as {@code B} (input index 3)</li>
     * </ol>
     */
    @Override
    public List<DataType> calculateOutputDataTypes(List<DataType> dataTypes) {
        return Arrays.asList(dataTypes.get(0), dataTypes.get(3));
    }
}
