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
 * Backward pass for {@link BsrToDense}: gathers the upstream dense gradient back into BSR block values.
 *
 * <p>C++ op name: {@code bsr_to_dense_bp}
 *
 * <p><b>Inputs (3 arrays):</b>
 * <ol>
 *   <li>{@code bsrColIdx} – 1D [nnzb], INT — block column indices</li>
 *   <li>{@code bsrRowPtr} – 1D [mb+1], INT — block row pointers</li>
 *   <li>{@code gradDense} – 2D [rows, cols], float — upstream gradient w.r.t. dense output</li>
 * </ol>
 *
 * <p><b>Integer arguments (IArgs):</b> {@code rows}, {@code cols}, {@code blockDim}
 *
 * <p><b>Output (1 array):</b>
 * <ol>
 *   <li>{@code dBsrValues} – 1D [nnzb*blockDim*blockDim], same float dtype as {@code gradDense}</li>
 * </ol>
 *
 * <p>This op is a backward primitive: {@code doDiff} is not implemented.
 */
public class BsrToDenseBp extends DynamicCustomOp {

    /** No-arg constructor required for ImportClassMapping reflection. */
    public BsrToDenseBp() {}

    /**
     * Eager (INDArray) constructor.
     *
     * @param bsrColIdx 1D [nnzb] INT block column indices
     * @param bsrRowPtr 1D [mb+1] INT block row pointers
     * @param gradDense 2D [rows, cols] float upstream gradient w.r.t. dense output
     * @param rows      number of rows in the logical dense matrix
     * @param cols      number of columns in the logical dense matrix
     * @param blockDim  the square block size
     */
    public BsrToDenseBp(INDArray bsrColIdx, INDArray bsrRowPtr, INDArray gradDense,
                        long rows, long cols, long blockDim) {
        super(new INDArray[]{bsrColIdx, bsrRowPtr, gradDense}, null);
        addIArgument(rows, cols, blockDim);
    }

    /**
     * SameDiff (symbolic) constructor.
     *
     * @param sd        the SameDiff graph
     * @param bsrColIdx SDVariable [nnzb] INT block column indices
     * @param bsrRowPtr SDVariable [mb+1] INT block row pointers
     * @param gradDense SDVariable [rows, cols] float upstream gradient w.r.t. dense output
     * @param rows      number of rows in the logical dense matrix
     * @param cols      number of columns in the logical dense matrix
     * @param blockDim  the square block size
     */
    public BsrToDenseBp(SameDiff sd,
                        SDVariable bsrColIdx, SDVariable bsrRowPtr, SDVariable gradDense,
                        long rows, long cols, long blockDim) {
        super(sd, new SDVariable[]{bsrColIdx, bsrRowPtr, gradDense});
        addIArgument(rows, cols, blockDim);
    }

    @Override
    public String opName() {
        return "bsr_to_dense_bp";
    }

    @Override
    public int getNumOutputs() {
        return 1;
    }

    /**
     * Output data type: same float dtype as {@code gradDense} (input index 2).
     */
    @Override
    public List<DataType> calculateOutputDataTypes(List<DataType> dataTypes) {
        return Collections.singletonList(dataTypes.get(2));
    }
}
