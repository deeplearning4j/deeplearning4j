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
 * Converts a BSR (Block Sparse Row) sparse matrix to a dense matrix.
 *
 * <p>C++ op name: {@code bsr_to_dense}
 *
 * <p>Inputs:
 * <ol>
 *   <li>bsrValues – 1D [nnzb * blockDim * blockDim], floating dtype —
 *       non-zero block values in row-major block order</li>
 *   <li>bsrColIdx – 1D [nnzb], INT32 — block-column index for each stored block</li>
 *   <li>bsrRowPtr – 1D [mb+1], INT32 — block-row pointers where mb = rows / blockDim</li>
 * </ol>
 * Integer args: rows, cols, blockDim
 *
 * <p>Output:
 * <ol>
 *   <li>dense – 2D [rows, cols], same dtype as bsrValues</li>
 * </ol>
 */
public class BsrToDense extends DynamicCustomOp {

    /** No-arg constructor required for ImportClassMapping reflection. */
    public BsrToDense() {}

    /**
     * Construct from pre-built BSR component arrays.
     *
     * @param bsrValues  1D [nnzb * blockDim * blockDim] non-zero block values
     * @param bsrColIdx  1D [nnzb] block-column indices (INT32)
     * @param bsrRowPtr  1D [mb+1] block-row pointers (INT32), mb = rows / blockDim
     * @param rows       number of rows in the logical dense matrix
     * @param cols       number of columns in the logical dense matrix
     * @param blockDim   the square block size
     */
    public BsrToDense(INDArray bsrValues, INDArray bsrColIdx, INDArray bsrRowPtr,
                      long rows, long cols, long blockDim) {
        super(new INDArray[]{bsrValues, bsrColIdx, bsrRowPtr}, null);
        addIArgument(rows, cols, blockDim);
    }

    /**
     * SameDiff constructor.
     *
     * @param sd         SameDiff instance
     * @param bsrValues  SD variable for BSR non-zero block values
     * @param bsrColIdx  SD variable for BSR block-column indices
     * @param bsrRowPtr  SD variable for BSR block-row pointers
     * @param rows       number of rows in the logical dense matrix
     * @param cols       number of columns in the logical dense matrix
     * @param blockDim   the square block size
     */
    public BsrToDense(SameDiff sd, SDVariable bsrValues, SDVariable bsrColIdx, SDVariable bsrRowPtr,
                      long rows, long cols, long blockDim) {
        super(sd, new SDVariable[]{bsrValues, bsrColIdx, bsrRowPtr});
        addIArgument(rows, cols, blockDim);
    }

    @Override
    public String opName() {
        return "bsr_to_dense";
    }

    @Override
    public List<DataType> calculateOutputDataTypes(List<DataType> dataTypes) {
        // Single dense output matching the bsrValues input dtype (index 0)
        return Collections.singletonList(dataTypes.get(0));
    }
}
