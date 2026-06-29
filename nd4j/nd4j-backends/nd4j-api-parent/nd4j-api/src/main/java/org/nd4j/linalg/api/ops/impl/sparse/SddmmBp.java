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
 * Backward pass for {@link Sddmm} (sampled dense-dense matrix multiplication).
 *
 * <p>Forward: {@code out[k at (i,j)] = D1[i,:] · D2[j,:]}
 *
 * <p>C++ op name: {@code sddmm_bp}
 * <ul>
 *   <li>Inputs:  rowPtr[rows+1 INT], colIdx[nnz INT], D1[rows,p], D2[cols,p],
 *                gradOut[nnz] (upstream gradient)</li>
 *   <li>IArgs:   rows, cols</li>
 *   <li>Outputs: dD1[rows,p], dD2[cols,p]</li>
 * </ul>
 *
 * <p>Gradient math:
 * <pre>
 *   dD1[i,:] += sum_{k: row=i} gradOut[k] * D2[j_k,:]
 *   dD2[j,:] += sum_{k: col=j} gradOut[k] * D1[i_k,:]
 * </pre>
 *
 * <p>This is a backward primitive: {@code doDiff} is not implemented.
 */
public class SddmmBp extends DynamicCustomOp {

    /** No-arg constructor required for op-registry reflection. */
    public SddmmBp() {}

    /**
     * Eager (INDArray) constructor.
     *
     * @param rowPtr   1D [rows+1] INT row pointers
     * @param colIdx   1D [nnz]   INT column indices
     * @param D1       2D [rows, p] left dense factor
     * @param D2       2D [cols, p] right dense factor
     * @param gradOut  1D [nnz] upstream gradient
     * @param rows     number of rows
     * @param cols     number of columns
     */
    public SddmmBp(INDArray rowPtr, INDArray colIdx, INDArray D1, INDArray D2,
                   INDArray gradOut, long rows, long cols) {
        super(new INDArray[]{rowPtr, colIdx, D1, D2, gradOut}, null);
        addIArgument(rows, cols);
    }

    /**
     * SameDiff (symbolic) constructor.
     *
     * @param sd       the SameDiff graph
     * @param rowPtr   SDVariable [rows+1] INT row pointers
     * @param colIdx   SDVariable [nnz] INT column indices
     * @param D1       SDVariable [rows, p] left dense factor
     * @param D2       SDVariable [cols, p] right dense factor
     * @param gradOut  SDVariable [nnz] upstream gradient
     * @param rows     number of rows
     * @param cols     number of columns
     */
    public SddmmBp(SameDiff sd, SDVariable rowPtr, SDVariable colIdx,
                   SDVariable D1, SDVariable D2, SDVariable gradOut,
                   long rows, long cols) {
        super(sd, new SDVariable[]{rowPtr, colIdx, D1, D2, gradOut});
        addIArgument(rows, cols);
    }

    @Override
    public String opName() { return "sddmm_bp"; }

    @Override
    public List<DataType> calculateOutputDataTypes(List<DataType> dataTypes) {
        // dD1: same dtype as D1 (input[2])
        // dD2: same dtype as D2 (input[3])
        return Arrays.asList(dataTypes.get(2), dataTypes.get(3));
    }
}
