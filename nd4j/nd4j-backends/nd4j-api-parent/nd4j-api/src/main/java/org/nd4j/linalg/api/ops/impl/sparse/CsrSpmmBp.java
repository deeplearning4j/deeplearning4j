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
 * Backward pass for {@link CsrSpmm}: C = op(A) * B.
 *
 * <p>C++ op name: {@code csr_spmm_bp}
 * <ul>
 *   <li>Inputs:  values[nnz], colIdx[nnz INT], rowPtr[rows+1 INT],
 *                B (forward input), gradC (upstream gradient)</li>
 *   <li>IArgs:   rows, cols, transposeA (0 or 1)</li>
 *   <li>Outputs: dAValues[nnz], dB</li>
 * </ul>
 *
 * <p>Gradient math:
 * <pre>
 *   transposeA=0: dAValues = SDDMM(rowPtr, colIdx, gradC, B)
 *                 dB       = A^T * gradC
 *   transposeA=1: dAValues = SDDMM(rowPtr, colIdx, B, gradC)
 *                 dB       = A   * gradC
 * </pre>
 *
 * <p>This is a backward primitive: {@code doDiff} is not implemented.
 */
public class CsrSpmmBp extends DynamicCustomOp {

    /** No-arg constructor required for op-registry reflection. */
    public CsrSpmmBp() {}

    /**
     * Eager (INDArray) constructor.
     *
     * @param values     1D [nnz] non-zero values of A (float dtype)
     * @param colIdx     1D [nnz] INT column indices
     * @param rowPtr     1D [rows+1] INT row pointers
     * @param B          Dense input matrix from the forward op
     * @param gradC      Upstream gradient (same shape as C)
     * @param rows       number of rows in A
     * @param cols       number of columns in A
     * @param transposeA if true the forward computed A^T*B
     */
    public CsrSpmmBp(INDArray values, INDArray colIdx, INDArray rowPtr,
                     INDArray B, INDArray gradC,
                     long rows, long cols, boolean transposeA) {
        super(new INDArray[]{values, colIdx, rowPtr, B, gradC}, null);
        addIArgument(rows, cols, transposeA ? 1L : 0L);
    }

    /**
     * SameDiff (symbolic) constructor.
     *
     * @param sd         the SameDiff graph
     * @param values     SDVariable [nnz] non-zero values
     * @param colIdx     SDVariable [nnz] INT column indices
     * @param rowPtr     SDVariable [rows+1] INT row pointers
     * @param B          SDVariable dense input matrix
     * @param gradC      SDVariable upstream gradient
     * @param rows       number of rows
     * @param cols       number of columns
     * @param transposeA if true the forward computed A^T*B
     */
    public CsrSpmmBp(SameDiff sd, SDVariable values, SDVariable colIdx, SDVariable rowPtr,
                     SDVariable B, SDVariable gradC, long rows, long cols, boolean transposeA) {
        super(sd, new SDVariable[]{values, colIdx, rowPtr, B, gradC});
        addIArgument(rows, cols, transposeA ? 1L : 0L);
    }

    @Override
    public String opName() { return "csr_spmm_bp"; }

    @Override
    public List<DataType> calculateOutputDataTypes(List<DataType> dataTypes) {
        // dAValues: same dtype as values (input[0])
        // dB:       same dtype as B      (input[3])
        return Arrays.asList(dataTypes.get(0), dataTypes.get(3));
    }
}
