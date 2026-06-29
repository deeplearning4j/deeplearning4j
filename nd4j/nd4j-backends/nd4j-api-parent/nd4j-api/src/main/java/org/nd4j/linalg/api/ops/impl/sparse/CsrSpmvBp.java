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
 * Backward pass for {@link CsrSpmv}: y = op(A) * x.
 *
 * <p>C++ op name: {@code csr_spmv_bp}
 * <ul>
 *   <li>Inputs:  values[nnz], colIdx[nnz INT], rowPtr[rows+1 INT],
 *                x (forward input), gradY (upstream gradient)</li>
 *   <li>IArgs:   rows, cols, transposeA (0 or 1)</li>
 *   <li>Outputs: dAValues[nnz], dx</li>
 * </ul>
 *
 * <p>Gradient math:
 * <pre>
 *   transposeA=0: dAValues[k at (i,j)] = gradY[i] * x[j]
 *                 dx = A^T * gradY
 *   transposeA=1: dAValues[k at (i,j)] = gradY[j] * x[i]
 *                 dx = A   * gradY
 * </pre>
 *
 * <p>This is a backward primitive: {@code doDiff} is not implemented.
 */
public class CsrSpmvBp extends DynamicCustomOp {

    /** No-arg constructor required for op-registry reflection. */
    public CsrSpmvBp() {}

    /**
     * Eager (INDArray) constructor.
     *
     * @param values     1D [nnz] non-zero values of A (float dtype)
     * @param colIdx     1D [nnz] INT column indices
     * @param rowPtr     1D [rows+1] INT row pointers
     * @param x          Dense input vector from the forward op
     * @param gradY      Upstream gradient (same shape as y)
     * @param rows       number of rows in A
     * @param cols       number of columns in A
     * @param transposeA if true the forward computed A^T*x
     */
    public CsrSpmvBp(INDArray values, INDArray colIdx, INDArray rowPtr,
                     INDArray x, INDArray gradY,
                     long rows, long cols, boolean transposeA) {
        super(new INDArray[]{values, colIdx, rowPtr, x, gradY}, null);
        addIArgument(rows, cols, transposeA ? 1L : 0L);
    }

    /**
     * SameDiff (symbolic) constructor.
     *
     * @param sd         the SameDiff graph
     * @param values     SDVariable [nnz] non-zero values
     * @param colIdx     SDVariable [nnz] INT column indices
     * @param rowPtr     SDVariable [rows+1] INT row pointers
     * @param x          SDVariable dense input vector
     * @param gradY      SDVariable upstream gradient
     * @param rows       number of rows
     * @param cols       number of columns
     * @param transposeA if true the forward computed A^T*x
     */
    public CsrSpmvBp(SameDiff sd, SDVariable values, SDVariable colIdx, SDVariable rowPtr,
                     SDVariable x, SDVariable gradY, long rows, long cols, boolean transposeA) {
        super(sd, new SDVariable[]{values, colIdx, rowPtr, x, gradY});
        addIArgument(rows, cols, transposeA ? 1L : 0L);
    }

    @Override
    public String opName() { return "csr_spmv_bp"; }

    @Override
    public List<DataType> calculateOutputDataTypes(List<DataType> dataTypes) {
        // dAValues: same dtype as values (input[0])
        // dx:       same dtype as x      (input[3])
        return Arrays.asList(dataTypes.get(0), dataTypes.get(3));
    }
}
