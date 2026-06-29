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
 * Converts a CSR sparse tensor to a dense matrix.
 *
 * <p>C++ op name: {@code csr_to_dense}
 *
 * <p>Inputs:
 * <ol>
 *   <li>values – 1D [nnz], floating dtype</li>
 *   <li>colIdx – 1D [nnz], INT32 or INT64</li>
 *   <li>rowPtr – 1D [rows+1], same INT dtype as colIdx</li>
 * </ol>
 * Integer args: rows, cols
 * <p>Output: dense matrix [rows, cols]
 */
public class CsrToDense extends DynamicCustomOp {

    /** No-arg constructor required for ImportClassMapping reflection. */
    public CsrToDense() {}

    /**
     * Construct from pre-built CSR component arrays.
     *
     * @param values  1D [nnz] non-zero values
     * @param colIdx  1D [nnz] column indices (INT32/INT64)
     * @param rowPtr  1D [rows+1] row pointers (INT32/INT64)
     * @param rows    number of rows in the dense output
     * @param cols    number of columns in the dense output
     */
    public CsrToDense(INDArray values, INDArray colIdx, INDArray rowPtr, long rows, long cols) {
        super(new INDArray[]{values, colIdx, rowPtr}, null);
        addIArgument(rows, cols);
    }

    /**
     * SameDiff constructor.
     *
     * @param sd      SameDiff instance
     * @param values  SD variable for non-zero values
     * @param colIdx  SD variable for column indices
     * @param rowPtr  SD variable for row pointers
     * @param rows    number of rows
     * @param cols    number of columns
     */
    public CsrToDense(SameDiff sd, SDVariable values, SDVariable colIdx, SDVariable rowPtr,
                      long rows, long cols) {
        super(sd, new SDVariable[]{values, colIdx, rowPtr});
        addIArgument(rows, cols);
    }

    @Override
    public String opName() {
        return "csr_to_dense";
    }

    @Override
    public List<DataType> calculateOutputDataTypes(List<DataType> dataTypes) {
        // Output dtype matches the values input (index 0)
        return Collections.singletonList(dataTypes.get(0));
    }

    /**
     * Backward pass for {@code csr_to_dense}.
     *
     * <p>The forward op scatters {@code values[e]} into {@code dense[i, colIdx[e]]}.
     * The gradient reverses this: for each non-zero entry {@code e},
     * {@code dValues[e] = gradDense[i, colIdx[e]]} (a pure gather at the CSR pattern).
     * The structural inputs {@code colIdx} and {@code rowPtr} receive zero gradients.
     *
     * @param grads one-element list; {@code grads.get(0)} is the upstream gradient for the
     *              dense output
     * @return gradients for [values, colIdx, rowPtr]
     */
    @Override
    public List<SDVariable> doDiff(List<SDVariable> grads) {
        SDVariable gradDense = grads.get(0);

        long rows = iArguments.get(0);
        long cols = iArguments.get(1);

        SDVariable values = arg(0);
        SDVariable colIdx = arg(1);
        SDVariable rowPtr = arg(2);

        SDVariable dValues = new CsrToDenseBp(sameDiff, colIdx, rowPtr, gradDense, rows, cols)
                .outputVariables()[0];

        // colIdx and rowPtr are structural INT arrays — zero gradient
        return Arrays.asList(
                dValues,
                sameDiff.zerosLike(colIdx),
                sameDiff.zerosLike(rowPtr));
    }
}
