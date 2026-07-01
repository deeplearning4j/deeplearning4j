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
 * Converts a CSC (Compressed Sparse Column) sparse tensor to a dense matrix.
 *
 * <p>C++ op name: {@code csc_to_dense}
 *
 * <p>Inputs:
 * <ol>
 *   <li>cscValues  – 1D [nnz], floating dtype — non-zero values in column-major order</li>
 *   <li>cscRowIdx  – 1D [nnz], INT32 — row index for each non-zero</li>
 *   <li>cscColPtr  – 1D [cols+1], INT32 — column pointers</li>
 * </ol>
 * Integer args: rows, cols
 *
 * <p>Output: dense matrix [rows, cols]
 */
public class CscToDense extends DynamicCustomOp {

    /** No-arg constructor required for ImportClassMapping reflection. */
    public CscToDense() {}

    /**
     * Construct from pre-built CSC component arrays.
     *
     * @param cscValues  1D [nnz] non-zero values in column-major order
     * @param cscRowIdx  1D [nnz] row indices (INT32)
     * @param cscColPtr  1D [cols+1] column pointers (INT32)
     * @param rows       number of rows in the dense output
     * @param cols       number of columns in the dense output
     */
    public CscToDense(INDArray cscValues, INDArray cscRowIdx, INDArray cscColPtr,
                      long rows, long cols) {
        super(new INDArray[]{cscValues, cscRowIdx, cscColPtr}, null);
        addIArgument(rows, cols);
    }

    /**
     * SameDiff constructor.
     *
     * @param sd         SameDiff instance
     * @param cscValues  SD variable for non-zero values
     * @param cscRowIdx  SD variable for row indices
     * @param cscColPtr  SD variable for column pointers
     * @param rows       number of rows
     * @param cols       number of columns
     */
    public CscToDense(SameDiff sd, SDVariable cscValues, SDVariable cscRowIdx,
                      SDVariable cscColPtr, long rows, long cols) {
        super(sd, new SDVariable[]{cscValues, cscRowIdx, cscColPtr});
        addIArgument(rows, cols);
    }

    @Override
    public String opName() {
        return "csc_to_dense";
    }

    @Override
    public List<DataType> calculateOutputDataTypes(List<DataType> dataTypes) {
        // Output dtype matches the values input (index 0)
        return Collections.singletonList(dataTypes.get(0));
    }

    @Override
    public List<SDVariable> doDiff(List<SDVariable> grads) {
        SDVariable gradDense = grads.get(0);
        long rows = iArguments.get(0);
        long cols = iArguments.get(1);
        SDVariable cscRowIdx = arg(1);
        SDVariable cscColPtr = arg(2);
        SDVariable dCscValues = new CscToDenseBp(sameDiff, cscRowIdx, cscColPtr, gradDense, rows, cols)
                .outputVariables()[0];
        return Arrays.asList(dCscValues, sameDiff.zerosLike(cscRowIdx), sameDiff.zerosLike(cscColPtr));
    }
}
