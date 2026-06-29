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
import java.util.ArrayList;

/**
 * Converts COO (Coordinate) sparse representation to CSR (Compressed Sparse Row) format.
 *
 * <p>C++ op name: {@code coo_to_csr}
 *
 * <p>Inputs:
 * <ol>
 *   <li>indices – 2D [nnz, 2], INT64 — row/col index pairs for each non-zero</li>
 *   <li>values  – 1D [nnz], floating dtype — the non-zero values</li>
 * </ol>
 * Integer args: rows, cols
 *
 * <p>Outputs:
 * <ol>
 *   <li>values  – 1D [nnz], same dtype as input values (index 1)</li>
 *   <li>colIdx  – 1D [nnz], INT32 — column index for each non-zero</li>
 *   <li>rowPtr  – 1D [rows+1], INT32 — row pointers</li>
 * </ol>
 */
public class CooToCsr extends DynamicCustomOp {

    /** No-arg constructor required for ImportClassMapping reflection. */
    public CooToCsr() {}

    /**
     * Construct from pre-built COO component arrays.
     *
     * @param indices 2D [nnz, 2] INT64 row/col index pairs
     * @param values  1D [nnz] non-zero values
     * @param rows    number of rows in the logical dense shape
     * @param cols    number of columns in the logical dense shape
     */
    public CooToCsr(INDArray indices, INDArray values, long rows, long cols) {
        super(new INDArray[]{indices, values}, null);
        addIArgument(rows, cols);
    }

    /**
     * SameDiff constructor.
     *
     * @param sd      SameDiff instance
     * @param indices SD variable for the COO index pairs
     * @param values  SD variable for the non-zero values
     * @param rows    number of rows
     * @param cols    number of columns
     */
    public CooToCsr(SameDiff sd, SDVariable indices, SDVariable values, long rows, long cols) {
        super(sd, new SDVariable[]{indices, values});
        addIArgument(rows, cols);
    }

    @Override
    public String opName() {
        return "coo_to_csr";
    }

    @Override
    public List<DataType> calculateOutputDataTypes(List<DataType> dataTypes) {
        // Output 0 (values) has same dtype as the values input (index 1)
        // Output 1 (colIdx) is INT32; output 2 (rowPtr) is INT32
        return Arrays.asList(dataTypes.get(1), DataType.INT32, DataType.INT32);
    }

    /**
     * Gradient w.r.t. the COO inputs.
     *
     * <p>Forward outputs: [0] csrValues [csr_nnz], [1] csrColIdx [csr_nnz], [2] csrRowPtr [rows+1]
     * <br>
     * {@code grads.get(0)} = upstream gradient for {@code csrValues} [csr_nnz] (float).
     * {@code grads.get(1)} and {@code grads.get(2)} correspond to structural INT outputs — ignored.
     *
     * <p>Returns: {@code [dCooIndices (zero), dCooValues]} — gradients for each forward input.
     */
    @Override
    public List<SDVariable> doDiff(List<SDVariable> grads) {
        SDVariable gradCsrValues = grads.get(0);       // [csr_nnz] float
        SDVariable cooIndices    = arg(0);              // [coo_nnz, 2] INT
        SDVariable[] fwdOuts     = outputVariables();
        SDVariable csrColIdx     = fwdOuts[1];          // [csr_nnz] INT32
        SDVariable csrRowPtr     = fwdOuts[2];          // [rows+1]  INT32

        long rows = iArguments.get(0);
        long cols = iArguments.get(1);

        SDVariable dCooValues = new CooToCsrBp(sameDiff,
                cooIndices, csrColIdx, csrRowPtr, gradCsrValues,
                rows, cols).outputVariables()[0];

        // Return gradients in forward-input order:
        //   [0] cooIndices — structural INT, zero gradient
        //   [1] cooValues  — dCooValues from the bp op
        List<SDVariable> result = new ArrayList<>();
        result.add(sameDiff.zerosLike(cooIndices));
        result.add(dCooValues);
        return result;
    }
}
