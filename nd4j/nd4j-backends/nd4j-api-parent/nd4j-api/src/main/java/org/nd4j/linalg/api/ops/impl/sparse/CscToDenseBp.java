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
 * Backward pass for {@link CscToDense}: gathers the upstream dense gradient back into CSC values.
 *
 * <p>C++ op name: {@code csc_to_dense_bp}
 *
 * <p><b>Inputs (3 arrays):</b>
 * <ol>
 *   <li>{@code cscRowIdx} – 1D [nnz], INT — row indices</li>
 *   <li>{@code cscColPtr} – 1D [cols+1], INT — column pointers</li>
 *   <li>{@code gradDense} – 2D [rows, cols], float — upstream gradient w.r.t. dense output</li>
 * </ol>
 *
 * <p><b>Integer arguments (IArgs):</b> {@code rows}, {@code cols}
 *
 * <p><b>Output (1 array):</b>
 * <ol>
 *   <li>{@code dCscValues} – 1D [nnz], same float dtype as {@code gradDense}</li>
 * </ol>
 *
 * <p>This op is a backward primitive: {@code doDiff} is not implemented.
 */
public class CscToDenseBp extends DynamicCustomOp {

    /** No-arg constructor required for ImportClassMapping reflection. */
    public CscToDenseBp() {}

    /**
     * Eager (INDArray) constructor.
     *
     * @param cscRowIdx 1D [nnz] INT row indices
     * @param cscColPtr 1D [cols+1] INT column pointers
     * @param gradDense 2D [rows, cols] float upstream gradient w.r.t. dense output
     * @param rows      number of rows in the dense matrix
     * @param cols      number of columns in the dense matrix
     */
    public CscToDenseBp(INDArray cscRowIdx, INDArray cscColPtr, INDArray gradDense,
                        long rows, long cols) {
        super(new INDArray[]{cscRowIdx, cscColPtr, gradDense}, null);
        addIArgument(rows, cols);
    }

    /**
     * SameDiff (symbolic) constructor.
     *
     * @param sd        the SameDiff graph
     * @param cscRowIdx SDVariable [nnz] INT row indices
     * @param cscColPtr SDVariable [cols+1] INT column pointers
     * @param gradDense SDVariable [rows, cols] float upstream gradient w.r.t. dense output
     * @param rows      number of rows in the dense matrix
     * @param cols      number of columns in the dense matrix
     */
    public CscToDenseBp(SameDiff sd,
                        SDVariable cscRowIdx, SDVariable cscColPtr, SDVariable gradDense,
                        long rows, long cols) {
        super(sd, new SDVariable[]{cscRowIdx, cscColPtr, gradDense});
        addIArgument(rows, cols);
    }

    @Override
    public String opName() {
        return "csc_to_dense_bp";
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
