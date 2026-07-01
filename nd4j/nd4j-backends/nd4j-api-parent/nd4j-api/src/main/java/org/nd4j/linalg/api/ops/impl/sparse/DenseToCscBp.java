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
 * Backward pass for {@link DenseToCsc}: scatters CSC value gradients back into a dense matrix.
 *
 * <p>C++ op name: {@code dense_to_csc_bp}
 *
 * <p><b>Inputs (3 arrays):</b>
 * <ol>
 *   <li>{@code cscRowIdx}    – 1D [nnz], INT — row indices (forward output[1])</li>
 *   <li>{@code cscColPtr}    – 1D [cols+1], INT — column pointers (forward output[2])</li>
 *   <li>{@code gradCscValues}– 1D [nnz], float — upstream gradient w.r.t. cscValues</li>
 * </ol>
 *
 * <p><b>Integer arguments (IArgs):</b> {@code rows}, {@code cols}
 *
 * <p><b>Output (1 array):</b>
 * <ol>
 *   <li>{@code dDense} – 2D [rows, cols], same float dtype as {@code gradCscValues}
 *       (scatter back into dense)</li>
 * </ol>
 *
 * <p>This op is a backward primitive: {@code doDiff} is not implemented.
 */
public class DenseToCscBp extends DynamicCustomOp {

    /** No-arg constructor required for ImportClassMapping reflection. */
    public DenseToCscBp() {}

    /**
     * Eager (INDArray) constructor.
     *
     * @param cscRowIdx     1D [nnz] INT row indices
     * @param cscColPtr     1D [cols+1] INT column pointers
     * @param gradCscValues 1D [nnz] float upstream gradient w.r.t. cscValues
     * @param rows          number of rows in the original dense matrix
     * @param cols          number of columns in the original dense matrix
     */
    public DenseToCscBp(INDArray cscRowIdx, INDArray cscColPtr, INDArray gradCscValues,
                        long rows, long cols) {
        super(new INDArray[]{cscRowIdx, cscColPtr, gradCscValues}, null);
        addIArgument(rows, cols);
    }

    /**
     * SameDiff (symbolic) constructor.
     *
     * @param sd            the SameDiff graph
     * @param cscRowIdx     SDVariable [nnz] INT row indices
     * @param cscColPtr     SDVariable [cols+1] INT column pointers
     * @param gradCscValues SDVariable [nnz] float upstream gradient w.r.t. cscValues
     * @param rows          number of rows in the original dense matrix
     * @param cols          number of columns in the original dense matrix
     */
    public DenseToCscBp(SameDiff sd,
                        SDVariable cscRowIdx, SDVariable cscColPtr, SDVariable gradCscValues,
                        long rows, long cols) {
        super(sd, new SDVariable[]{cscRowIdx, cscColPtr, gradCscValues});
        addIArgument(rows, cols);
    }

    @Override
    public String opName() {
        return "dense_to_csc_bp";
    }

    @Override
    public int getNumOutputs() {
        return 1;
    }

    /**
     * Output data type: same float dtype as {@code gradCscValues} (input index 2).
     */
    @Override
    public List<DataType> calculateOutputDataTypes(List<DataType> dataTypes) {
        return Collections.singletonList(dataTypes.get(2));
    }
}
