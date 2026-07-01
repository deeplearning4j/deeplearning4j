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
 * Converts a dense matrix to CSC (Compressed Sparse Column) sparse representation.
 *
 * <p>C++ op name: {@code dense_to_csc}
 *
 * <p>Input:
 * <ol>
 *   <li>dense – 2D matrix [rows, cols]</li>
 * </ol>
 * Float args: threshold (default 0.0 — keep entries where |x| > threshold)
 *
 * <p>Outputs:
 * <ol>
 *   <li>cscValues  – 1D [nnz], same dtype as input — non-zero values in column-major order</li>
 *   <li>cscRowIdx  – 1D [nnz], INT32 — row index for each non-zero</li>
 *   <li>cscColPtr  – 1D [cols+1], INT32 — column pointers</li>
 * </ol>
 * The number of non-zeros {@code nnz} is data-dependent and determined by the native shape
 * function at runtime.
 */
public class DenseToCsc extends DynamicCustomOp {

    /** No-arg constructor required for ImportClassMapping reflection. */
    public DenseToCsc() {}

    /**
     * Construct from a dense matrix with an explicit threshold.
     *
     * @param dense     2D input matrix [rows, cols]
     * @param threshold keep entries where |x| > threshold (pass 0.0 to keep all non-zeros)
     */
    public DenseToCsc(INDArray dense, double threshold) {
        super(new INDArray[]{dense}, null);
        addTArgument(threshold);
    }

    /**
     * Convenience constructor with threshold = 0.0 (keep all structurally non-zero entries).
     *
     * @param dense 2D input matrix [rows, cols]
     */
    public DenseToCsc(INDArray dense) {
        this(dense, 0.0);
    }

    /**
     * SameDiff constructor.
     *
     * @param sd        SameDiff instance
     * @param dense     SD variable for the dense matrix
     * @param threshold keep threshold
     */
    public DenseToCsc(SameDiff sd, SDVariable dense, double threshold) {
        super(sd, new SDVariable[]{dense});
        addTArgument(threshold);
    }

    @Override
    public String opName() {
        return "dense_to_csc";
    }

    @Override
    public List<DataType> calculateOutputDataTypes(List<DataType> dataTypes) {
        // Output 0 (cscValues) has same dtype as the dense input (index 0)
        // Output 1 (cscRowIdx) is INT32; output 2 (cscColPtr) is INT32
        return Arrays.asList(dataTypes.get(0), DataType.INT32, DataType.INT32);
    }

    @Override
    public List<SDVariable> doDiff(List<SDVariable> grads) {
        SDVariable gradCscValues = grads.get(0);
        SDVariable dense = arg(0);
        long[] denseShape = dense.getShape();
        long rows = denseShape[0];
        long cols = denseShape[1];
        SDVariable[] fwdOuts = outputVariables();
        SDVariable cscRowIdx = fwdOuts[1];
        SDVariable cscColPtr = fwdOuts[2];
        SDVariable dDense = new DenseToCscBp(sameDiff, cscRowIdx, cscColPtr, gradCscValues, rows, cols)
                .outputVariables()[0];
        return Collections.singletonList(dDense);
    }
}
