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
 * Converts a dense matrix to CSR (Compressed Sparse Row) sparse representation.
 *
 * <p>C++ op name: {@code dense_to_csr}
 *
 * <p>Input:
 * <ol>
 *   <li>dense – 2D matrix [rows, cols]</li>
 * </ol>
 * Float args: threshold (default 0.0 — keep entries where |x| > threshold)
 *
 * <p>Outputs:
 * <ol>
 *   <li>values  – 1D [nnz], same dtype as input</li>
 *   <li>colIdx  – 1D [nnz], INT32</li>
 *   <li>rowPtr  – 1D [rows+1], INT32</li>
 * </ol>
 */
public class DenseToCsr extends DynamicCustomOp {

    /** No-arg constructor required for ImportClassMapping reflection. */
    public DenseToCsr() {}

    /**
     * Construct from a dense matrix.
     *
     * @param dense     2D input matrix [rows, cols]
     * @param threshold keep entries where |x| > threshold (pass 0.0 to keep all non-zeros)
     */
    public DenseToCsr(INDArray dense, double threshold) {
        super(new INDArray[]{dense}, null);
        addTArgument(threshold);
    }

    /**
     * Convenience constructor with threshold = 0.0 (keep all structurally non-zero entries).
     *
     * @param dense 2D input matrix [rows, cols]
     */
    public DenseToCsr(INDArray dense) {
        this(dense, 0.0);
    }

    /**
     * SameDiff constructor.
     *
     * @param sd        SameDiff instance
     * @param dense     SD variable for the dense matrix
     * @param threshold keep threshold
     */
    public DenseToCsr(SameDiff sd, SDVariable dense, double threshold) {
        super(sd, new SDVariable[]{dense});
        addTArgument(threshold);
    }

    @Override
    public String opName() {
        return "dense_to_csr";
    }

    @Override
    public List<DataType> calculateOutputDataTypes(List<DataType> dataTypes) {
        // Output 0 (values) has same dtype as the dense input (index 0)
        // Outputs 1 (colIdx) and 2 (rowPtr) are INT32
        return Arrays.asList(dataTypes.get(0), DataType.INT32, DataType.INT32);
    }

    /**
     * Backward pass for {@code dense_to_csr}.
     *
     * <p>The forward op compresses {@code dense} into CSR format; the gradient
     * scatters {@code gradValues[e]} back into {@code dDense[i, colIdx[e]]}.
     * The structural outputs {@code colIdx} and {@code rowPtr} are INT32 and receive
     * no meaningful gradient.
     *
     * <p>{@code rows} and {@code cols} are recovered from the static shape of the
     * forward dense input.  If the shape is not statically known at graph-build time,
     * use eager execution.
     *
     * @param grads three-element list (one per forward output):
     *              {@code grads.get(0)} is the upstream gradient for the values output
     * @return one-element list containing the gradient w.r.t. the dense input
     */
    @Override
    public List<SDVariable> doDiff(List<SDVariable> grads) {
        SDVariable gradValues = grads.get(0);   // gradient w.r.t. values output

        SDVariable dense = arg(0);
        long[] denseShape = dense.getShape();
        long rows = denseShape[0];
        long cols = denseShape[1];

        // colIdx (output[1]) and rowPtr (output[2]) from the forward pass
        SDVariable[] fwdOuts = outputVariables();
        SDVariable colIdx = fwdOuts[1];
        SDVariable rowPtr = fwdOuts[2];

        SDVariable dDense = new DenseToCsrBp(sameDiff, colIdx, rowPtr, gradValues, rows, cols)
                .outputVariables()[0];

        // Single dense input — return one gradient
        return Collections.singletonList(dDense);
    }
}
