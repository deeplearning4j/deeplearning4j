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
 * Converts a dense matrix to COO (Coordinate) sparse representation.
 *
 * <p>C++ op name: {@code dense_to_coo}
 *
 * <p>Input:
 * <ol>
 *   <li>dense – 2D matrix [rows, cols]</li>
 * </ol>
 * Float args: threshold (default 0.0 — keep entries where |x| > threshold)
 *
 * <p>Outputs:
 * <ol>
 *   <li>indices – 2D [nnz, 2], INT64 — row/col index pairs for each non-zero</li>
 *   <li>values  – 1D [nnz], same dtype as input — the non-zero values</li>
 * </ol>
 */
public class DenseToCoo extends DynamicCustomOp {

    // Stored for doDiff when shape inference has not yet resolved arg(0).getShape().
    private long rows = -1;
    private long cols = -1;

    /** No-arg constructor required for ImportClassMapping reflection. */
    public DenseToCoo() {}

    /**
     * Construct from a dense matrix with an explicit sparsity threshold.
     *
     * @param dense     2D input matrix [rows, cols]
     * @param threshold keep entries where |x| > threshold (pass 0.0 to keep all non-zeros)
     */
    public DenseToCoo(INDArray dense, double threshold) {
        super(new INDArray[]{dense}, null);
        addTArgument(threshold);
        this.rows = dense.rows();
        this.cols = dense.columns();
    }

    /**
     * Convenience constructor with threshold = 0.0 (keep all structurally non-zero entries).
     *
     * @param dense 2D input matrix [rows, cols]
     */
    public DenseToCoo(INDArray dense) {
        this(dense, 0.0);
    }

    /**
     * SameDiff constructor.
     *
     * @param sd        SameDiff instance
     * @param dense     SD variable for the dense matrix
     * @param threshold keep threshold
     */
    public DenseToCoo(SameDiff sd, SDVariable dense, double threshold) {
        super(sd, new SDVariable[]{dense});
        addTArgument(threshold);
        // rows/cols resolved lazily from arg(0).getShape() inside doDiff.
    }

    @Override
    public String opName() {
        return "dense_to_coo";
    }

    @Override
    public List<DataType> calculateOutputDataTypes(List<DataType> dataTypes) {
        // Output 0 (indices) is INT64; output 1 (values) matches the dense input dtype
        return Arrays.asList(DataType.INT64, dataTypes.get(0));
    }

    /**
     * Gradient w.r.t. the dense input.
     *
     * <p>Forward outputs: [0] indices [nnz, 2], [1] values [nnz]
     * <br>
     * {@code grads.get(0)} = upstream gradient for {@code indices} (structural INT — ignored).
     * <br>
     * {@code grads.get(1)} = upstream gradient for {@code values} [nnz].
     *
     * <p>Returns: {@code [dDense]} — gradient w.r.t. the single dense input.
     */
    @Override
    public List<SDVariable> doDiff(List<SDVariable> grads) {
        SDVariable gradValues = grads.get(1);         // [nnz] float
        SDVariable indices    = outputVariables()[0]; // [nnz, 2] INT64

        // Resolve rows/cols: prefer live shape from the input variable,
        // fall back to the stored fields (set in the INDArray constructor).
        long[] shape = arg(0).getShape();
        long r = (shape != null && shape.length >= 2) ? shape[0] : rows;
        long c = (shape != null && shape.length >= 2) ? shape[1] : cols;

        SDVariable dDense = new DenseToCooBp(sameDiff, indices, gradValues, r, c)
                .outputVariables()[0];

        // One gradient for each forward input (dense only).
        return Collections.singletonList(dDense);
    }
}
