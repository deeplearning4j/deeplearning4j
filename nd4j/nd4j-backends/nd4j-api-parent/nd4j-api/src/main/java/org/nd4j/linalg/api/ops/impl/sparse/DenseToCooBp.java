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
 * Native backward pass for {@link DenseToCoo}: scatters the upstream gradient
 * for the COO values back into a dense gradient matrix.
 *
 * <p>C++ op name: {@code dense_to_coo_bp}
 *
 * <p>Forward recap: {@code dense[rows,cols] → (indices[nnz,2], values[nnz])}.
 *
 * <p><b>Inputs (2 arrays):</b>
 * <ol>
 *   <li>{@code indices}    – 2D [nnz, 2], INT64 — row/col index pairs from the forward pass</li>
 *   <li>{@code gradValues} – 1D [nnz], float — upstream gradient w.r.t. the COO values output</li>
 * </ol>
 *
 * <p><b>Integer arguments (IArgs):</b> {@code rows}, {@code cols}.
 *
 * <p><b>Output (1 array):</b>
 * <ol>
 *   <li>{@code dDense} – 2D [rows, cols], same float dtype as {@code gradValues}</li>
 * </ol>
 *
 * <p>Math: {@code dDense} is zero-initialised, then for each {@code k}:
 * {@code dDense[indices[k,0], indices[k,1]] = gradValues[k]}.
 * No atomics are needed because {@link DenseToCoo} emits unique index pairs.
 *
 * <p>This op is forward-only (no {@code doDiff}); it is the gradient leaf for
 * {@link DenseToCoo#doDiff}.
 */
public class DenseToCooBp extends DynamicCustomOp {

    /** No-arg constructor required for op-registry reflection. */
    public DenseToCooBp() {}

    /**
     * Eager (INDArray) constructor.
     *
     * @param indices    2D [nnz, 2] INT64 — row/col index pairs from the forward pass
     * @param gradValues 1D [nnz] float — upstream gradient w.r.t. the COO values output
     * @param rows       number of rows in the dense matrix
     * @param cols       number of columns in the dense matrix
     */
    public DenseToCooBp(INDArray indices, INDArray gradValues, long rows, long cols) {
        super(new INDArray[]{indices, gradValues}, null);
        addIArgument(rows, cols);
    }

    /**
     * SameDiff (symbolic) constructor.
     *
     * @param sd         the SameDiff graph
     * @param indices    SDVariable [nnz, 2] INT64 — row/col index pairs from the forward pass
     * @param gradValues SDVariable [nnz] float — upstream gradient w.r.t. the COO values output
     * @param rows       number of rows in the dense matrix
     * @param cols       number of columns in the dense matrix
     */
    public DenseToCooBp(SameDiff sd, SDVariable indices, SDVariable gradValues,
                        long rows, long cols) {
        super(sd, new SDVariable[]{indices, gradValues});
        addIArgument(rows, cols);
    }

    @Override
    public String opName() {
        return "dense_to_coo_bp";
    }

    /**
     * Output data type: {@code dDense} has the same float dtype as {@code gradValues}
     * (input index 1).
     */
    @Override
    public List<DataType> calculateOutputDataTypes(List<DataType> dataTypes) {
        return Collections.singletonList(dataTypes.get(1));
    }
}
