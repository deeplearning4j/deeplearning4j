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
 * Builds an {@code n × n} diagonal CSR sparse matrix from a 1D diagonal vector.
 *
 * <p>C++ op name: {@code spdiags}
 *
 * <p><b>Input (1 array):</b>
 * <ol>
 *   <li>{@code diag} – 1D [n], floating dtype — the diagonal values</li>
 * </ol>
 *
 * <p><b>Integer argument (IArg):</b> {@code n} — size of the resulting square matrix.
 *
 * <p><b>Outputs (3 arrays, CSR of the n×n diagonal matrix):</b>
 * <ol>
 *   <li>{@code values}  – 1D [n], same dtype as {@code diag} — the non-zero values
 *       (identical to {@code diag})</li>
 *   <li>{@code colIdx}  – 1D [n], INT32 — column indices {@code [0, 1, ..., n-1]}</li>
 *   <li>{@code rowPtr}  – 1D [n+1], INT32 — row pointers {@code [0, 1, ..., n]}</li>
 * </ol>
 *
 * <p><b>Autodiff:</b> fully differentiable — gradient flows directly to {@code diag}
 * from the upstream gradient of the {@code values} output, because
 * {@code values[k] == diag[k]} identically.
 */
public class Spdiags extends DynamicCustomOp {

    /** No-arg constructor required for op-registry reflection. */
    public Spdiags() {}

    /**
     * Eager (INDArray) constructor.
     *
     * @param diag 1D [n] diagonal values (floating dtype)
     * @param n    size of the square output matrix
     */
    public Spdiags(INDArray diag, long n) {
        super(new INDArray[]{diag}, null);
        addIArgument(n);
    }

    /**
     * SameDiff (symbolic) constructor.
     *
     * @param sd   the SameDiff graph
     * @param diag SDVariable [n] diagonal values
     * @param n    size of the square output matrix
     */
    public Spdiags(SameDiff sd, SDVariable diag, long n) {
        super(sd, new SDVariable[]{diag});
        addIArgument(n);
    }

    @Override
    public String opName() {
        return "spdiags";
    }

    /**
     * Output data types:
     * <ol>
     *   <li>values  – same dtype as {@code diag} (input index 0)</li>
     *   <li>colIdx  – INT32</li>
     *   <li>rowPtr  – INT32</li>
     * </ol>
     */
    @Override
    public List<DataType> calculateOutputDataTypes(List<DataType> dataTypes) {
        return Arrays.asList(dataTypes.get(0), DataType.INT32, DataType.INT32);
    }

    /**
     * Backward pass for spdiags.
     *
     * <p>Because {@code values[k] == diag[k]} exactly, the gradient of the loss with
     * respect to {@code diag} is simply the upstream gradient of the {@code values} output:
     * <pre>
     *   d(loss)/d(diag) = grads.get(0)   (upstream grad of values)
     * </pre>
     * {@code colIdx} and {@code rowPtr} are integer/structural and carry no gradient.
     *
     * @param grads upstream gradients; {@code grads.get(0)} is the gradient w.r.t. values output
     * @return list with a single element: the gradient w.r.t. {@code diag}
     */
    @Override
    public List<SDVariable> doDiff(List<SDVariable> grads) {
        // grads has one entry per output: [d_values, d_colIdx, d_rowPtr]
        // d_colIdx and d_rowPtr are always zero (structural INT32 outputs).
        // d(diag) = d_values because values IS the diag.
        return Collections.singletonList(grads.get(0));
    }
}
