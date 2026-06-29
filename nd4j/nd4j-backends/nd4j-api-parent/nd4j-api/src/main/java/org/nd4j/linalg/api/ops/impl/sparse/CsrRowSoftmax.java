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
 * Per-row softmax over CSR non-zero values: the GAT edge-softmax primitive.
 *
 * <p>For each row {@code i}, computes the softmax over the stored (non-zero) entries in
 * that row:
 * <pre>
 *   alpha[k] = exp(values[k]) / sum_{k' in row i} exp(values[k'])
 * </pre>
 * This is the key operation inside Graph Attention Networks (GAT, Veličković et al. 2018)
 * for normalising attention coefficients over a node's neighbourhood without materialising
 * the dense N×N attention matrix.
 *
 * <p>C++ op name: {@code csr_row_softmax}
 * <ul>
 *   <li>Inputs:  values[nnz] (floating), rowPtr[rows+1] (INT32)</li>
 *   <li>IArgs:   rows</li>
 *   <li>Output:  alpha[nnz] — per-row-softmax-normalised attention weights</li>
 * </ul>
 *
 * <p>Only {@code values} is differentiable; {@code rowPtr} is structural and receives a
 * zero gradient.  The backward pass is handled by {@link CsrRowSoftmaxBp}.
 */
public class CsrRowSoftmax extends DynamicCustomOp {

    /** Number of rows in the sparse matrix. Stored so doDiff can use it. */
    private long rows;

    /** No-arg constructor required for op-registry reflection. */
    public CsrRowSoftmax() {}

    /**
     * Eager (INDArray) constructor.
     *
     * @param values  1D [nnz] non-zero attention logits (floating dtype)
     * @param rowPtr  1D [rows+1] INT32 row pointers
     * @param rows    number of rows (source nodes / graph edges groups)
     */
    public CsrRowSoftmax(INDArray values, INDArray rowPtr, long rows) {
        super(new INDArray[]{values, rowPtr}, null);
        this.rows = rows;
        addIArgument(rows);
    }

    /**
     * SameDiff (symbolic) constructor.
     *
     * @param sd      the SameDiff graph
     * @param values  SDVariable [nnz] attention logits
     * @param rowPtr  SDVariable [rows+1] INT32 row pointers
     * @param rows    number of rows
     */
    public CsrRowSoftmax(SameDiff sd, SDVariable values, SDVariable rowPtr, long rows) {
        super(sd, new SDVariable[]{values, rowPtr});
        this.rows = rows;
        addIArgument(rows);
    }

    @Override
    public String opName() { return "csr_row_softmax"; }

    @Override
    public List<DataType> calculateOutputDataTypes(List<DataType> dataTypes) {
        // Output alpha has the same floating dtype as values (input[0])
        return Collections.singletonList(dataTypes.get(0));
    }

    /**
     * Backward pass for csr_row_softmax — delegates to {@link CsrRowSoftmaxBp}.
     *
     * <p>The backward op receives the forward output {@code alpha} (the softmax result),
     * the structural {@code rowPtr}, and the upstream gradient {@code g}:
     * <pre>
     *   dValues[k] = alpha[k] * (g[k] - sum_{k' in row i} alpha[k'] * g[k'])
     * </pre>
     *
     * @param grads upstream gradients; {@code grads.get(0)} is d(loss)/d(alpha[nnz])
     * @return gradients for [values, rowPtr]
     */
    @Override
    public List<SDVariable> doDiff(List<SDVariable> grads) {
        SDVariable g = grads.get(0);

        long r = iArguments.size() > 0 ? iArguments.get(0) : this.rows;

        SDVariable rowPtrArg = arg(1);

        // alpha = this op's output — needed by the backward Jacobian-vector product
        SDVariable alpha = outputVariable();

        SDVariable dValues = new CsrRowSoftmaxBp(sameDiff, alpha, rowPtrArg, g, r)
                .outputVariable();

        return Arrays.asList(dValues, sameDiff.zerosLike(rowPtrArg));
    }
}
