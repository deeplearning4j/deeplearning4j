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
 * Backward pass for {@link CsrRowSoftmax}: per-row softmax gradient.
 *
 * <p>C++ op name: {@code csr_row_softmax_bp}
 * <ul>
 *   <li>Inputs:  alpha[nnz] (forward output), rowPtr[rows+1] (INT32),
 *                gradOut[nnz] (upstream gradient)</li>
 *   <li>IArgs:   rows</li>
 *   <li>Output:  dValues[nnz] — gradient w.r.t. the forward input values</li>
 * </ul>
 *
 * <p>Gradient math (softmax Jacobian-vector product, per row):
 * <pre>
 *   dValues[k] = alpha[k] * (gradOut[k] - dot(alpha[row i], gradOut[row i]))
 * </pre>
 * where {@code row i} is the row containing non-zero index {@code k}.
 *
 * <p>This is a backward primitive: {@code doDiff} is not implemented.
 */
public class CsrRowSoftmaxBp extends DynamicCustomOp {

    /** No-arg constructor required for op-registry reflection. */
    public CsrRowSoftmaxBp() {}

    /**
     * Eager (INDArray) constructor.
     *
     * @param alpha    1D [nnz] per-row-softmax output from the forward pass
     * @param rowPtr   1D [rows+1] INT32 row pointers
     * @param gradOut  1D [nnz] upstream gradient d(loss)/d(alpha)
     * @param rows     number of rows
     */
    public CsrRowSoftmaxBp(INDArray alpha, INDArray rowPtr, INDArray gradOut, long rows) {
        super(new INDArray[]{alpha, rowPtr, gradOut}, null);
        addIArgument(rows);
    }

    /**
     * SameDiff (symbolic) constructor.
     *
     * @param sd       the SameDiff graph
     * @param alpha    SDVariable [nnz] per-row-softmax output
     * @param rowPtr   SDVariable [rows+1] INT32 row pointers
     * @param gradOut  SDVariable [nnz] upstream gradient
     * @param rows     number of rows
     */
    public CsrRowSoftmaxBp(SameDiff sd, SDVariable alpha, SDVariable rowPtr,
                            SDVariable gradOut, long rows) {
        super(sd, new SDVariable[]{alpha, rowPtr, gradOut});
        addIArgument(rows);
    }

    @Override
    public String opName() { return "csr_row_softmax_bp"; }

    @Override
    public List<DataType> calculateOutputDataTypes(List<DataType> dataTypes) {
        // dValues: same dtype as alpha (input[0])
        return Collections.singletonList(dataTypes.get(0));
    }
}
