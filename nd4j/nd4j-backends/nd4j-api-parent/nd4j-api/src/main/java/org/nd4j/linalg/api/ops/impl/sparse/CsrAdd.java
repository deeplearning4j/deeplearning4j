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

/**
 * Elementwise sparse matrix addition in CSR format: C = A + B.
 *
 * <p>C++ op name: {@code csr_add}
 *
 * <p><b>Inputs (6 arrays):</b>
 * <ol>
 *   <li>{@code aValues}  – 1D [nnzA], floating dtype — non-zero values of A</li>
 *   <li>{@code aColIdx}  – 1D [nnzA], INT32 — column indices of A</li>
 *   <li>{@code aRowPtr}  – 1D [m+1],  INT32 — row pointers of A</li>
 *   <li>{@code bValues}  – 1D [nnzB], floating dtype — non-zero values of B (same dtype as aValues)</li>
 *   <li>{@code bColIdx}  – 1D [nnzB], INT32 — column indices of B</li>
 *   <li>{@code bRowPtr}  – 1D [m+1],  INT32 — row pointers of B</li>
 * </ol>
 *
 * <p><b>Integer arguments (IArgs):</b> {@code m} (rows), {@code n} (cols).
 * Both A and B must have the same logical shape [m, n].
 *
 * <p><b>Outputs (3 arrays, cnnz is data-dependent, computed by native DECLARE_SHAPE_FN):</b>
 * <ol>
 *   <li>{@code cValues}  – 1D [cnnz], same floating dtype as {@code aValues}</li>
 *   <li>{@code cColIdx}  – 1D [cnnz], INT32 — column indices of C</li>
 *   <li>{@code cRowPtr}  – 1D [m+1],  INT32 — row pointers of C</li>
 * </ol>
 *
 * <p><b>Note:</b> This op is <em>forward-only</em> in v1; automatic differentiation (autodiff)
 * is not yet supported. Attempting to call {@link SameDiff#grad} through this op will throw
 * an {@link UnsupportedOperationException}.
 */
public class CsrAdd extends DynamicCustomOp {

    /** No-arg constructor required for op-registry reflection. */
    public CsrAdd() {}

    /**
     * Eager (INDArray) constructor.
     *
     * @param aValues  1D [nnzA] non-zero values of A (floating dtype)
     * @param aColIdx  1D [nnzA] INT32 column indices of A
     * @param aRowPtr  1D [m+1]  INT32 row pointers of A
     * @param bValues  1D [nnzB] non-zero values of B (same dtype as {@code aValues})
     * @param bColIdx  1D [nnzB] INT32 column indices of B
     * @param bRowPtr  1D [m+1]  INT32 row pointers of B
     * @param m        number of rows (same for A and B)
     * @param n        number of columns (same for A and B)
     */
    public CsrAdd(INDArray aValues, INDArray aColIdx, INDArray aRowPtr,
                  INDArray bValues, INDArray bColIdx, INDArray bRowPtr,
                  long m, long n) {
        super(new INDArray[]{aValues, aColIdx, aRowPtr, bValues, bColIdx, bRowPtr}, null);
        addIArgument(m, n);
    }

    /**
     * SameDiff (symbolic) constructor.
     *
     * @param sd       the SameDiff graph
     * @param aValues  SDVariable [nnzA] non-zero values of A
     * @param aColIdx  SDVariable [nnzA] INT32 column indices of A
     * @param aRowPtr  SDVariable [m+1]  INT32 row pointers of A
     * @param bValues  SDVariable [nnzB] non-zero values of B
     * @param bColIdx  SDVariable [nnzB] INT32 column indices of B
     * @param bRowPtr  SDVariable [m+1]  INT32 row pointers of B
     * @param m        number of rows (same for A and B)
     * @param n        number of columns (same for A and B)
     */
    public CsrAdd(SameDiff sd,
                  SDVariable aValues, SDVariable aColIdx, SDVariable aRowPtr,
                  SDVariable bValues, SDVariable bColIdx, SDVariable bRowPtr,
                  long m, long n) {
        super(sd, new SDVariable[]{aValues, aColIdx, aRowPtr, bValues, bColIdx, bRowPtr});
        addIArgument(m, n);
    }

    @Override
    public String opName() {
        return "csr_add";
    }

    /**
     * Output data types:
     * <ol>
     *   <li>cValues  – same floating dtype as {@code aValues} (input index 0)</li>
     *   <li>cColIdx  – INT32</li>
     *   <li>cRowPtr  – INT32</li>
     * </ol>
     */
    @Override
    public List<DataType> calculateOutputDataTypes(List<DataType> dataTypes) {
        // dataTypes.get(0) == aValues dtype (floating)
        return Arrays.asList(dataTypes.get(0), DataType.INT32, DataType.INT32);
    }

    /**
     * Autodiff is not yet implemented for csr_add.
     *
     * @throws UnsupportedOperationException always
     */
    @Override
    public List<SDVariable> doDiff(List<SDVariable> grads) {
        throw new UnsupportedOperationException(
                "Automatic differentiation is not yet supported for " + opName()
                + ". This op is forward-only in v1.");
    }
}
