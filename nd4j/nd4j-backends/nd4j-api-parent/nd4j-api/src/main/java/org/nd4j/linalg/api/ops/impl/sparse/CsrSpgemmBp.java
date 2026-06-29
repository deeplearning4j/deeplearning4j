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
 * Native backward pass for CSR sparse matrix–matrix product (SpGEMM): {@code C = A · B}.
 *
 * <p>Given the upstream gradient {@code dC} (represented by {@code gradCValues, cColIdx, cRowPtr})
 * and the forward-pass inputs {@code A} and {@code B}, computes:
 * <pre>
 *   dA[i,j] = dot(dC row i, B row j)         sampled at A's sparsity pattern
 *   dB[j,l] = dot(Aᵀ row j, dCᵀ row l)       sampled at B's sparsity pattern
 * </pre>
 * All arithmetic is performed device-side inside the native op {@code csr_spgemm_bp},
 * which orchestrates {@code csr_to_csc} and {@code csr_sddmm_sparse} helpers.
 *
 * <p>C++ op name: {@code csr_spgemm_bp}
 *
 * <p><b>Inputs (9 arrays):</b>
 * <ol>
 *   <li>{@code aValues}     – 1D [nnzA], floating dtype — non-zero values of A</li>
 *   <li>{@code aColIdx}     – 1D [nnzA], INT32 — column indices of A</li>
 *   <li>{@code aRowPtr}     – 1D [m+1],  INT32 — row pointers of A</li>
 *   <li>{@code bValues}     – 1D [nnzB], floating dtype — non-zero values of B</li>
 *   <li>{@code bColIdx}     – 1D [nnzB], INT32 — column indices of B</li>
 *   <li>{@code bRowPtr}     – 1D [k+1],  INT32 — row pointers of B</li>
 *   <li>{@code cColIdx}     – 1D [nnzC], INT32 — column indices of C (from forward)</li>
 *   <li>{@code cRowPtr}     – 1D [m+1],  INT32 — row pointers of C (from forward)</li>
 *   <li>{@code gradCValues} – 1D [nnzC], floating dtype — upstream gradient w.r.t. C values</li>
 * </ol>
 *
 * <p><b>Integer arguments (IArgs):</b> {@code m} (rows of A), {@code k} (cols of A = rows of B),
 * {@code n} (cols of B).
 *
 * <p><b>Outputs (2 arrays):</b>
 * <ol>
 *   <li>{@code dAValues} – 1D [nnzA], same floating dtype as {@code aValues}</li>
 *   <li>{@code dBValues} – 1D [nnzB], same floating dtype as {@code bValues}</li>
 * </ol>
 *
 * <p>This op is forward-only (no {@code doDiff}); it is the gradient leaf for
 * {@link CsrSpgemm#doDiff}.
 */
public class CsrSpgemmBp extends DynamicCustomOp {

    /** No-arg constructor required for op-registry reflection. */
    public CsrSpgemmBp() {}

    /**
     * Eager (INDArray) constructor.
     *
     * @param aValues     1D [nnzA] non-zero values of A (floating dtype)
     * @param aColIdx     1D [nnzA] INT32 column indices of A
     * @param aRowPtr     1D [m+1]  INT32 row pointers of A
     * @param bValues     1D [nnzB] non-zero values of B (same dtype as {@code aValues})
     * @param bColIdx     1D [nnzB] INT32 column indices of B
     * @param bRowPtr     1D [k+1]  INT32 row pointers of B
     * @param cColIdx     1D [nnzC] INT32 column indices of C (from forward pass)
     * @param cRowPtr     1D [m+1]  INT32 row pointers of C (from forward pass)
     * @param gradCValues 1D [nnzC] upstream gradient w.r.t. C values
     * @param m           number of rows of A (= rows of C)
     * @param k           number of columns of A (= rows of B)
     * @param n           number of columns of B (= columns of C)
     */
    public CsrSpgemmBp(INDArray aValues,  INDArray aColIdx,  INDArray aRowPtr,
                       INDArray bValues,  INDArray bColIdx,  INDArray bRowPtr,
                       INDArray cColIdx,  INDArray cRowPtr,
                       INDArray gradCValues,
                       long m, long k, long n) {
        super(new INDArray[]{aValues, aColIdx, aRowPtr,
                             bValues, bColIdx, bRowPtr,
                             cColIdx, cRowPtr, gradCValues}, null);
        addIArgument(m, k, n);
    }

    /**
     * SameDiff (symbolic) constructor.
     *
     * @param sd          the SameDiff graph
     * @param aValues     SDVariable [nnzA] non-zero values of A
     * @param aColIdx     SDVariable [nnzA] INT32 column indices of A
     * @param aRowPtr     SDVariable [m+1]  INT32 row pointers of A
     * @param bValues     SDVariable [nnzB] non-zero values of B
     * @param bColIdx     SDVariable [nnzB] INT32 column indices of B
     * @param bRowPtr     SDVariable [k+1]  INT32 row pointers of B
     * @param cColIdx     SDVariable [nnzC] INT32 column indices of C
     * @param cRowPtr     SDVariable [m+1]  INT32 row pointers of C
     * @param gradCValues SDVariable [nnzC] upstream gradient w.r.t. C values
     * @param m           number of rows of A
     * @param k           number of columns of A (= rows of B)
     * @param n           number of columns of B
     */
    public CsrSpgemmBp(SameDiff sd,
                       SDVariable aValues,  SDVariable aColIdx,  SDVariable aRowPtr,
                       SDVariable bValues,  SDVariable bColIdx,  SDVariable bRowPtr,
                       SDVariable cColIdx,  SDVariable cRowPtr,
                       SDVariable gradCValues,
                       long m, long k, long n) {
        super(sd, new SDVariable[]{aValues, aColIdx, aRowPtr,
                                   bValues, bColIdx, bRowPtr,
                                   cColIdx, cRowPtr, gradCValues});
        addIArgument(m, k, n);
    }

    @Override
    public String opName() {
        return "csr_spgemm_bp";
    }

    /**
     * Output data types:
     * <ol>
     *   <li>{@code dAValues} – same floating dtype as {@code aValues} (input index 0)</li>
     *   <li>{@code dBValues} – same floating dtype as {@code bValues} (input index 3)</li>
     * </ol>
     */
    @Override
    public List<DataType> calculateOutputDataTypes(List<DataType> dataTypes) {
        return Arrays.asList(dataTypes.get(0), dataTypes.get(3));
    }
}
