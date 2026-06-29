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
 * Sparse matrix-matrix product (SpGEMM) in CSR format: C = A · B.
 *
 * <p>C++ op name: {@code csr_spgemm}
 *
 * <p><b>Inputs (6 arrays):</b>
 * <ol>
 *   <li>{@code aValues}  – 1D [nnzA], floating dtype — non-zero values of A</li>
 *   <li>{@code aColIdx}  – 1D [nnzA], INT32 — column indices of A</li>
 *   <li>{@code aRowPtr}  – 1D [m+1],  INT32 — row pointers of A</li>
 *   <li>{@code bValues}  – 1D [nnzB], floating dtype — non-zero values of B</li>
 *   <li>{@code bColIdx}  – 1D [nnzB], INT32 — column indices of B</li>
 *   <li>{@code bRowPtr}  – 1D [k+1],  INT32 — row pointers of B</li>
 * </ol>
 *
 * <p><b>Integer arguments (IArgs):</b> {@code m} (rows of A), {@code k} (cols of A = rows of B),
 * {@code n} (cols of B).
 *
 * <p><b>Outputs (3 arrays, nnz = data-dependent, computed by native DECLARE_SHAPE_FN):</b>
 * <ol>
 *   <li>{@code cValues}  – 1D [cnnz], same floating dtype as {@code aValues}</li>
 *   <li>{@code cColIdx}  – 1D [cnnz], INT32 — column indices of C</li>
 *   <li>{@code cRowPtr}  – 1D [m+1],  INT32 — row pointers of C</li>
 * </ol>
 *
 * <p><b>Automatic differentiation:</b> {@link #doDiff} is implemented via two
 * {@link CsrSddmmSparse} sampling calls (one for dA, one for dB), using {@link CsrToCsc}
 * to build the sparse transposes needed for the dB computation.
 */
public class CsrSpgemm extends DynamicCustomOp {

    /**
     * Stored rows of A — kept for no-arg-ctor safety; the canonical source in doDiff is
     * {@code iArguments.get(0/1/2)}.
     */
    private long m;
    private long k;
    private long n;

    /** No-arg constructor required for op-registry reflection. */
    public CsrSpgemm() {}

    /**
     * Eager (INDArray) constructor.
     *
     * @param aValues  1D [nnzA] non-zero values of A (floating dtype)
     * @param aColIdx  1D [nnzA] INT32 column indices of A
     * @param aRowPtr  1D [m+1]  INT32 row pointers of A
     * @param bValues  1D [nnzB] non-zero values of B (same dtype as {@code aValues})
     * @param bColIdx  1D [nnzB] INT32 column indices of B
     * @param bRowPtr  1D [k+1]  INT32 row pointers of B
     * @param m        number of rows of A (= rows of C)
     * @param k        number of columns of A (= rows of B)
     * @param n        number of columns of B (= columns of C)
     */
    public CsrSpgemm(INDArray aValues, INDArray aColIdx, INDArray aRowPtr,
                     INDArray bValues, INDArray bColIdx, INDArray bRowPtr,
                     long m, long k, long n) {
        super(new INDArray[]{aValues, aColIdx, aRowPtr, bValues, bColIdx, bRowPtr}, null);
        this.m = m;
        this.k = k;
        this.n = n;
        addIArgument(m, k, n);
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
     * @param bRowPtr  SDVariable [k+1]  INT32 row pointers of B
     * @param m        number of rows of A
     * @param k        number of columns of A (= rows of B)
     * @param n        number of columns of B
     */
    public CsrSpgemm(SameDiff sd,
                     SDVariable aValues, SDVariable aColIdx, SDVariable aRowPtr,
                     SDVariable bValues, SDVariable bColIdx, SDVariable bRowPtr,
                     long m, long k, long n) {
        super(sd, new SDVariable[]{aValues, aColIdx, aRowPtr, bValues, bColIdx, bRowPtr});
        this.m = m;
        this.k = k;
        this.n = n;
        addIArgument(m, k, n);
    }

    @Override
    public String opName() {
        return "csr_spgemm";
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
     * Backward pass for {@code C = A · B} (SpGEMM).
     *
     * <p>Delegates entirely to the native op {@link CsrSpgemmBp} ({@code csr_spgemm_bp}),
     * which performs the following device-side:
     * <pre>
     *   dA[i,j] = dot(dC row i, B row j)          sampled at A's sparsity pattern
     *   dB[j,l] = dot(Aᵀ row j, dCᵀ row l)        sampled at B's sparsity pattern
     * </pre>
     * where {@code Aᵀ} and {@code dCᵀ} are built internally via {@code csr_to_csc}.
     *
     * <p>colIdx and rowPtr arrays for A and B are INT32/structural — zero gradient.
     *
     * @param grads one-element list; {@code grads.get(0)} is the upstream gradient for cValues
     * @return gradients for [aValues, aColIdx, aRowPtr, bValues, bColIdx, bRowPtr]
     */
    @Override
    public List<SDVariable> doDiff(List<SDVariable> grads) {
        SDVariable dCValues = grads.get(0);

        // Recover m, k, n from iArguments (always populated by the constructors)
        long mVal = iArguments.size() > 0 ? iArguments.get(0) : this.m;
        long kVal = iArguments.size() > 1 ? iArguments.get(1) : this.k;
        long nVal = iArguments.size() > 2 ? iArguments.get(2) : this.n;

        SDVariable aValues = arg(0);
        SDVariable aColIdx = arg(1);
        SDVariable aRowPtr = arg(2);
        SDVariable bValues = arg(3);
        SDVariable bColIdx = arg(4);
        SDVariable bRowPtr = arg(5);

        // C's sparsity structure is stored as forward output variables 1 (cColIdx) and 2 (cRowPtr).
        SDVariable[] fwdOuts = outputVariables();
        SDVariable cColIdx  = fwdOuts[1];
        SDVariable cRowPtr  = fwdOuts[2];

        // Single native bp op handles both dA and dB (csr_to_csc + csr_sddmm_sparse device-side).
        SDVariable[] bpOuts = new CsrSpgemmBp(sameDiff,
                aValues, aColIdx, aRowPtr,
                bValues, bColIdx, bRowPtr,
                cColIdx, cRowPtr, dCValues,
                mVal, kVal, nVal).outputVariables();

        SDVariable dAValues = bpOuts[0];
        SDVariable dBValues = bpOuts[1];

        // colIdx and rowPtr arrays are INT32/structural — zero gradient
        return Arrays.asList(
                dAValues,
                sameDiff.zerosLike(aColIdx),
                sameDiff.zerosLike(aRowPtr),
                dBValues,
                sameDiff.zerosLike(bColIdx),
                sameDiff.zerosLike(bRowPtr));
    }
}
