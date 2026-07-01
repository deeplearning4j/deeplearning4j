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
 * Sparse Sampled Dense-Dense Matrix Multiplication (SDDMM) over two CSR sparse matrices.
 *
 * <p>For each non-zero position {@code t} at row {@code p}, column {@code q} in the target
 * sparsity pattern, computes:
 * <pre>
 *   out[t] = dot(L row p,  M row q)
 *          = sum_{r=0}^{R-1}  L[p, r] * M[q, r]
 * </pre>
 * where the dot product is taken over the shared inner dimension {@code R}.
 *
 * <p>Equivalently, {@code out} is the element-wise sampling of the dense matrix product
 * {@code L · Mᵀ} at the positions given by the target pattern.
 *
 * <p>C++ op name: {@code csr_sddmm_sparse}
 *
 * <p><b>Inputs (8 arrays):</b>
 * <ol>
 *   <li>{@code targetRowPtr} – 1D [P+1], INT32 — row pointers of the target sparsity pattern</li>
 *   <li>{@code targetColIdx} – 1D [tnnz], INT32 — column indices of the target pattern</li>
 *   <li>{@code Lvalues}      – 1D [Lnnz], floating dtype — non-zero values of L [P, R] CSR</li>
 *   <li>{@code LcolIdx}      – 1D [Lnnz], INT32 — column indices of L</li>
 *   <li>{@code LrowPtr}      – 1D [P+1],  INT32 — row pointers of L</li>
 *   <li>{@code Mvalues}      – 1D [Mnnz], floating dtype — non-zero values of M [Q, R] CSR</li>
 *   <li>{@code McolIdx}      – 1D [Mnnz], INT32 — column indices of M</li>
 *   <li>{@code MrowPtr}      – 1D [Q+1],  INT32 — row pointers of M</li>
 * </ol>
 *
 * <p><b>Integer arguments (IArgs):</b> {@code P} (rows of L), {@code Q} (rows of M),
 * {@code R} (shared inner dimension = cols of L = cols of M).
 *
 * <p><b>Output (1 array):</b>
 * <ol>
 *   <li>{@code outValues} – 1D [tnnz], same floating dtype as {@code Lvalues} —
 *       sampled values of (L · Mᵀ) at the target pattern</li>
 * </ol>
 *
 * <p>This op is used as the <em>gradient kernel</em> for {@link CsrSpgemm}: it is forward-only
 * by design (no {@code doDiff} is implemented here; second-order gradients through SpGEMM
 * are not supported).
 */
public class CsrSddmmSparse extends DynamicCustomOp {

    /** No-arg constructor required for ImportClassMapping reflection. */
    public CsrSddmmSparse() {}

    /**
     * Eager (INDArray) constructor.
     *
     * @param targetRowPtr 1D [P+1] INT32 row pointers of the target sparsity pattern
     * @param targetColIdx 1D [tnnz] INT32 column indices of the target pattern
     * @param Lvalues      1D [Lnnz] non-zero values of L [P, R] (floating dtype)
     * @param LcolIdx      1D [Lnnz] INT32 column indices of L
     * @param LrowPtr      1D [P+1]  INT32 row pointers of L
     * @param Mvalues      1D [Mnnz] non-zero values of M [Q, R] (same dtype as Lvalues)
     * @param McolIdx      1D [Mnnz] INT32 column indices of M
     * @param MrowPtr      1D [Q+1]  INT32 row pointers of M
     * @param P            number of rows in L (= number of rows in the target pattern)
     * @param Q            number of rows in M (= number of columns in the target pattern)
     * @param R            shared inner dimension (cols of L = cols of M)
     */
    public CsrSddmmSparse(INDArray targetRowPtr, INDArray targetColIdx,
                          INDArray Lvalues, INDArray LcolIdx, INDArray LrowPtr,
                          INDArray Mvalues, INDArray McolIdx, INDArray MrowPtr,
                          long P, long Q, long R) {
        super(new INDArray[]{targetRowPtr, targetColIdx,
                             Lvalues, LcolIdx, LrowPtr,
                             Mvalues, McolIdx, MrowPtr}, null);
        addIArgument(P, Q, R);
    }

    /**
     * SameDiff (symbolic) constructor.
     *
     * @param sd           the SameDiff graph
     * @param targetRowPtr SDVariable [P+1] INT32 row pointers of the target pattern
     * @param targetColIdx SDVariable [tnnz] INT32 column indices of the target pattern
     * @param Lvalues      SDVariable [Lnnz] non-zero values of L [P, R]
     * @param LcolIdx      SDVariable [Lnnz] INT32 column indices of L
     * @param LrowPtr      SDVariable [P+1]  INT32 row pointers of L
     * @param Mvalues      SDVariable [Mnnz] non-zero values of M [Q, R]
     * @param McolIdx      SDVariable [Mnnz] INT32 column indices of M
     * @param MrowPtr      SDVariable [Q+1]  INT32 row pointers of M
     * @param P            number of rows in L
     * @param Q            number of rows in M
     * @param R            shared inner dimension
     */
    public CsrSddmmSparse(SameDiff sd,
                          SDVariable targetRowPtr, SDVariable targetColIdx,
                          SDVariable Lvalues, SDVariable LcolIdx, SDVariable LrowPtr,
                          SDVariable Mvalues, SDVariable McolIdx, SDVariable MrowPtr,
                          long P, long Q, long R) {
        super(sd, new SDVariable[]{targetRowPtr, targetColIdx,
                                   Lvalues, LcolIdx, LrowPtr,
                                   Mvalues, McolIdx, MrowPtr});
        addIArgument(P, Q, R);
    }

    @Override
    public String opName() {
        return "csr_sddmm_sparse";
    }

    /**
     * Output data type: {@code outValues} has the same floating dtype as {@code Lvalues}
     * (input index 2).
     */
    @Override
    public List<DataType> calculateOutputDataTypes(List<DataType> dataTypes) {
        return Collections.singletonList(dataTypes.get(2));
    }

    @Override
    public List<SDVariable> doDiff(List<SDVariable> grads) {
        SDVariable gradOut = grads.get(0);
        long P = iArguments.get(0);
        long Q = iArguments.get(1);
        long R = iArguments.get(2);
        SDVariable targetRowPtr = arg(0);
        SDVariable targetColIdx = arg(1);
        SDVariable Lvalues = arg(2);
        SDVariable LcolIdx = arg(3);
        SDVariable LrowPtr = arg(4);
        SDVariable Mvalues = arg(5);
        SDVariable McolIdx = arg(6);
        SDVariable MrowPtr = arg(7);
        SDVariable[] bpOut = new CsrSddmmSparseBp(sameDiff,
            targetRowPtr, targetColIdx, LcolIdx, LrowPtr, McolIdx, MrowPtr,
            Lvalues, Mvalues, gradOut, P, Q, R).outputVariables();
        return Arrays.asList(
            sameDiff.zerosLike(targetRowPtr), sameDiff.zerosLike(targetColIdx),
            bpOut[0],
            sameDiff.zerosLike(LcolIdx), sameDiff.zerosLike(LrowPtr),
            bpOut[1],
            sameDiff.zerosLike(McolIdx), sameDiff.zerosLike(MrowPtr));
    }
}
