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
 * Backward pass for {@link CsrSddmmSparse}: computes gradients w.r.t. L values and M values.
 *
 * <p>C++ op name: {@code csr_sddmm_sparse_bp}
 *
 * <p><b>Inputs (9 arrays):</b>
 * <ol>
 *   <li>{@code targetRowPtr} – 1D [P+1], INT — row pointers of the target sparsity pattern</li>
 *   <li>{@code targetColIdx} – 1D [tnnz], INT — column indices of the target pattern</li>
 *   <li>{@code LcolIdx}      – 1D [Lnnz], INT — column indices of L</li>
 *   <li>{@code LrowPtr}      – 1D [P+1], INT — row pointers of L</li>
 *   <li>{@code McolIdx}      – 1D [Mnnz], INT — column indices of M</li>
 *   <li>{@code MrowPtr}      – 1D [Q+1], INT — row pointers of M</li>
 *   <li>{@code Lvalues}      – 1D [Lnnz], float — non-zero values of L [P, R]</li>
 *   <li>{@code Mvalues}      – 1D [Mnnz], float — non-zero values of M [Q, R]</li>
 *   <li>{@code gradOut}      – 1D [tnnz], float — upstream gradient w.r.t. outValues</li>
 * </ol>
 *
 * <p><b>Integer arguments (IArgs):</b> {@code P} (rows of L), {@code Q} (rows of M),
 * {@code R} (shared inner dimension)
 *
 * <p><b>Outputs (2 arrays):</b>
 * <ol>
 *   <li>{@code dLvalues} – 1D [Lnnz], same dtype as {@code Lvalues} (input index 6)</li>
 *   <li>{@code dMvalues} – 1D [Mnnz], same dtype as {@code Mvalues} (input index 7)</li>
 * </ol>
 *
 * <p>This op is a backward primitive: {@code doDiff} is not implemented.
 */
public class CsrSddmmSparseBp extends DynamicCustomOp {

    /** No-arg constructor required for ImportClassMapping reflection. */
    public CsrSddmmSparseBp() {}

    /**
     * Eager (INDArray) constructor.
     *
     * @param targetRowPtr 1D [P+1] INT row pointers of the target sparsity pattern
     * @param targetColIdx 1D [tnnz] INT column indices of the target pattern
     * @param LcolIdx      1D [Lnnz] INT column indices of L
     * @param LrowPtr      1D [P+1] INT row pointers of L
     * @param McolIdx      1D [Mnnz] INT column indices of M
     * @param MrowPtr      1D [Q+1] INT row pointers of M
     * @param Lvalues      1D [Lnnz] float non-zero values of L [P, R]
     * @param Mvalues      1D [Mnnz] float non-zero values of M [Q, R]
     * @param gradOut      1D [tnnz] float upstream gradient w.r.t. outValues
     * @param P            number of rows in L
     * @param Q            number of rows in M
     * @param R            shared inner dimension
     */
    public CsrSddmmSparseBp(INDArray targetRowPtr, INDArray targetColIdx,
                             INDArray LcolIdx, INDArray LrowPtr,
                             INDArray McolIdx, INDArray MrowPtr,
                             INDArray Lvalues, INDArray Mvalues,
                             INDArray gradOut,
                             long P, long Q, long R) {
        super(new INDArray[]{targetRowPtr, targetColIdx,
                             LcolIdx, LrowPtr,
                             McolIdx, MrowPtr,
                             Lvalues, Mvalues,
                             gradOut}, null);
        addIArgument(P, Q, R);
    }

    /**
     * SameDiff (symbolic) constructor.
     *
     * @param sd           the SameDiff graph
     * @param targetRowPtr SDVariable [P+1] INT row pointers of the target sparsity pattern
     * @param targetColIdx SDVariable [tnnz] INT column indices of the target pattern
     * @param LcolIdx      SDVariable [Lnnz] INT column indices of L
     * @param LrowPtr      SDVariable [P+1] INT row pointers of L
     * @param McolIdx      SDVariable [Mnnz] INT column indices of M
     * @param MrowPtr      SDVariable [Q+1] INT row pointers of M
     * @param Lvalues      SDVariable [Lnnz] float non-zero values of L [P, R]
     * @param Mvalues      SDVariable [Mnnz] float non-zero values of M [Q, R]
     * @param gradOut      SDVariable [tnnz] float upstream gradient w.r.t. outValues
     * @param P            number of rows in L
     * @param Q            number of rows in M
     * @param R            shared inner dimension
     */
    public CsrSddmmSparseBp(SameDiff sd,
                             SDVariable targetRowPtr, SDVariable targetColIdx,
                             SDVariable LcolIdx, SDVariable LrowPtr,
                             SDVariable McolIdx, SDVariable MrowPtr,
                             SDVariable Lvalues, SDVariable Mvalues,
                             SDVariable gradOut,
                             long P, long Q, long R) {
        super(sd, new SDVariable[]{targetRowPtr, targetColIdx,
                                   LcolIdx, LrowPtr,
                                   McolIdx, MrowPtr,
                                   Lvalues, Mvalues,
                                   gradOut});
        addIArgument(P, Q, R);
    }

    @Override
    public String opName() {
        return "csr_sddmm_sparse_bp";
    }

    @Override
    public int getNumOutputs() {
        return 2;
    }

    /**
     * Output data types:
     * <ol>
     *   <li>dLvalues – same dtype as {@code Lvalues} (input index 6)</li>
     *   <li>dMvalues – same dtype as {@code Mvalues} (input index 7)</li>
     * </ol>
     */
    @Override
    public List<DataType> calculateOutputDataTypes(List<DataType> dataTypes) {
        return Arrays.asList(dataTypes.get(6), dataTypes.get(7));
    }
}
