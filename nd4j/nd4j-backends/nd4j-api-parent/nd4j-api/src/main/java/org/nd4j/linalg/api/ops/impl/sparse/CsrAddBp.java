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
 * Backward pass for {@link CsrAdd}: gradient flows to A and B values via gather from gradCValues.
 *
 * <p>C++ op name: {@code csr_add_bp}
 *
 * <p><b>Inputs (7 arrays):</b>
 * <ol>
 *   <li>{@code aColIdx}    – 1D [nnzA], INT — A's column indices</li>
 *   <li>{@code aRowPtr}    – 1D [m+1], INT — A's row pointers</li>
 *   <li>{@code bColIdx}    – 1D [nnzB], INT — B's column indices</li>
 *   <li>{@code bRowPtr}    – 1D [m+1], INT — B's row pointers</li>
 *   <li>{@code cColIdx}    – 1D [nnzC], INT — C's column indices (forward output[1])</li>
 *   <li>{@code cRowPtr}    – 1D [m+1], INT — C's row pointers (forward output[2])</li>
 *   <li>{@code gradCValues}– 1D [nnzC], float — upstream gradient w.r.t. C's values</li>
 * </ol>
 *
 * <p><b>Integer arguments (IArgs):</b> {@code m} (rows), {@code n} (cols)
 *
 * <p><b>Outputs (2 arrays):</b>
 * <ol>
 *   <li>{@code dAValues} – 1D [nnzA], same float dtype as {@code gradCValues}</li>
 *   <li>{@code dBValues} – 1D [nnzB], same float dtype as {@code gradCValues}</li>
 * </ol>
 *
 * <p>This op is a backward primitive: {@code doDiff} is not implemented.
 */
public class CsrAddBp extends DynamicCustomOp {

    /** No-arg constructor required for ImportClassMapping reflection. */
    public CsrAddBp() {}

    /**
     * Eager (INDArray) constructor.
     *
     * @param aColIdx     1D [nnzA] INT — A's column indices
     * @param aRowPtr     1D [m+1] INT — A's row pointers
     * @param bColIdx     1D [nnzB] INT — B's column indices
     * @param bRowPtr     1D [m+1] INT — B's row pointers
     * @param cColIdx     1D [nnzC] INT — C's column indices (forward output[1])
     * @param cRowPtr     1D [m+1] INT — C's row pointers (forward output[2])
     * @param gradCValues 1D [nnzC] float — upstream gradient w.r.t. C's values
     * @param m           number of rows
     * @param n           number of columns
     */
    public CsrAddBp(INDArray aColIdx, INDArray aRowPtr,
                    INDArray bColIdx, INDArray bRowPtr,
                    INDArray cColIdx, INDArray cRowPtr,
                    INDArray gradCValues,
                    long m, long n) {
        super(new INDArray[]{aColIdx, aRowPtr, bColIdx, bRowPtr, cColIdx, cRowPtr, gradCValues}, null);
        addIArgument(m, n);
    }

    /**
     * SameDiff (symbolic) constructor.
     *
     * @param sd          the SameDiff graph
     * @param aColIdx     SDVariable [nnzA] INT — A's column indices
     * @param aRowPtr     SDVariable [m+1] INT — A's row pointers
     * @param bColIdx     SDVariable [nnzB] INT — B's column indices
     * @param bRowPtr     SDVariable [m+1] INT — B's row pointers
     * @param cColIdx     SDVariable [nnzC] INT — C's column indices
     * @param cRowPtr     SDVariable [m+1] INT — C's row pointers
     * @param gradCValues SDVariable [nnzC] float — upstream gradient w.r.t. C's values
     * @param m           number of rows
     * @param n           number of columns
     */
    public CsrAddBp(SameDiff sd,
                    SDVariable aColIdx, SDVariable aRowPtr,
                    SDVariable bColIdx, SDVariable bRowPtr,
                    SDVariable cColIdx, SDVariable cRowPtr,
                    SDVariable gradCValues,
                    long m, long n) {
        super(sd, new SDVariable[]{aColIdx, aRowPtr, bColIdx, bRowPtr, cColIdx, cRowPtr, gradCValues});
        addIArgument(m, n);
    }

    @Override
    public String opName() {
        return "csr_add_bp";
    }

    @Override
    public int getNumOutputs() {
        return 2;
    }

    /**
     * Output data types: both gradient arrays share the float dtype of {@code gradCValues} (input index 6).
     */
    @Override
    public List<DataType> calculateOutputDataTypes(List<DataType> dataTypes) {
        return Arrays.asList(dataTypes.get(6), dataTypes.get(6));
    }
}
