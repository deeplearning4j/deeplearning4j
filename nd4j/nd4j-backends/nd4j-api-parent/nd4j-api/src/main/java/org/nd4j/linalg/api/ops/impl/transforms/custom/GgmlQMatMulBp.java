/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  *  See the NOTICE file distributed with this work for additional
 *  *  information regarding copyright ownership.
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

package org.nd4j.linalg.api.ops.impl.transforms.custom;

import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.common.base.Preconditions;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.DynamicCustomOp;

import java.util.Collections;
import java.util.List;

/**
 * Backprop op for {@link GgmlQMatMul}.
 *
 * <p>Computes the gradient of the loss with respect to the activations A.
 * The packed weight W is frozen (CONSTANT), so its gradient is never materialised.
 *
 * <p>Semantics (matching the C++ kernel):
 * <pre>
 *   gradActivations = gradOut @ dequant(W)    // [M,K]  or  [B,S,K]
 * </pre>
 * where {@code gradOut} is the upstream gradient [M,N] / [B,S,N] and
 * {@code dequant(W)} is the logical [N,K] weight materialised on-the-fly.
 *
 * <b>Inputs (positional):</b>
 * <ol>
 *   <li>activations   — forward-pass activations [M,K] or [B,S,K]</li>
 *   <li>packedWeights — INT8 packed GGML byte buffer (rank 1)</li>
 *   <li>gradOut       — upstream gradient, same shape as forward output [M,N] / [B,S,N]</li>
 * </ol>
 *
 * <b>Integer args (same convention as GgmlQMatMul):</b>
 * <ol>
 *   <li>quantType  (4=Q8_0, 8=Q4_K, 10=Q6_K)</li>
 *   <li>N          — number of weight rows (output columns)</li>
 *   <li>K          — inner dimension</li>
 * </ol>
 *
 * <b>Output:</b> gradActivations [M,K] or [B,S,K] (same shape as activations, FLOAT dtype).
 */
public class GgmlQMatMulBp extends DynamicCustomOp {

    public GgmlQMatMulBp() {
        // no-arg for serialisation
    }

    /**
     * SameDiff constructor.
     *
     * @param sd            SameDiff instance
     * @param activations   forward activations [M,K] or [B,S,K]
     * @param packedWeights INT8 packed GGML byte buffer
     * @param gradOut       upstream gradient [M,N] or [B,S,N]
     * @param quantType     GgmlQMatMul quantisation type constant (4, 8, or 10)
     * @param N             number of weight rows
     * @param K             inner dimension
     */
    public GgmlQMatMulBp(SameDiff sd, SDVariable activations, SDVariable packedWeights,
                         SDVariable gradOut, int quantType, long N, long K) {
        super(null, sd, new SDVariable[]{activations, packedWeights, gradOut});
        addIArgument(quantType, N, K);
    }

    /**
     * Eager (INDArray) convenience method.
     */
    public static INDArray exec(INDArray activations, INDArray packedWeights, INDArray gradOut,
                                int quantType, long N, long K) {
        Preconditions.checkArgument(
            quantType == GgmlQMatMul.GGML_QUANT_Q8_0
                || quantType == GgmlQMatMul.GGML_QUANT_Q4_K
                || quantType == GgmlQMatMul.GGML_QUANT_Q6_K,
            "GgmlQMatMulBp: unsupported quantType %s", quantType);

        GgmlQMatMulBp op = new GgmlQMatMulBp();
        op.addInputArgument(activations, packedWeights, gradOut);
        op.addIArgument(quantType, N, K);
        return org.nd4j.linalg.factory.Nd4j.exec(op)[0];
    }

    @Override
    public String opName() {
        return "ggml_qmatmul_bp";
    }

    @Override
    public int getNumOutputs() {
        return 1;  // gradActivations only; packed weight is frozen
    }

    @Override
    public List<DataType> calculateOutputDataTypes(List<DataType> inputDataTypes) {
        Preconditions.checkState(inputDataTypes != null && inputDataTypes.size() == 3,
            "Expected 3 input data types, got %s", inputDataTypes);
        // Backprop through the quantized base accumulates in FP32. Returning FLOAT avoids
        // overflowing low-precision activation gradients before lower layers consume them.
        return Collections.singletonList(DataType.FLOAT);
    }
}
