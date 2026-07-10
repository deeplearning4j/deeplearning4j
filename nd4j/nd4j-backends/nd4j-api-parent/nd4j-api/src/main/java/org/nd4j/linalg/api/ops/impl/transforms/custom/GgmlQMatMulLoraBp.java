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
import org.nd4j.linalg.factory.Nd4j;

import java.util.Arrays;
import java.util.List;

/**
 * Backprop op for {@link GgmlQMatMulLora}.
 *
 * <p>Computes gradients for the three trainable quantities:
 * <pre>
 *   dActivations  = gradOut @ dequant(W) + scaling * (gradOut @ loraB) @ loraA   // [M,K] / [B,S,K]
 *   dLoraA        = scaling * (gradOut @ loraB)ᵀ @ A_flat                          // [rank,K]
 *   dLoraB        = scaling * gradOutᵀ @ (A_flat @ loraAᵀ)                         // [N,rank]
 * </pre>
 * Packed weight is CONSTANT: its gradient is never computed here.
 *
 * <b>Inputs (positional):</b>
 * <ol>
 *   <li>activations   — [M,K] or [B,S,K]</li>
 *   <li>packedWeights — INT8 rank-1 byte buffer (frozen)</li>
 *   <li>loraA         — [rank, K]</li>
 *   <li>loraB         — [N, rank]</li>
 *   <li>gradOut       — upstream gradient [M,N] or [B,S,N]</li>
 * </ol>
 *
 * <b>Float args:</b>
 * <ol>
 *   <li>scaling — LoRA scaling factor</li>
 * </ol>
 *
 * <b>Integer args:</b>
 * <ol>
 *   <li>quantType (4=Q8_0, 8=Q4_K, 10=Q6_K)</li>
 *   <li>N</li>
 *   <li>K</li>
 * </ol>
 *
 * <b>Outputs:</b> [dActivations, dLoraA, dLoraB]
 */
public class GgmlQMatMulLoraBp extends DynamicCustomOp {

    public GgmlQMatMulLoraBp() {
        // no-arg for serialisation
    }

    /**
     * SameDiff constructor.
     */
    public GgmlQMatMulLoraBp(SameDiff sd,
                              SDVariable activations, SDVariable packedWeights,
                              SDVariable loraA, SDVariable loraB, SDVariable gradOut,
                              double scaling, int quantType, long N, long K) {
        super(null, sd, new SDVariable[]{activations, packedWeights, loraA, loraB, gradOut});
        addTArgument(scaling);
        addIArgument(quantType, N, K);
    }

    /**
     * Eager convenience method.
     *
     * @return INDArray[3] = {dActivations, dLoraA, dLoraB}
     */
    public static INDArray[] exec(INDArray activations, INDArray packedWeights,
                                  INDArray loraA, INDArray loraB, INDArray gradOut,
                                  double scaling, int quantType, long N, long K) {
        Preconditions.checkArgument(
            quantType == GgmlQMatMul.GGML_QUANT_Q8_0
                || quantType == GgmlQMatMul.GGML_QUANT_Q4_K
                || quantType == GgmlQMatMul.GGML_QUANT_Q6_K,
            "GgmlQMatMulLoraBp: unsupported quantType %s", quantType);

        GgmlQMatMulLoraBp op = new GgmlQMatMulLoraBp();
        op.addInputArgument(activations, packedWeights, loraA, loraB, gradOut);
        op.addTArgument(scaling);
        op.addIArgument(quantType, N, K);
        return Nd4j.exec(op);
    }

    @Override
    public String opName() {
        return "ggml_qmatmul_lora_bp";
    }

    @Override
    public int getNumOutputs() {
        return 3;  // dActivations, dLoraA, dLoraB
    }

    @Override
    public List<DataType> calculateOutputDataTypes(List<DataType> inputDataTypes) {
        Preconditions.checkState(inputDataTypes != null && inputDataTypes.size() == 5,
            "Expected 5 input data types, got %s", inputDataTypes);
        // All gradients share the activation dtype (input 0)
        DataType dt = inputDataTypes.get(0);
        return Arrays.asList(dt, dt, dt);
    }
}
