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
import java.util.Collections;
import java.util.List;

/**
 * Fused QLoRA matmul: runtime-quantized base weight plus LoRA residual.
 *
 * <p>Computes:
 * <pre>
 *   out = ggml_qmatmul(A, W) + scaling * ((A @ loraAᵀ) @ loraBᵀ)
 * </pre>
 * where W is the packed/quantised weight, loraA ∈ [rank, K], loraB ∈ [N, rank].
 *
 * <p>Supports rank-2 activations [M,K] and rank-3 activations [B,S,K]; the LoRA
 * delta is computed via the same reshape-to-2D path used by the graph-level residual
 * in {@link org.nd4j.autodiff.samediff.peft.PeftModel}.
 *
 * <b>Inputs (positional):</b>
 * <ol>
 *   <li>activations   — FLOAT32 or FLOAT16, [M,K] or [B,S,K]</li>
 *   <li>packedWeights — INT8 rank-1 GGML byte buffer (frozen CONSTANT)</li>
 *   <li>loraA         — [rank, K] trainable down-projection</li>
 *   <li>loraB         — [N, rank]  trainable up-projection</li>
 * </ol>
 *
 * <b>Float args:</b>
 * <ol>
 *   <li>scaling  — LoRA scaling factor (α/r)</li>
 * </ol>
 *
 * <b>Integer args (same convention as GgmlQMatMul):</b>
 * <ol>
 *   <li>quantType   (4=Q8_0, 8=Q4_K, 10=Q6_K)</li>
 *   <li>N</li>
 *   <li>K</li>
 *   <li>outputDtype (0=FP32, 1=FP16)</li>
 * </ol>
 *
 * <b>Output:</b> [M,N] or [B,S,N]
 */
public class GgmlQMatMulLora extends DynamicCustomOp {

    private double scaling = 1.0;

    public GgmlQMatMulLora() {
        // no-arg for serialisation
    }

    /**
     * SameDiff constructor.
     *
     * @param sd            SameDiff instance
     * @param activations   [M,K] or [B,S,K]
     * @param packedWeights INT8 rank-1 byte buffer (CONSTANT / frozen)
     * @param loraA         [rank, K] trainable
     * @param loraB         [N, rank] trainable
     * @param scaling       LoRA scaling (α/r)
     * @param quantType     GgmlQMatMul type constant (4, 8, or 10)
     * @param N             output columns
     * @param K             inner dimension
     * @param outputDtype   0=FP32, 1=FP16
     */
    public GgmlQMatMulLora(SameDiff sd, SDVariable activations, SDVariable packedWeights,
                           SDVariable loraA, SDVariable loraB,
                           double scaling, int quantType, long N, long K, int outputDtype) {
        super(null, sd, new SDVariable[]{activations, packedWeights, loraA, loraB});
        this.scaling = scaling;
        addTArgument(scaling);
        addIArgument(quantType, N, K, outputDtype);
    }

    /**
     * Eager (INDArray) constructor for the generated fluent ND API.
     */
    public GgmlQMatMulLora(INDArray activations, INDArray packedWeights,
                           INDArray loraA, INDArray loraB,
                           double scaling, int quantType, long N, long K, int outputDtype) {
        super(null, new INDArray[]{activations, packedWeights, loraA, loraB}, null);
        this.scaling = scaling;
        addTArgument(scaling);
        addIArgument(quantType, N, K, outputDtype);
    }

    /**
     * Eager convenience method.
     */
    public static INDArray exec(INDArray activations, INDArray packedWeights,
                                INDArray loraA, INDArray loraB,
                                double scaling, int quantType, long N, long K, int outputDtype) {
        Preconditions.checkArgument(
            quantType == GgmlQMatMul.GGML_QUANT_Q8_0
                || quantType == GgmlQMatMul.GGML_QUANT_Q4_K
                || quantType == GgmlQMatMul.GGML_QUANT_Q6_K,
            "GgmlQMatMulLora: unsupported quantType %s", quantType);

        GgmlQMatMulLora op = new GgmlQMatMulLora();
        op.addInputArgument(activations, packedWeights, loraA, loraB);
        op.addTArgument(scaling);
        op.addIArgument(quantType, N, K, outputDtype);
        return Nd4j.exec(op)[0];
    }

    @Override
    public String opName() {
        return "ggml_qmatmul_lora";
    }

    @Override
    public int getNumOutputs() {
        return 1;
    }

    /**
     * Backprop: delegate to {@link GgmlQMatMulLoraBp}.
     *
     * <p>Returns gradients for [activations, packedWeights, loraA, loraB].
     * packedWeights is frozen — its slot always gets {@code null} (treated as
     * zerosLike by the SameDiff gradient machinery).
     */
    @Override
    public List<SDVariable> doDiff(List<SDVariable> gradients) {
        SDVariable gradOut = gradients.get(0);

        SDVariable activations   = arg(0);
        SDVariable packedWeights = arg(1);
        SDVariable loraA         = arg(2);
        SDVariable loraB         = arg(3);

        // Recover iArgs (quantType, N, K, outputDtype) via public accessor
        long[]   iArgsArr = iArgs();
        double[] tArgsArr = tArgs();
        int quantType   = (iArgsArr != null && iArgsArr.length > 0) ? (int) iArgsArr[0] : 0;
        long N          = (iArgsArr != null && iArgsArr.length > 1) ? iArgsArr[1] : 0L;
        long K          = (iArgsArr != null && iArgsArr.length > 2) ? iArgsArr[2] : 0L;
        double sc       = (tArgsArr != null && tArgsArr.length > 0) ? tArgsArr[0] : scaling;

        // Gradient w.r.t. activations, loraA, loraB via the fused bp op
        GgmlQMatMulLoraBp bpOp = new GgmlQMatMulLoraBp(
            sameDiff, activations, packedWeights, loraA, loraB, gradOut,
            sc, quantType, N, K);
        SDVariable[] bpOuts = bpOp.outputVariables();

        SDVariable dActivations = bpOuts[0];  // [M,K] or [B,S,K]
        SDVariable dLoraA       = bpOuts[1];  // [rank,K]
        SDVariable dLoraB       = bpOuts[2];  // [N,rank]

        // packedWeights is CONSTANT — gradient is a zeros placeholder
        SDVariable dPacked = sameDiff.zerosLike(packedWeights);

        return Arrays.asList(dActivations, dPacked, dLoraA, dLoraB);
    }

    @Override
    public List<DataType> calculateOutputDataTypes(List<DataType> inputDataTypes) {
        // outputDtype: INT_ARG(3)
        long[] iArgsArr = iArgs();
        if (iArgsArr != null && iArgsArr.length >= 4) {
            int outputDtype = (int) iArgsArr[3];
            return Collections.singletonList(
                outputDtype == GgmlQMatMul.OUTPUT_FLOAT16 ? DataType.HALF : DataType.FLOAT);
        }
        return Collections.singletonList(DataType.FLOAT);
    }
}
