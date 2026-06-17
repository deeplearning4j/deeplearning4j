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

package org.nd4j.autodiff.samediff.config;

import lombok.*;
import lombok.experimental.SuperBuilder;
import org.nd4j.common.base.Preconditions;
import org.nd4j.linalg.api.buffer.DataType;

import java.util.List;

/**
 * Configuration for QLoRA (Quantized Low-Rank Adaptation).
 * <p>
 * QLoRA combines 4-bit quantization of base model weights with LoRA adapters
 * in higher precision, enabling efficient fine-tuning of very large models
 * on consumer hardware.
 * <p>
 * Key features:
 * <ul>
 *   <li>Base weights: Quantized to 4-bit (NF4 or FP4)</li>
 *   <li>LoRA weights: Full precision (FP16/BF16/FP32)</li>
 *   <li>Computation: BF16 for forward/backward pass</li>
 *   <li>Master weights: FP32 for optimizer states</li>
 * </ul>
 * <p>
 * Memory savings example:
 * <ul>
 *   <li>65B model: 130GB FP16 → ~33GB 4-bit + LoRA</li>
 *   <li>Enables fine-tuning on single 48GB GPU</li>
 * </ul>
 *
 * @author Adam Gibson
 * @see LoraConfig
 * @see <a href="https://arxiv.org/abs/2305.14314">QLoRA Paper</a>
 */
@Data
@SuperBuilder
@NoArgsConstructor
@AllArgsConstructor
@EqualsAndHashCode(callSuper = true)
public class QLoraConfig extends LoraConfig {

    /**
     * Quantization type for base model weights.
     * Options: "nf4" (Normal Float 4-bit), "fp4" (Float 4-bit)
     * NF4 is recommended as it's optimized for normally distributed weights.
     * Default: "nf4"
     */
    @Builder.Default
    private String quantType = "nf4";

    /**
     * Number of bits for quantization.
     * Options: 4, 8
     * Default: 4
     */
    @Builder.Default
    private int bits = 4;

    /**
     * Whether to use double quantization.
     * Quantizes the quantization constants for additional memory savings.
     * Default: true
     */
    @Builder.Default
    private boolean doubleQuant = true;

    /**
     * Data type for computation during forward/backward pass.
     * Recommended: BFLOAT16 (no loss scaling needed) or FLOAT16
     * Default: BFLOAT16
     */
    @Builder.Default
    private DataType computeDataType = DataType.BFLOAT16;

    /**
     * Data type for LoRA adapter weights.
     * Should match computeDataType or be FP32.
     * Default: BFLOAT16
     */
    @Builder.Default
    private DataType loraDataType = DataType.BFLOAT16;

    /**
     * Block size for quantization.
     * Smaller blocks = better accuracy but more overhead.
     * Default: 64
     */
    @Builder.Default
    private int blockSize = 64;

    /**
     * Whether to quantize the embedding layer.
     * Usually kept in FP16/BF16 for stability.
     * Default: false
     */
    @Builder.Default
    private boolean quantizeEmbeddings = false;

    /**
     * Whether to quantize the LM head (output layer).
     * Usually kept in FP16/BF16 for stability.
     * Default: false
     */
    @Builder.Default
    private boolean quantizeLmHead = false;

    /**
     * Modules to exclude from quantization.
     * Default: ["lm_head", "embed_tokens"]
     */
    private List<String> quantizeExcludeModules;

    @Override
    public PeftType getPeftType() {
        return PeftType.QLORA;
    }

    @Override
    public void validate() {
        super.validate();
        Preconditions.checkState(bits == 4 || bits == 8,
            "Quantization bits must be 4 or 8. Got: %s", bits);
        Preconditions.checkState(quantType.equals("nf4") || quantType.equals("fp4"),
            "Quantization type must be 'nf4' or 'fp4'. Got: %s", quantType);
        Preconditions.checkState(blockSize > 0 && blockSize <= 256,
            "Block size must be in (0, 256]. Got: %s", blockSize);
        Preconditions.checkState(computeDataType == DataType.FLOAT16 ||
            computeDataType == DataType.BFLOAT16 || computeDataType == DataType.FLOAT,
            "Compute data type must be FLOAT16, BFLOAT16, or FLOAT. Got: %s", computeDataType);
    }

    @Override
    public String getSummary() {
        return String.format(
            "QLoraConfig(r=%d, alpha=%d, bits=%d, type=%s, doubleQuant=%s, compute=%s, targets=%s)",
            getR(), getLoraAlpha(), bits, quantType, doubleQuant, computeDataType, getTargetModules());
    }

    /**
     * Estimate memory savings compared to full precision.
     *
     * @param totalParams Total number of model parameters
     * @return Approximate memory in bytes for quantized model
     */
    public long estimateMemoryBytes(long totalParams) {
        // Base weights: bits per weight / 8 = bytes per weight
        double bytesPerBaseWeight = bits / 8.0;

        // Quantization overhead (scales, zeros) per block
        double overheadPerBlock = 4.0; // FP32 scale + zero
        if (doubleQuant) {
            overheadPerBlock = 1.0; // Further compressed
        }
        double overheadFactor = 1.0 + (overheadPerBlock / blockSize);

        long baseMemory = (long) (totalParams * bytesPerBaseWeight * overheadFactor);

        // LoRA weights in loraDataType
        int loraBytes = loraDataType.width();
        long loraParams = calculateTrainableParameters(totalParams);
        long loraMemory = loraParams * loraBytes;

        return baseMemory + loraMemory;
    }

    /**
     * Create a default QLoRA configuration for 4-bit quantization.
     */
    public static QLoraConfig default4Bit(List<String> targetModules) {
        return QLoraConfig.builder()
            .r(16)
            .loraAlpha(32)
            .loraDropout(0.05)
            .bits(4)
            .quantType("nf4")
            .doubleQuant(true)
            .computeDataType(DataType.BFLOAT16)
            .loraDataType(DataType.BFLOAT16)
            .blockSize(64)
            .targetModules(targetModules)
            .taskType(TaskType.CAUSAL_LM)
            .build();
    }

    /**
     * Create a QLoRA configuration for 8-bit quantization.
     * Less memory efficient but potentially higher quality.
     */
    public static QLoraConfig default8Bit(List<String> targetModules) {
        return QLoraConfig.builder()
            .r(16)
            .loraAlpha(32)
            .loraDropout(0.05)
            .bits(8)
            .quantType("fp4") // FP4 is not applicable for 8-bit, will be ignored
            .doubleQuant(false)
            .computeDataType(DataType.BFLOAT16)
            .loraDataType(DataType.BFLOAT16)
            .blockSize(128)
            .targetModules(targetModules)
            .taskType(TaskType.CAUSAL_LM)
            .build();
    }
}
