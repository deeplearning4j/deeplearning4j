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
import org.nd4j.shade.jackson.annotation.JsonIgnore;
import org.nd4j.shade.jackson.annotation.JsonIgnoreProperties;

import java.util.Arrays;
import java.util.List;

/**
 * Configuration for LoftQ (LoRA-Fine-Tuning-aware Quantization) initialization.
 * <p>
 * LoftQ initializes LoRA adapter matrices using SVD of the quantization residual,
 * providing better starting point than random/Kaiming initialization for quantized
 * models. The algorithm alternates between quantization and SVD:
 * <ol>
 *   <li>Quantize pretrained weight W to get Q</li>
 *   <li>Compute residual R = W - Q</li>
 *   <li>SVD of R: R = U @ diag(S) @ Vt</li>
 *   <li>Initialize B = U[:, :r] @ diag(sqrt(S[:r])), A = diag(sqrt(S[:r])) @ Vt[:r, :]</li>
 *   <li>Optionally iterate: re-quantize W - B@A, compute new residual, repeat</li>
 * </ol>
 * <p>
 * At training/inference time the model behaves identically to standard LoRA;
 * LoftQ only affects how A and B are initially set. The {@code getPeftType()}
 * returns {@code PeftType.LORA} so that the PEFT model infrastructure applies
 * the standard LoRA forward pass.
 *
 * <p>Example usage:</p>
 * <pre>
 * // 4-bit NF4 with rank 16
 * LoftQConfig config = LoftQConfig.default4Bit(16);
 *
 * // Custom configuration
 * LoftQConfig config = LoftQConfig.builder()
 *     .r(16)
 *     .loraAlpha(32)
 *     .numIterations(5)
 *     .quantType("nf4")
 *     .quantBits(4)
 *     .blockSize(64)
 *     .targetModules(Arrays.asList("q_proj", "v_proj"))
 *     .taskType(TaskType.CAUSAL_LM)
 *     .build();
 * </pre>
 *
 * @author Adam Gibson
 * @see LoraConfig
 * @see <a href="https://arxiv.org/abs/2310.08659">LoftQ: LoRA-Fine-Tuning-Aware Quantization for Large Language Models</a>
 */
@Data
@SuperBuilder
@NoArgsConstructor
@AllArgsConstructor
@EqualsAndHashCode(callSuper = true)
@JsonIgnoreProperties(ignoreUnknown = true)
public class LoftQConfig extends LoraConfig {

    /**
     * Number of alternating quantization+SVD iterations.
     * More iterations generally yield lower residual error.
     * Default: 1
     */
    @Builder.Default
    private int numIterations = 1;

    /**
     * Quantization type for base model weights.
     * Options: "nf4" (Normal Float 4-bit, recommended for normally distributed weights),
     * "fp4" (Float 4-bit).
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
    private int quantBits = 4;

    /**
     * Quantization block size for group quantization.
     * Each block of {@code blockSize} elements shares a single quantization scale.
     * Smaller values improve accuracy at the cost of larger overhead.
     * Default: 64
     */
    @Builder.Default
    private int blockSize = 64;

    /**
     * LoftQ always uses LoRA at runtime; only initialization differs.
     *
     * @return {@link PeftType#LORA}
     */
    @Override
    public PeftType getPeftType() {
        return PeftType.LORA;
    }

    /**
     * Signals to the LoRA layer factory that LoftQ initialization should be used.
     *
     * @return "loftq"
     */
    @Override
    public String getInitLoraWeights() {
        return "loftq";
    }

    @Override
    public void validate() {
        super.validate();
        Preconditions.checkState(numIterations >= 1,
            "numIterations must be >= 1. Got: %s", numIterations);
        Preconditions.checkState(quantBits == 4 || quantBits == 8,
            "quantBits must be 4 or 8. Got: %s", quantBits);
        Preconditions.checkState("nf4".equals(quantType) || "fp4".equals(quantType),
            "quantType must be 'nf4' or 'fp4'. Got: %s", quantType);
        Preconditions.checkState(blockSize > 0 && blockSize <= 256,
            "blockSize must be in (0, 256]. Got: %s", blockSize);
    }

    @Override
    @JsonIgnore
    public String getSummary() {
        return String.format(
            "LoftQConfig(r=%d, alpha=%d, scaling=%.4f, quantBits=%d, quantType=%s, " +
            "blockSize=%d, numIterations=%d, targets=%s)",
            getR(), getLoraAlpha(), getScaling(), quantBits, quantType,
            blockSize, numIterations, getTargetModules());
    }

    /**
     * Create a default LoftQ configuration using 4-bit NF4 quantization.
     *
     * @param rank LoRA rank
     * @return LoftQConfig ready for use
     */
    public static LoftQConfig default4Bit(int rank) {
        return LoftQConfig.builder()
            .r(rank)
            .loraAlpha(rank * 2)
            .loraDropout(0.05)
            .numIterations(1)
            .quantType("nf4")
            .quantBits(4)
            .blockSize(64)
            .targetModules(TRANSFORMER_TARGET_MODULES)
            .taskType(TaskType.CAUSAL_LM)
            .build();
    }

    /**
     * Create a default LoftQ configuration using 8-bit quantization.
     *
     * @param rank LoRA rank
     * @return LoftQConfig ready for use
     */
    public static LoftQConfig default8Bit(int rank) {
        return LoftQConfig.builder()
            .r(rank)
            .loraAlpha(rank * 2)
            .loraDropout(0.05)
            .numIterations(1)
            .quantType("fp4")
            .quantBits(8)
            .blockSize(128)
            .targetModules(TRANSFORMER_TARGET_MODULES)
            .taskType(TaskType.CAUSAL_LM)
            .build();
    }

    /**
     * Create a LoftQ configuration with multiple alternating iterations.
     * More iterations reduce the residual approximation error.
     *
     * @param rank          LoRA rank
     * @param numIterations Number of alternating optimization iterations
     * @param targetModules Target modules to apply LoRA to
     * @return LoftQConfig with multiple iterations
     */
    public static LoftQConfig withIterations(int rank, int numIterations, List<String> targetModules) {
        return LoftQConfig.builder()
            .r(rank)
            .loraAlpha(rank * 2)
            .loraDropout(0.05)
            .numIterations(numIterations)
            .quantType("nf4")
            .quantBits(4)
            .blockSize(64)
            .targetModules(targetModules)
            .taskType(TaskType.CAUSAL_LM)
            .build();
    }
}
