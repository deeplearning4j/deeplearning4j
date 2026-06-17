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
 * Configuration for Low-Rank Adaptation (LoRA).
 * <p>
 * LoRA decomposes weight updates into low-rank matrices: W = W₀ + BA
 * where B ∈ R^{d×r} and A ∈ R^{r×k} with rank r << min(d, k).
 * <p>
 * This significantly reduces trainable parameters while maintaining
 * performance comparable to full fine-tuning. For example, with GPT-3 175B,
 * LoRA can reduce trainable parameters by 10,000x and GPU memory by 3x.
 *
 * <p>Mathematical formulation:</p>
 * <pre>
 * h = W₀x + ΔWx = W₀x + BAx
 *
 * where:
 *   W₀ ∈ R^{d×k} is the frozen pretrained weight
 *   B ∈ R^{d×r} is the down-projection matrix (initialized to zero)
 *   A ∈ R^{r×k} is the up-projection matrix (initialized randomly)
 *   r is the rank (r << min(d, k))
 *
 * Scaling: output = W₀x + (α/r) × BAx
 * </pre>
 *
 * <p>Example usage:</p>
 * <pre>
 * LoraConfig config = LoraConfig.builder()
 *     .r(16)                    // Rank of low-rank matrices
 *     .loraAlpha(32)            // Scaling factor
 *     .loraDropout(0.1)         // Dropout probability
 *     .targetModules(Arrays.asList("query", "key", "value", "output"))
 *     .taskType(TaskType.CAUSAL_LM)
 *     .build();
 * </pre>
 *
 * @author Adam Gibson
 * @see PeftConfig
 * @see <a href="https://arxiv.org/abs/2106.09685">LoRA Paper</a>
 */
@Data
@SuperBuilder
@NoArgsConstructor
@AllArgsConstructor
@EqualsAndHashCode(callSuper = true)
public class LoraConfig extends PeftConfig {

    /**
     * The rank of the low-rank decomposition matrices.
     * Lower rank = fewer parameters but potentially lower expressiveness.
     * Recommended starting values: 8-32.
     * Default: 8
     */
    @Builder.Default
    private int r = 8;

    /**
     * The scaling factor for LoRA updates.
     * The actual scaling applied is (loraAlpha / r).
     * Common practice: set alpha = 2 * r.
     * Default: 16
     */
    @Builder.Default
    private int loraAlpha = 16;

    /**
     * Dropout probability for LoRA layers.
     * Applied to the output of LoRA computation before adding to base output.
     * Default: 0.0 (no dropout)
     */
    @Builder.Default
    private double loraDropout = 0.0;

    /**
     * Whether to use rank-stabilized LoRA (rsLoRA).
     * Uses α/√r instead of α/r for better stability with higher ranks.
     * Default: false
     */
    @Builder.Default
    private boolean useRsLora = false;

    /**
     * Whether to apply LoRA to bias terms as well.
     * Options: "none", "all", "lora_only"
     * Default: "none"
     */
    @Builder.Default
    private String bias = "none";

    /**
     * Fan-in/fan-out configuration for weight initialization.
     * Default: false
     */
    @Builder.Default
    private boolean fanInFanOut = false;

    /**
     * Initialization method for matrix A.
     * Options: "gaussian", "kaiming_uniform", "zeros"
     * Default: "kaiming_uniform"
     */
    @Builder.Default
    private String initLoraWeights = "kaiming_uniform";

    /**
     * Standard deviation for Gaussian initialization.
     * Only used when initLoraWeights = "gaussian"
     * Default: 0.02
     */
    @Builder.Default
    private double initStd = 0.02;

    /**
     * Default target modules for common architectures.
     */
    public static final List<String> TRANSFORMER_TARGET_MODULES =
        Arrays.asList("q_proj", "k_proj", "v_proj", "o_proj");

    public static final List<String> TRANSFORMER_ALL_LINEAR =
        Arrays.asList("q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj");

    public static final List<String> ATTENTION_ONLY =
        Arrays.asList("query", "key", "value", "output");

    @Override
    public PeftType getPeftType() {
        return PeftType.LORA;
    }

    @Override
    public long calculateTrainableParameters(long originalParamCount) {
        // For each target module of size [d, k]:
        // LoRA adds: d*r + r*k = r*(d+k) parameters
        // This is a rough estimate; actual count depends on target module dimensions
        if (targetModules == null || targetModules.isEmpty()) {
            return 0;
        }
        // Estimate: assume each target contributes r * 2 * avgDim parameters
        // More accurate calculation would require knowing actual dimensions
        long estimatedPerModule = 2L * r * 1024; // Placeholder estimate
        return targetModules.size() * estimatedPerModule;
    }

    /**
     * Calculate the scaling factor for LoRA output.
     *
     * @return The scaling factor: α/r or α/√r for rsLoRA
     */
    public double getScaling() {
        if (useRsLora) {
            return loraAlpha / Math.sqrt(r);
        }
        return (double) loraAlpha / r;
    }

    @Override
    public void validate() {
        Preconditions.checkState(r > 0, "Rank (r) must be positive. Got: %s", r);
        Preconditions.checkState(loraAlpha > 0, "LoRA alpha must be positive. Got: %s", loraAlpha);
        Preconditions.checkState(loraDropout >= 0 && loraDropout < 1,
            "LoRA dropout must be in [0, 1). Got: %s", loraDropout);
        validateTargetModules();
        Preconditions.checkState(bias.equals("none") || bias.equals("all") || bias.equals("lora_only"),
            "Bias must be 'none', 'all', or 'lora_only'. Got: %s", bias);
    }

    @Override
    @JsonIgnore
    public String getSummary() {
        return String.format(
            "LoraConfig(r=%d, alpha=%d, scaling=%.4f, dropout=%.2f, targets=%s, rsLora=%s)",
            r, loraAlpha, getScaling(), loraDropout, targetModules, useRsLora);
    }

    /**
     * Create a default LoRA configuration for transformer models.
     */
    public static LoraConfig defaultTransformer() {
        return LoraConfig.builder()
            .r(16)
            .loraAlpha(32)
            .loraDropout(0.05)
            .targetModules(TRANSFORMER_TARGET_MODULES)
            .taskType(TaskType.CAUSAL_LM)
            .build();
    }

    /**
     * Create a LoRA configuration targeting all linear layers.
     */
    public static LoraConfig allLinear(int rank) {
        return LoraConfig.builder()
            .r(rank)
            .loraAlpha(rank * 2)
            .loraDropout(0.05)
            .targetModules(TRANSFORMER_ALL_LINEAR)
            .taskType(TaskType.CAUSAL_LM)
            .build();
    }

    /**
     * Create a minimal LoRA configuration for memory-constrained scenarios.
     */
    public static LoraConfig minimal() {
        return LoraConfig.builder()
            .r(4)
            .loraAlpha(8)
            .loraDropout(0.0)
            .targetModules(Arrays.asList("query", "value"))
            .taskType(TaskType.CAUSAL_LM)
            .build();
    }
}
