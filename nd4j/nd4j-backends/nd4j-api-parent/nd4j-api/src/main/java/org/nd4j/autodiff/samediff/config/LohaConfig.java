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

import java.util.List;

/**
 * Configuration for LoHa (Low-Rank Hadamard Product Adaptation).
 * <p>
 * LoHa uses element-wise Hadamard products of two low-rank decompositions
 * instead of a single decomposition like standard LoRA. This provides
 * greater expressiveness with the same parameter count.
 * <p>
 * Mathematical formulation:
 * <pre>
 * ΔW = s × (B₁A₁ ⊙ B₂A₂)
 *
 * where:
 *   B₁, A₁ = first low-rank pair [d×r] and [r×k]
 *   B₂, A₂ = second low-rank pair [d×r] and [r×k]
 *   ⊙ = Hadamard (element-wise) product
 *   s = scaling factor
 * </pre>
 * <p>
 * Key insight: The Hadamard product of two rank-r matrices has effective
 * rank up to r². So LoHa with dimension 8 acts like LoRA with dimension 64.
 * <p>
 * Originally developed for image generation (Stable Diffusion), but also
 * effective for LLMs.
 *
 * @author Adam Gibson
 * @see LoraConfig
 * @see <a href="https://github.com/KohakuBlueleaf/LyCORIS">LyCORIS Library</a>
 */
@Data
@SuperBuilder
@NoArgsConstructor
@AllArgsConstructor
@EqualsAndHashCode(callSuper = true)
public class LohaConfig extends PeftConfig {

    /**
     * Dimension of the low-rank matrices.
     * Note: Effective rank is up to dim², so dim=8 → effective rank up to 64.
     * Recommended max: 64 (beyond which use standard LoRA)
     * Default: 8
     */
    @Builder.Default
    private int dim = 8;

    /**
     * Scaling factor for the Hadamard product.
     * Default: 1.0
     */
    @Builder.Default
    private double alpha = 1.0;

    /**
     * Dropout probability applied to the LoHa output.
     * Default: 0.0
     */
    @Builder.Default
    private double dropout = 0.0;

    /**
     * Whether to use Tucker decomposition (LoHa variant).
     * Default: false
     */
    @Builder.Default
    private boolean useTucker = false;

    /**
     * Initialization method for LoHa matrices.
     * Options: "kaiming", "gaussian", "zeros"
     * Default: "kaiming"
     */
    @Builder.Default
    private String initMethod = "kaiming";

    @Override
    public PeftType getPeftType() {
        return PeftType.LOHA;
    }

    @Override
    public long calculateTrainableParameters(long originalParamCount) {
        // LoHa has 4 matrices: B₁, A₁, B₂, A₂
        // For each target of size [d, k]:
        // Params = 2 * (d*dim + dim*k) = 2 * dim * (d + k)
        // This is 2x the parameters of LoRA for the same dim
        if (getTargetModules() == null || getTargetModules().isEmpty()) {
            return 0;
        }
        long estimatedPerModule = 4L * dim * 1024; // Rough estimate
        return getTargetModules().size() * estimatedPerModule;
    }

    /**
     * Get the effective rank of the LoHa update.
     * @return Maximum possible rank (dim²)
     */
    public int getEffectiveMaxRank() {
        return dim * dim;
    }

    @Override
    public void validate() {
        Preconditions.checkState(dim > 0, "Dimension must be positive. Got: %s", dim);
        Preconditions.checkState(dim <= 64,
            "Dimension > 64 not recommended (effective rank would be > 4096). Got: %s", dim);
        Preconditions.checkState(alpha > 0, "Alpha must be positive. Got: %s", alpha);
        Preconditions.checkState(dropout >= 0 && dropout < 1,
            "Dropout must be in [0, 1). Got: %s", dropout);
        Preconditions.checkState(getTargetModules() != null && !getTargetModules().isEmpty(),
            "Target modules must be specified");
    }

    @Override
    public String getSummary() {
        return String.format(
            "LohaConfig(dim=%d, effectiveRank≤%d, alpha=%.2f, targets=%s)",
            dim, getEffectiveMaxRank(), alpha, getTargetModules());
    }

    /**
     * Create a default LoHa configuration.
     */
    public static LohaConfig defaultConfig(List<String> targetModules) {
        return LohaConfig.builder()
            .dim(8)
            .alpha(1.0)
            .dropout(0.0)
            .targetModules(targetModules)
            .taskType(TaskType.CAUSAL_LM)
            .build();
    }
}
