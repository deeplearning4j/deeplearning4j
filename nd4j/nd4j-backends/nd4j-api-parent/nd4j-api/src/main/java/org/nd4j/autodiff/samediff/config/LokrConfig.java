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
 * Configuration for LoKr (Low-Rank Kronecker Product Adaptation).
 * <p>
 * LoKr uses Kronecker products to construct weight updates, maintaining
 * matrix rank through block structures. This provides efficient parameter
 * updates by stacking matrix columns.
 * <p>
 * Mathematical formulation:
 * <pre>
 * ΔW = s × (C ⊗ (BA))
 *
 * where:
 *   C = Kronecker factor matrix
 *   B, A = low-rank matrices
 *   ⊗ = Kronecker product
 *   s = scaling factor
 * </pre>
 * <p>
 * The Kronecker factor setting (-1 to 8) controls the trade-off:
 * <ul>
 *   <li>Smaller factor → smaller file size</li>
 *   <li>Larger factor → more similar to standard LoRA behavior</li>
 * </ul>
 * <p>
 * Originally developed for image generation, also effective for LLMs.
 *
 * @author Adam Gibson
 * @see LoraConfig
 * @see LohaConfig
 * @see <a href="https://github.com/KohakuBlueleaf/LyCORIS">LyCORIS Library</a>
 */
@Data
@SuperBuilder
@NoArgsConstructor
@AllArgsConstructor
@EqualsAndHashCode(callSuper = true)
public class LokrConfig extends PeftConfig {

    /**
     * Dimension of the low-rank matrices (A, B).
     * Default: 8
     */
    @Builder.Default
    private int dim = 8;

    /**
     * Kronecker factor dimension.
     * Range: -1 to 8
     * <ul>
     *   <li>-1: Auto-determine based on weight shape</li>
     *   <li>Lower values: Smaller file size</li>
     *   <li>Higher values: More LoRA-like behavior</li>
     * </ul>
     * Default: -1 (auto)
     */
    @Builder.Default
    private int factor = -1;

    /**
     * Scaling factor for the Kronecker product output.
     * Default: 1.0
     */
    @Builder.Default
    private double alpha = 1.0;

    /**
     * Dropout probability applied to the LoKr output.
     * Default: 0.0
     */
    @Builder.Default
    private double dropout = 0.0;

    /**
     * Whether to decompose the Kronecker factor into two smaller matrices.
     * Can further reduce parameter count.
     * Default: false
     */
    @Builder.Default
    private boolean decomposeKronecker = false;

    /**
     * Whether to use the full-matrix mode (no Kronecker decomposition).
     * Primarily for debugging/comparison.
     * Default: false
     */
    @Builder.Default
    private boolean fullMatrix = false;

    @Override
    public PeftType getPeftType() {
        return PeftType.LOKR;
    }

    @Override
    public long calculateTrainableParameters(long originalParamCount) {
        // LoKr parameter count depends on factor and decomposition settings
        // Rough estimate assuming standard configuration
        if (getTargetModules() == null || getTargetModules().isEmpty()) {
            return 0;
        }
        long estimatedPerModule = 2L * dim * 1024; // Rough estimate
        if (decomposeKronecker) {
            estimatedPerModule = (long) (estimatedPerModule * 0.5);
        }
        return getTargetModules().size() * estimatedPerModule;
    }

    @Override
    public void validate() {
        Preconditions.checkState(dim > 0, "Dimension must be positive. Got: %s", dim);
        Preconditions.checkState(factor >= -1 && factor <= 8,
            "Factor must be in [-1, 8]. Got: %s", factor);
        Preconditions.checkState(alpha > 0, "Alpha must be positive. Got: %s", alpha);
        Preconditions.checkState(dropout >= 0 && dropout < 1,
            "Dropout must be in [0, 1). Got: %s", dropout);
        Preconditions.checkState(getTargetModules() != null && !getTargetModules().isEmpty(),
            "Target modules must be specified");
    }

    @Override
    public String getSummary() {
        return String.format(
            "LokrConfig(dim=%d, factor=%d, alpha=%.2f, decompose=%s, targets=%s)",
            dim, factor, alpha, decomposeKronecker, getTargetModules());
    }

    /**
     * Create a default LoKr configuration.
     */
    public static LokrConfig defaultConfig(List<String> targetModules) {
        return LokrConfig.builder()
            .dim(8)
            .factor(-1)
            .alpha(1.0)
            .dropout(0.0)
            .targetModules(targetModules)
            .taskType(TaskType.CAUSAL_LM)
            .build();
    }
}
