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
 * Configuration for AdaLoRA (Adaptive Low-Rank Adaptation).
 * <p>
 * AdaLoRA adaptively allocates the parameter budget based on the importance of
 * weight matrices and layers. Unlike standard LoRA which evenly distributes
 * parameters (fixed rank), AdaLoRA adjusts the rank of LoRA matrices by
 * considering their singular values.
 * <p>
 * Key difference from LoRA:
 * <ul>
 *   <li>LoRA: Fixed rank across all target modules</li>
 *   <li>AdaLoRA: Starts with uniform rank, prunes to target average rank</li>
 *   <li>DyLoRA: Orders rank (trains with nested ranks)</li>
 * </ul>
 * <p>
 * AdaLoRA uses SVD-based importance scoring:
 * <pre>
 * W = W₀ + PΛQ  (SVD decomposition)
 *
 * where:
 *   P, Q = orthogonal matrices
 *   Λ = diagonal matrix of singular values (importance)
 *   Pruning removes singular values below threshold
 * </pre>
 *
 * @author Adam Gibson
 * @see LoraConfig
 * @see <a href="https://arxiv.org/abs/2303.10512">AdaLoRA Paper</a>
 */
@Data
@SuperBuilder
@NoArgsConstructor
@AllArgsConstructor
@EqualsAndHashCode(callSuper = true)
public class AdaLoraConfig extends LoraConfig {

    /**
     * Initial rank for all LoRA modules before pruning.
     * Should be >= targetRank.
     * Default: 12
     */
    @Builder.Default
    private int initRank = 12;

    /**
     * Target average rank after pruning.
     * Default: 8
     */
    @Builder.Default
    private int targetRank = 8;

    /**
     * Number of warmup steps before starting to prune.
     * Default: 0
     */
    @Builder.Default
    private int warmupSteps = 0;

    /**
     * Number of steps over which to perform pruning.
     * Rank is reduced linearly from initRank to targetRank.
     * Default: 1000
     */
    @Builder.Default
    private int totalPruningSteps = 1000;

    /**
     * Regularization coefficient for singular value sparsity.
     * Higher values encourage more aggressive pruning.
     * Default: 0.0
     */
    @Builder.Default
    private double regOrthoCoef = 0.0;

    /**
     * Whether to use orthogonal regularization for P and Q.
     * Helps maintain orthogonality during training.
     * Default: true
     */
    @Builder.Default
    private boolean orthogonalRegularization = true;

    /**
     * Beta1 for the importance score moving average.
     * Default: 0.85
     */
    @Builder.Default
    private double importanceBeta = 0.85;

    /**
     * Sensitivity threshold for determining importance.
     * Singular values with sensitivity below this are pruned.
     * Default: 0.0
     */
    @Builder.Default
    private double sensitivityThreshold = 0.0;

    @Override
    public PeftType getPeftType() {
        return PeftType.ADALORA;
    }

    @Override
    public void validate() {
        // Don't call super.validate() as we use initRank instead of r
        Preconditions.checkState(initRank > 0, "Initial rank must be positive. Got: %s", initRank);
        Preconditions.checkState(targetRank > 0, "Target rank must be positive. Got: %s", targetRank);
        Preconditions.checkState(initRank >= targetRank,
            "Initial rank must be >= target rank. Got init=%s, target=%s", initRank, targetRank);
        Preconditions.checkState(getLoraAlpha() > 0, "LoRA alpha must be positive. Got: %s", getLoraAlpha());
        validateTargetModules();
        Preconditions.checkState(warmupSteps >= 0, "Warmup steps must be >= 0. Got: %s", warmupSteps);
        Preconditions.checkState(totalPruningSteps > 0, "Total pruning steps must be positive. Got: %s", totalPruningSteps);
    }

    @Override
    public String getSummary() {
        return String.format(
            "AdaLoraConfig(initRank=%d, targetRank=%d, alpha=%d, pruneSteps=%d, targets=%s)",
            initRank, targetRank, getLoraAlpha(), totalPruningSteps, getTargetModules());
    }

    /**
     * Calculate the current rank at a given training step.
     *
     * @param currentStep Current training step
     * @return The rank to use at this step
     */
    public int getCurrentRank(int currentStep) {
        if (currentStep < warmupSteps) {
            return initRank;
        }

        int stepsIntoPruning = currentStep - warmupSteps;
        if (stepsIntoPruning >= totalPruningSteps) {
            return targetRank;
        }

        // Linear interpolation
        double progress = (double) stepsIntoPruning / totalPruningSteps;
        return (int) Math.round(initRank - progress * (initRank - targetRank));
    }

    /**
     * Create a default AdaLoRA configuration.
     */
    public static AdaLoraConfig defaultConfig(List<String> targetModules) {
        return AdaLoraConfig.builder()
            .initRank(12)
            .targetRank(8)
            .loraAlpha(32)
            .loraDropout(0.0)
            .warmupSteps(0)
            .totalPruningSteps(1000)
            .orthogonalRegularization(true)
            .targetModules(targetModules)
            .taskType(TaskType.CAUSAL_LM)
            .build();
    }
}
