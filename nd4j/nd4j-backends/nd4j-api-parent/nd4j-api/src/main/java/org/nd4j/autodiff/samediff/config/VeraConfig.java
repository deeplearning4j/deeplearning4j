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
 * Configuration for VeRA (Vectorized Rank Adaptation).
 * <p>
 * VeRA is extremely parameter-efficient: it uses a single pair of shared frozen
 * random matrices A and B across all target modules, with only per-layer learned
 * scaling vectors d and b.
 *
 * <p>Mathematical formulation:</p>
 * <pre>
 * deltaW = lambda * diag(d) @ B_shared @ diag(b) @ A_shared
 *
 * where:
 *   A_shared ∈ R^{r×k} is a shared frozen random matrix (seeded)
 *   B_shared ∈ R^{d×r} is a shared frozen random matrix (seeded)
 *   d ∈ R^d is a per-layer learned scaling vector
 *   b ∈ R^r is a per-layer learned scaling vector
 *   lambda is a global scaling factor
 * </pre>
 *
 * <p>Trainable parameters per layer: d + r (vs r*(d+k) for LoRA)</p>
 *
 * Adam Gibson
 * @see PeftConfig
 * @see <a href="https://arxiv.org/abs/2310.11454">VeRA Paper</a>
 */
@Data
@SuperBuilder
@NoArgsConstructor
@AllArgsConstructor
@EqualsAndHashCode(callSuper = true)
public class VeraConfig extends PeftConfig {

    /**
     * Rank of the shared random matrices.
     * Default: 256 (VeRA uses higher ranks since only scaling vectors are trained)
     */
    @Builder.Default
    private int r = 256;

    /**
     * Random seed for generating the shared frozen matrices.
     * Using the same seed ensures reproducibility.
     * Default: 42
     */
    @Builder.Default
    private long sharedSeed = 42;

    /**
     * Global scaling factor (lambda) for VeRA updates.
     * Default: 1.0
     */
    @Builder.Default
    private double lambdaScaling = 1.0;

    @Override
    public PeftType getPeftType() {
        return PeftType.VERA;
    }

    @Override
    public long calculateTrainableParameters(long originalParamCount) {
        if (targetModules == null || targetModules.isEmpty()) return 0;
        // Per module: d (outFeatures) + r (rank) parameters
        // Rough estimate assuming average dimension of 1024
        return targetModules.size() * (1024L + r);
    }

    @Override
    public void validate() {
        Preconditions.checkState(r > 0, "Rank (r) must be positive. Got: %s", r);
        Preconditions.checkState(lambdaScaling > 0, "Lambda scaling must be positive. Got: %s", lambdaScaling);
        Preconditions.checkState(targetModules != null && !targetModules.isEmpty(),
            "Target modules must be specified for VeRA");
    }

    @Override
    public String getSummary() {
        return String.format("VeraConfig(r=%d, seed=%d, lambda=%.4f, targets=%s)",
            r, sharedSeed, lambdaScaling, targetModules);
    }
}
