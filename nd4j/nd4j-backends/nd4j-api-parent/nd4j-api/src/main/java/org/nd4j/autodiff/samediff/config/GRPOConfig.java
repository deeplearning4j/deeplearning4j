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

import lombok.Data;
import lombok.EqualsAndHashCode;
import lombok.NoArgsConstructor;
import lombok.experimental.SuperBuilder;

/**
 * Configuration for Group Relative Policy Optimization (GRPO) training.
 *
 * GRPO generates multiple completions per prompt, scores them with a reward function,
 * and uses z-score normalized advantages within each group for policy gradient updates.
 * The loss uses a clipped surrogate objective with KL penalty against a reference policy.
 *
 * Loss = -min(ratio * advantage, clip(ratio, 1-epsilon, 1+epsilon) * advantage) + klPenalty * KL
 *
 * Adam Gibson
 */
@Data
@SuperBuilder
@NoArgsConstructor
@EqualsAndHashCode(callSuper = true)
public class GRPOConfig extends RLAlignmentConfig {

    /**
     * Number of completions to generate per prompt for group-based advantage estimation.
     * Default: 8
     */
    @lombok.Builder.Default
    private int groupSize = 8;

    /**
     * Clipping epsilon for the PPO-style surrogate objective.
     * Ratio is clipped to [1 - clipEpsilon, 1 + clipEpsilon].
     * Default: 0.2
     */
    @lombok.Builder.Default
    private double clipEpsilon = 0.2;

    /**
     * KL penalty coefficient applied to the KL divergence between policy and reference.
     * Default: 0.01
     */
    @lombok.Builder.Default
    private double klPenalty = 0.01;

    /**
     * Maximum number of new tokens to generate per completion.
     * Default: 256
     */
    @lombok.Builder.Default
    private int maxNewTokens = 256;

    @Override
    public String getMethodName() {
        return "GRPO";
    }

    @Override
    public void validate() {
        super.validate();
        if (groupSize < 2) {
            throw new IllegalStateException("groupSize must be >= 2, got: " + groupSize);
        }
        if (clipEpsilon <= 0) {
            throw new IllegalStateException("clipEpsilon must be positive, got: " + clipEpsilon);
        }
        if (klPenalty < 0) {
            throw new IllegalStateException("klPenalty must be non-negative, got: " + klPenalty);
        }
        if (maxNewTokens <= 0) {
            throw new IllegalStateException("maxNewTokens must be positive, got: " + maxNewTokens);
        }
    }

    /**
     * Create a standard GRPO configuration.
     */
    public static GRPOConfig standard(String policyLogitVar, int groupSize) {
        return GRPOConfig.builder()
                .policyLogitVariable(policyLogitVar)
                .groupSize(groupSize)
                .build();
    }
}
